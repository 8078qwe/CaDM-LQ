import torch
from torch import nn
import torch.nn.functional as F
import math
import copy
from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)
import numpy as np
import clip
from datasets.hico_text_label import hico_text_label, hico_obj_text_label, hico_unseen_index
from datasets.vcoco_text_label import vcoco_hoi_text_label, vcoco_obj_text_label

from .backbone import build_backbone
from .matcher import build_matcher
from .deformable_transformer import build_deformable_transformer
from .utils import sigmoid_focal_loss, MLP

import os
from typing import Tuple
import sys
from omegaconf import OmegaConf
from einops import rearrange, repeat

from ldm.util import instantiate_from_config
from SD_Extractor import UNetWrapper, TextAdapter


def _sigmoid(x):
    y = torch.clamp(x.sigmoid(), min=1e-4, max=1 - 1e-4)
    return y


class ImportanceUpdater(nn.Module):
    """Lightweight CNN/identity updater for pair-importance matrices."""

    def __init__(self, mapper: str = "identity"):
        super().__init__()
        self.mapper = mapper
        if mapper == "identity" or mapper is None:
            self.net = None
        elif mapper == "conv_tiny":
            self.net = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 1, kernel_size=1),
            )
        elif mapper == "conv_small":
            self.net = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 1, kernel_size=1),
            )
        else:
            raise ValueError(f"Unknown pair_mapper: {mapper}")

    def forward(self, importance: torch.Tensor) -> torch.Tensor:
        if self.net is None:
            return importance
        x = importance.unsqueeze(1)  # [B, 1, N, N]
        x = self.net(x)
        return x.squeeze(1)


class HumanObjectPairLearner(nn.Module):
    """Human-object pair learner (Pair Importance + TopK pairing + Pair Query generation).

    Minimal implementation inspired by PairNet's pairing head. It produces Top-K (human_idx, obj_idx)
    pairs and corresponding pair-query features.

    Notes:
        - We treat DETR-style queries as (num_pairs, 2) where index 0 is 'human/subject' and index 1 is 'object'.
        - We do NOT implement the paper's GCN here; we provide an optional CNN updater on the importance matrix.
    """

    def __init__(self, hidden_dim: int, num_obj_query: int, num_pair_query: int, mapper: str = "identity"):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_obj_query = num_obj_query
        self.num_pair_query = num_pair_query

        self.sub_query_update = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.obj_query_update = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.update_importance = ImportanceUpdater(mapper)

        self.pair_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

    @staticmethod
    def _split_sub_obj(hs_layer: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, Q, C = hs_layer.shape
        assert Q % 2 == 0, "Expected 2*N queries (subject/object per pair)."
        N = Q // 2
        hs_pair = hs_layer.view(B, N, 2, C)
        return hs_pair[:, :, 0, :], hs_pair[:, :, 1, :]

    def select_topk_indices(self, hs_last: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sub_feat, obj_feat = self._split_sub_obj(hs_last)  # [B,N,C], [B,N,C]

        sub_emb = self.sub_query_update(sub_feat)
        obj_emb = self.obj_query_update(obj_feat)
        sub_emb = sub_emb / (sub_emb.norm(dim=-1, keepdim=True) + 1e-6)
        obj_emb = obj_emb / (obj_emb.norm(dim=-1, keepdim=True) + 1e-6)

        importance = torch.matmul(sub_emb, obj_emb.transpose(1, 2))  # [B,N,N]
        importance = self.update_importance(importance)

        B, N, _ = importance.shape
        K = min(self.num_pair_query, N * N)
        flat = importance.flatten(1)  # [B, N*N]
        _, topk_idx = torch.topk(flat, K, dim=1, largest=True, sorted=False)  # [B,K]
        sub_pos = topk_idx // N
        obj_pos = topk_idx % N
        return sub_pos.long(), obj_pos.long(), importance

    def build_pair_queries(self, hs_layer: torch.Tensor, sub_pos: torch.Tensor, obj_pos: torch.Tensor) -> torch.Tensor:
        sub_feat, obj_feat = self._split_sub_obj(hs_layer)  # [B,N,C], [B,N,C]
        B, N, C = sub_feat.shape
        K = sub_pos.shape[1]

        sub_idx = sub_pos.unsqueeze(-1).expand(B, K, C)
        obj_idx = obj_pos.unsqueeze(-1).expand(B, K, C)

        sel_sub = torch.gather(sub_feat, 1, sub_idx)
        sel_obj = torch.gather(obj_feat, 1, obj_idx)

        pair_feat = torch.cat([sel_sub, sel_obj], dim=-1)  # [B,K,2C]
        return self.pair_mlp(pair_feat)  # [B,K,C]


class PairSpatialEncoding(nn.Module):
    """Encode subject/object box pairs into a spatial prior embedding.

    Input boxes are expected in normalized cxcywh format.
    We use a lightweight MLP over concatenated (sub_box, obj_box) -> hidden_dim.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(8, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, sub_boxes: torch.Tensor, obj_boxes: torch.Tensor) -> torch.Tensor:
        # sub_boxes/obj_boxes: [B, K, 4]
        x = torch.cat([sub_boxes, obj_boxes], dim=-1)  # [B, K, 8]
        return self.mlp(x)  # [B, K, C]


class LocalityAwareDecoderLayer(nn.Module):
    """A single layer of locality-aware interaction decoding.

    Implements a practical variant of Eq.(13) in the paper:
      - Inject pairwise spatial prior into self-attention (Q + E_sp)
      - Cross-attend to image features with positional encoding (F + E_ps)
      - Residual + LayerNorm + FFN
    """

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.ReLU(inplace=True)

    def forward(self, q: torch.Tensor, mem: torch.Tensor, mem_pos: torch.Tensor, sp: torch.Tensor) -> torch.Tensor:
        # q: [B, K, C], mem: [B, L, C], mem_pos: [B, L, C] (or None), sp: [B, K, C]
        q_sp = q + sp
        attn_out, _ = self.self_attn(q_sp, q_sp, q_sp, need_weights=False)
        q = self.norm1(q + self.dropout1(attn_out))

        k = mem if mem_pos is None else (mem + mem_pos)

        # Cross-attention with spatial-prior token concatenation (practical approximation of CAT(Q, E_sp))
        q_cat = torch.cat([q, sp], dim=1)  # [B, 2K, C]
        ca_out, _ = self.cross_attn(q_cat, k, mem, need_weights=False)
        ca_out = ca_out[:, :q.shape[1], :]  # keep only the first K (query tokens)
        q = self.norm2(q + self.dropout2(ca_out))

        ffn_out = self.linear2(self.dropout(self.activation(self.linear1(q))))
        q = self.norm3(q + self.dropout3(ffn_out))
        return q


class LocalityAwareInteractionDecoder(nn.Module):
    """Stacked locality-aware decoder layers."""

    def __init__(self, d_model: int, nhead: int, num_layers: int = 2, dim_feedforward: int = 1024,
                 dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            LocalityAwareDecoderLayer(d_model, nhead, dim_feedforward=dim_feedforward, dropout=dropout)
            for _ in range(num_layers)
        ])

    def forward(self, q: torch.Tensor, mem: torch.Tensor, mem_pos: torch.Tensor, sp: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            q = layer(q, mem, mem_pos, sp)
        return q


class DiffHOI_L(nn.Module):
    """ This is the Cross-Attention Detector module that performs object detection """

    def __init__(self, backbone, transformer, num_classes, num_queries,
                 aux_loss=False, iter_update=False,
                 query_dim=2,
                 random_refpoints_xy=False,
                 fix_refpoints_hw=-1,
                 num_feature_levels=1,
                 nheads=8,
                 # two stage
                 two_stage_type='no',  # ['no', 'standard']
                 two_stage_add_query_num=0,
                 dec_pred_class_embed_share=True,
                 dec_pred_bbox_embed_share=True,
                 two_stage_class_embed_share=True,
                 two_stage_bbox_embed_share=True,
                 decoder_sa_type='sa',
                 num_patterns=0,
                 dn_number=100,
                 dn_box_noise_scale=0.4,
                 dn_label_noise_ratio=0.5,
                 dn_labelbook_size=100,
                 args=None,
                 unet_config=dict()
                 ):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         Conditional DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.

            fix_refpoints_hw: -1(default): learn w and h for each box seperately
                                >0 : given fixed number
                                -2 : learn a shared w and h
        """
        super().__init__()

        config_path = os.environ.get("SD_Config")
        ckpt_path = os.environ.get("SD_ckpt")
        config = OmegaConf.load(config_path)
        config.model.params.ckpt_path = ckpt_path
        config.model.params.cond_stage_config.target = 'ldm.modules.encoders.modules.AbstractEncoder'
        sd_model = instantiate_from_config(config.model)
        self.encoder_vq = sd_model.first_stage_model
        self.unet = UNetWrapper(sd_model.model, **unet_config)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.unet.to(device)
        # self.unet.eval()
        sd_model.model = None
        sd_model.first_stage_model = None
        del sd_model.cond_stage_model
        del self.encoder_vq.decoder
        self.sd_model = sd_model
        self.text_adapter = nn.Linear(args.clip_embed_dim, 768)
        self.clip_adapter = nn.Linear(args.clip_embed_dim, args.clip_embed_dim)

        self.num_queries = num_queries
        self.transformer = transformer
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim = transformer.d_model
        self.num_feature_levels = num_feature_levels
        self.nheads = nheads
        self.label_enc = nn.Embedding(dn_labelbook_size + 1, hidden_dim)
        self.args = args

        # ---------------- Human-object Pair Learner (Stage II) ----------------
        self.use_pair_learner = bool(getattr(args, "use_pair_learner", True))
        self.num_pair_queries = int(getattr(args, "num_pair_queries", max(1, num_queries // 2)))
        self.pair_mapper = getattr(args, "pair_mapper", "identity")
        self.return_pair_info = bool(getattr(args, "return_pair_info", False))
        self.pair_learner = HumanObjectPairLearner(
            hidden_dim=hidden_dim,
            num_obj_query=max(1, num_queries // 2),
            num_pair_query=self.num_pair_queries,
            mapper=self.pair_mapper,
        )
        # ---------------- Locality-aware Interaction Decoder (Stage II) ----------------
        self.use_locality_decoder = bool(getattr(args, "use_locality_decoder", True))
        self.interaction_decoder_layers = int(getattr(args, "interaction_decoder_layers", 2))
        self.interaction_decoder_heads = int(getattr(args, "interaction_decoder_heads", 8))
        self.interaction_decoder_ffn_dim = int(getattr(args, "interaction_decoder_ffn_dim", 1024))
        self.interaction_decoder_dropout = float(getattr(args, "interaction_decoder_dropout", 0.1))

        self.pair_spatial_encoder = PairSpatialEncoding(hidden_dim)
        self.locality_decoder = LocalityAwareInteractionDecoder(
            d_model=hidden_dim,
            nhead=self.interaction_decoder_heads,
            num_layers=self.interaction_decoder_layers,
            dim_feedforward=self.interaction_decoder_ffn_dim,
            dropout=self.interaction_decoder_dropout,
        )
        # setting query dim
        self.query_dim = query_dim
        assert query_dim == 4
        self.random_refpoints_xy = random_refpoints_xy
        self.fix_refpoints_hw = fix_refpoints_hw

        # for dn training
        self.num_patterns = num_patterns
        self.dn_number = dn_number
        self.dn_box_noise_scale = dn_box_noise_scale
        self.dn_label_noise_ratio = dn_label_noise_ratio
        self.dn_labelbook_size = dn_labelbook_size

        # prepare input projection layers
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.num_channels)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            assert two_stage_type == 'no', "two_stage_type should be no if num_feature_levels=1 !!!"
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[-1], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])

        self.backbone = backbone
        self.aux_loss = aux_loss
        self.box_pred_damping = box_pred_damping = None

        self.iter_update = iter_update
        assert iter_update, "Why not iter_update?"

        # prepare pred layers
        self.dec_pred_class_embed_share = dec_pred_class_embed_share
        self.dec_pred_bbox_embed_share = dec_pred_bbox_embed_share
        # prepare class & box embed
        _class_embed = nn.Linear(hidden_dim, num_classes)
        _bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        # init the two embed layers
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        _class_embed.bias.data = torch.ones(self.num_classes) * bias_value
        nn.init.constant_(_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_bbox_embed.layers[-1].bias.data, 0)

        if dec_pred_bbox_embed_share:
            box_embed_layerlist = [_bbox_embed for i in range(transformer.num_decoder_layers - 3)]
        else:
            box_embed_layerlist = [copy.deepcopy(_bbox_embed) for i in range(transformer.num_decoder_layers - 3)]
        if dec_pred_class_embed_share:
            class_embed_layerlist = [_class_embed for i in range(transformer.num_decoder_layers - 3)]
        else:
            class_embed_layerlist = [copy.deepcopy(_class_embed) for i in range(transformer.num_decoder_layers - 3)]

        self.bbox_embed = nn.ModuleList(box_embed_layerlist)
        self.class_embed = nn.ModuleList(class_embed_layerlist)
        self.transformer.decoder.bbox_embed = self.bbox_embed
        self.transformer.decoder.class_embed = self.class_embed

        # two stage
        self.two_stage_type = two_stage_type
        self.two_stage_add_query_num = two_stage_add_query_num
        assert two_stage_type in ['no', 'standard'], "unknown param {} of two_stage_type".format(two_stage_type)
        if two_stage_type != 'no':
            if two_stage_bbox_embed_share:
                assert dec_pred_class_embed_share and dec_pred_bbox_embed_share
                self.transformer.enc_out_bbox_embed = _bbox_embed
            else:
                self.transformer.enc_out_bbox_embed = copy.deepcopy(_bbox_embed)

            if two_stage_class_embed_share:
                assert dec_pred_class_embed_share and dec_pred_bbox_embed_share
                self.transformer.enc_out_class_embed = _class_embed
            else:
                self.transformer.enc_out_class_embed = copy.deepcopy(_class_embed)

            self.refpoint_embed = None
            if self.two_stage_add_query_num > 0:
                self.init_ref_points(two_stage_add_query_num)

        self.decoder_sa_type = decoder_sa_type
        assert decoder_sa_type in ['sa', 'ca_label', 'ca_content']
        if decoder_sa_type == 'ca_label':
            self.label_embedding = nn.Embedding(num_classes, hidden_dim)
            for layer in self.transformer.decoder.layers:
                layer.label_embedding = self.label_embedding
        else:
            for layer in self.transformer.decoder.layers:
                layer.label_embedding = None
            self.label_embedding = None

        self.dec_layers = 3

        self._reset_parameters()

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.obj_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        if self.args.dataset_file == 'hico':
            hoi_text_label = hico_text_label
            obj_text_label = hico_obj_text_label
            unseen_index = hico_unseen_index
        elif self.args.dataset_file == 'vcoco':
            hoi_text_label = vcoco_hoi_text_label
            obj_text_label = vcoco_obj_text_label
            unseen_index = None

        clip_label, obj_clip_label, v_linear_proj_weight, hoi_text, obj_text, train_clip_label = \
            self.init_classifier_with_CLIP(hoi_text_label, obj_text_label, unseen_index)
        num_obj_classes = len(obj_text) - 1  # del nothing

        self.hoi_class_fc = nn.Sequential(
            nn.Linear(hidden_dim, args.clip_embed_dim),
            nn.LayerNorm(args.clip_embed_dim),
        )

        if args.with_clip_label:
            self.visual_projection = nn.Linear(args.clip_embed_dim, len(hoi_text))
            self.visual_projection.weight.data = train_clip_label / train_clip_label.norm(dim=-1, keepdim=True)
            # ---- Soft-label distillation (Teacher CaDiff/CLIP) ----
            # Normalized HOI text embeddings used by the teacher to produce soft labels.
            self.register_buffer(
                "hoi_text_emb",
                train_clip_label / (train_clip_label.norm(dim=-1, keepdim=True) + 1e-6)
            )
            # KD controls (defaults keep behavior unchanged)
            self.kd_eps = getattr(args, "kd_eps", 1e-6)
            self.use_cadiff_kd = getattr(args, "use_cadiff_kd", False)
            if self.args.dataset_file == 'hico' and self.args.zero_shot_type != 'default':
                self.eval_visual_projection = nn.Linear(args.clip_embed_dim, 600)
                self.eval_visual_projection.weight.data = clip_label / clip_label.norm(dim=-1, keepdim=True)

        if args.with_obj_clip_label:
            self.obj_class_fc = nn.Sequential(
                nn.Linear(hidden_dim, args.clip_embed_dim),
                nn.LayerNorm(args.clip_embed_dim),
            )
            self.obj_visual_projection = nn.Linear(args.clip_embed_dim, num_obj_classes + 1)
            self.obj_visual_projection.weight.data = obj_clip_label / obj_clip_label.norm(dim=-1, keepdim=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, _ = clip.load(args.clip_model, device=device)

        self.image_adapter_4 = nn.Conv2d(1280, hidden_dim, kernel_size=1)

        self.image_adapter_3 = nn.Conv2d(1280, hidden_dim, kernel_size=1)

        self.image_adapter_2 = nn.Conv2d(640, hidden_dim, kernel_size=1)

        self.image_adapter_1 = nn.Conv2d(320, hidden_dim, kernel_size=1)

        self.clip_adapter = nn.Linear(args.clip_embed_dim, args.clip_embed_dim)
        # import pdb
        # pdb.set_trace()

        self.freeze()

    def freeze(self):
        for param in self.class_embed.parameters():
            param.requires_grad = False

        for param in self.transformer.enc_out_bbox_embed.parameters():
            param.requires_grad = False

    def _reset_parameters(self):
        # init input_proj
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

    def init_classifier_with_CLIP(self, hoi_text_label, obj_text_label, unseen_index):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        text_inputs = torch.cat([clip.tokenize(hoi_text_label[id]) for id in hoi_text_label.keys()])
        if self.args.del_unseen and unseen_index is not None:
            hoi_text_label_del = {}
            unseen_index_list = unseen_index.get(self.args.zero_shot_type, [])
            for idx, k in enumerate(hoi_text_label.keys()):
                if idx in unseen_index_list:
                    continue
                else:
                    hoi_text_label_del[k] = hoi_text_label[k]
        else:
            hoi_text_label_del = hoi_text_label.copy()
        text_inputs_del = torch.cat(
            [clip.tokenize(hoi_text_label[id]) for id in hoi_text_label_del.keys()])

        obj_text_inputs = torch.cat([clip.tokenize(obj_text[1]) for obj_text in obj_text_label])
        clip_model, preprocess = clip.load(self.args.clip_model, device=device)
        with torch.no_grad():
            text_embedding = clip_model.encode_text(text_inputs.to(device))
            text_embedding_del = clip_model.encode_text(text_inputs_del.to(device))
            obj_text_embedding = clip_model.encode_text(obj_text_inputs.to(device))
            v_linear_proj_weight = clip_model.visual.proj.detach()

        del clip_model

        return text_embedding.float(), obj_text_embedding.float(), v_linear_proj_weight.float(), \
            hoi_text_label_del, obj_text_inputs, text_embedding_del.float()

    def init_ref_points(self, use_num_queries):
        self.refpoint_embed = nn.Embedding(use_num_queries, self.query_dim)
        if self.random_refpoints_xy:
            self.refpoint_embed.weight.data[:, :2].uniform_(0, 1)
            self.refpoint_embed.weight.data[:, :2] = inverse_sigmoid(self.refpoint_embed.weight.data[:, :2])
            self.refpoint_embed.weight.data[:, :2].requires_grad = False

        if self.fix_refpoints_hw > 0:
            print("fix_refpoints_hw: {}".format(self.fix_refpoints_hw))
            assert self.random_refpoints_xy
            self.refpoint_embed.weight.data[:, 2:] = self.fix_refpoints_hw
            self.refpoint_embed.weight.data[:, 2:] = inverse_sigmoid(self.refpoint_embed.weight.data[:, 2:])
            self.refpoint_embed.weight.data[:, 2:].requires_grad = False
        elif int(self.fix_refpoints_hw) == -1:
            pass
        elif int(self.fix_refpoints_hw) == -2:
            print('learn a shared h and w')
            assert self.random_refpoints_xy
            self.refpoint_embed = nn.Embedding(use_num_queries, 2)
            self.refpoint_embed.weight.data[:, :2].uniform_(0, 1)
            self.refpoint_embed.weight.data[:, :2] = inverse_sigmoid(self.refpoint_embed.weight.data[:, :2])
            self.refpoint_embed.weight.data[:, :2].requires_grad = False
            self.hw_embed = nn.Embedding(1, 1)
        else:
            raise NotImplementedError('Unknown fix_refpoints_hw {}'.format(self.fix_refpoints_hw))

    def extract_feat(self, img, targets):
        """Extract features from images."""
        target_clip_inputs = torch.cat([t['clip_inputs'].unsqueeze(0) for t in targets])
        target_sd_inputs = torch.cat([t['sd_inputs'].unsqueeze(0) for t in targets])
        with torch.no_grad():
            clip_feats = self.clip_model.encode_image(target_clip_inputs)
        clip_feats = self.text_adapter(clip_feats.unsqueeze(1).float())
        t = torch.zeros((img.shape[0],), device=img.device).long()
        with torch.no_grad():
            latents = self.encoder_vq.encode(target_sd_inputs)
            latents = latents.mode().detach()
            outs = self.unet(latents, t, c_crossattn=[clip_feats])
        return outs

    def forward(self, samples: NestedTensor, targets, is_training=True):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x num_classes]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, width, height). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, poss = self.backbone(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                poss.append(pos_l)

        input_query_bbox = input_query_label = attn_mask = dn_meta = None

        sd_feat = self.extract_feat(samples.decompose()[0], targets)
        sd_decoder_4 = self.image_adapter_4(sd_feat[-1])
        sd_decoder_3 = self.image_adapter_3(sd_feat[-2])
        sd_decoder_2 = self.image_adapter_2(sd_feat[-3])
        sd_decoder_1 = self.image_adapter_1(sd_feat[-4])
        sd_decoder_4 = torch.nn.functional.interpolate(sd_decoder_4, size=[srcs[-1].shape[-2], srcs[-1].shape[-1]])
        sd_decoder_3 = torch.nn.functional.interpolate(sd_decoder_3, size=[srcs[-2].shape[-2], srcs[-2].shape[-1]])
        sd_decoder_2 = torch.nn.functional.interpolate(sd_decoder_2, size=[srcs[-3].shape[-2], srcs[-3].shape[-1]])
        sd_decoder_1 = torch.nn.functional.interpolate(sd_decoder_1, size=[srcs[-4].shape[-2], srcs[-4].shape[-1]])
        sd_decoder = [sd_decoder_1, sd_decoder_2, sd_decoder_3, sd_decoder_4]

        hs, hs_interaction, reference, hs_enc, ref_enc, init_box_proposal = self.transformer(srcs, masks,
                                                                                             input_query_bbox, poss,
                                                                                             input_query_label,
                                                                                             sd_decoder, attn_mask)
        # In case num object=0
        hs[0] += self.label_enc.weight[0, 0] * 0.0

        # Stack decoder outputs for downstream heads
        hs_stack = torch.stack(hs)  # [dec, B, 2*N, C]

        # Build interaction (pair) queries
        pair_sub_pos = pair_obj_pos = pair_importance = None
        if self.use_pair_learner:
            pair_sub_pos, pair_obj_pos, pair_importance = self.pair_learner.select_topk_indices(hs[-1])
            pair_hs_layers = [self.pair_learner.build_pair_queries(layer_hs, pair_sub_pos, pair_obj_pos) for layer_hs in
                              hs]
            hs_interaction = torch.stack(pair_hs_layers)  # [dec, B, K, C]
        else:
            hs_interaction = torch.stack(hs_interaction)  # [dec, B, Q_inter, C]
        # deformable-detr-like anchor update
        # reference_before_sigmoid = inverse_sigmoid(reference[:-1]) # n_dec, bs, nq, 4
        outputs_coord_list = []
        for dec_lid, (layer_ref_sig, layer_bbox_embed, layer_hs) in enumerate(zip(reference[:-1], self.bbox_embed, hs)):
            layer_delta_unsig = layer_bbox_embed(layer_hs)
            layer_outputs_unsig = layer_delta_unsig + inverse_sigmoid(layer_ref_sig)
            layer_outputs_unsig = layer_outputs_unsig.sigmoid()
            outputs_coord_list.append(layer_outputs_unsig)
        outputs_coord_list = torch.stack(outputs_coord_list)

        # clip_feat
        target_clip_inputs = torch.cat([t['clip_inputs'].unsqueeze(0) for t in targets])
        with torch.no_grad():
            target_clip_feats = self.clip_model.encode_image(target_clip_inputs)
        target_clip_feats = self.clip_adapter(target_clip_feats.float())

        if self.args.with_obj_clip_label:
            obj_logit_scale = self.obj_logit_scale.exp()
            o_hs = self.obj_class_fc(hs_stack)
            o_hs = o_hs / o_hs.norm(dim=-1, keepdim=True)
            outputs_obj_class = obj_logit_scale * self.obj_visual_projection(o_hs)

        # HOI logits will be computed after (optional) locality-aware decoding, once pair boxes are gathered.
        outputs_hoi_class = None

        outputs_coord_list = outputs_coord_list.view(outputs_coord_list.shape[0], outputs_coord_list.shape[1],
                                                     int(self.num_queries / 2), 2, 4)

        outputs_class = outputs_obj_class.view(outputs_obj_class.shape[0], outputs_obj_class.shape[1],
                                               int(self.num_queries / 2), 2, outputs_obj_class.shape[-1])

        # -------- Pair selection: gather boxes / object logits to align with selected pairs --------
        outputs_sub_coord = outputs_coord_list[:, :, :, 0, :]  # [dec, B, N, 4]
        outputs_obj_coord = outputs_coord_list[:, :, :, 1, :]  # [dec, B, N, 4]
        outputs_obj_logits = outputs_class[:, :, :, 1, :]  # [dec, B, N, num_obj+1]

        if self.use_pair_learner and pair_sub_pos is not None and pair_obj_pos is not None:
            dec, B, N, _ = outputs_sub_coord.shape
            K = pair_sub_pos.shape[1]

            sub_idx = pair_sub_pos.unsqueeze(0).unsqueeze(-1).expand(dec, B, K, 4)
            obj_idx_box = pair_obj_pos.unsqueeze(0).unsqueeze(-1).expand(dec, B, K, 4)
            obj_idx_logit = pair_obj_pos.unsqueeze(0).unsqueeze(-1).expand(dec, B, K, outputs_obj_logits.shape[-1])

            outputs_sub_coord = torch.gather(outputs_sub_coord, 2, sub_idx)
            outputs_obj_coord = torch.gather(outputs_obj_coord, 2, obj_idx_box)
            outputs_obj_logits = torch.gather(outputs_obj_logits, 2, obj_idx_logit)

        # (Optional) expose pair info for debugging / analysis
        if self.return_pair_info and self.use_pair_learner and pair_importance is not None:
            out_pair_info = {
                "pair_sub_pos": pair_sub_pos,
                "pair_obj_pos": pair_obj_pos,
                "pair_importance": pair_importance,
            }
        else:
            out_pair_info = None
        # -------- Pairwise spatial prior + Locality-aware Interaction Decoder --------
        # We refine interaction queries by injecting per-pair spatial priors at each decoding layer.
        if self.args.with_clip_label:
            if self.use_locality_decoder:
                # Use the last feature level as image memory (B, HW, C)
                mem = srcs[-1].flatten(2).transpose(1, 2)
                mem_pos = poss[-1].flatten(2).transpose(1, 2) if (poss is not None and len(poss) > 0) else None

                # Build pairwise spatial priors from gathered subject/object boxes
                dec, B, K, _ = outputs_sub_coord.shape
                sub_flat = outputs_sub_coord.reshape(dec * B, K, 4)
                obj_flat = outputs_obj_coord.reshape(dec * B, K, 4)
                sp_flat = self.pair_spatial_encoder(sub_flat, obj_flat)  # [dec*B, K, C]
                sp = sp_flat.reshape(dec, B, K, self.hidden_dim)

                # Apply locality-aware decoding per layer (keeps aux outputs consistent)
                refined_layers = []
                for lid in range(hs_interaction.shape[0]):
                    refined_layers.append(self.locality_decoder(hs_interaction[lid], mem, mem_pos, sp[lid]))
                hs_interaction = torch.stack(refined_layers)  # [dec, B, K, C]

            # HOI classification head (CLIP-aligned)
            logit_scale = self.logit_scale.exp()
            inter_hs = self.hoi_class_fc(hs_interaction)
            inter_hs = inter_hs + target_clip_feats[None, :, None, :].repeat(inter_hs.shape[0], 1, inter_hs.shape[2], 1)
            inter_hs = inter_hs / inter_hs.norm(dim=-1, keepdim=True)

            if self.args.dataset_file == 'hico' and self.args.zero_shot_type != 'default' and (
                    self.args.eval or not is_training):
                outputs_hoi_class = logit_scale * self.eval_visual_projection(inter_hs)
            else:
                outputs_hoi_class = logit_scale * self.visual_projection(inter_hs)

        # ---- Teacher soft labels (mechanism verification: CLIP image embedding teacher; can be replaced by CaDiff) ----
        teacher_hoi_probs = None
        if getattr(self, "use_cadiff_kd", False):
            with torch.no_grad():
                img_emb = target_clip_feats / (target_clip_feats.norm(dim=-1, keepdim=True) + 1e-6)  # [bs, clip_dim]
                txt_emb = self.hoi_text_emb  # already normalized buffer [num_hoi, clip_dim]
                t_logit_scale = self.logit_scale.exp()
                t_logits = t_logit_scale * (img_emb @ txt_emb.t())  # [bs, num_hoi]
                t_probs = torch.sigmoid(t_logits)  # [bs, num_hoi]
                # SoftMask in the paper uses S_dif != 0. We sparsify small probs to 0.
                t_probs = torch.where(t_probs > self.kd_eps, t_probs, torch.zeros_like(t_probs))
            nq = outputs_hoi_class[-1].shape[1]
            teacher_hoi_probs = t_probs[:, None, :].expand(-1, nq, -1).contiguous()  # [bs, nq, num_hoi]

        out = {'pred_hoi_logits': outputs_hoi_class[-1], 'pred_obj_logits': outputs_obj_logits[-1],
               'pred_sub_boxes': outputs_sub_coord[-1], 'pred_obj_boxes': outputs_obj_coord[-1], }
        if teacher_hoi_probs is not None:
            out['teacher_hoi_probs'] = teacher_hoi_probs
        if out_pair_info is not None:
            out['pair_info'] = out_pair_info

        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss_triplet(outputs_hoi_class, outputs_obj_logits, outputs_sub_coord,
                                                            outputs_obj_coord)
            # Propagate teacher probs to auxiliary outputs (optional, but keeps KD consistent across layers)
            if teacher_hoi_probs is not None:
                for _aux in out['aux_outputs']:
                    _aux['teacher_hoi_probs'] = teacher_hoi_probs

        out['dn_meta'] = dn_meta

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    @torch.jit.unused
    def _set_aux_loss_triplet(self, outputs_hoi_class, outputs_obj_class,
                              outputs_sub_coord, outputs_obj_coord, outputs_inter_hs=None):

        aux_outputs = {'pred_hoi_logits': outputs_hoi_class[-self.dec_layers: -1],
                       'pred_obj_logits': outputs_obj_class[-self.dec_layers: -1],
                       'pred_sub_boxes': outputs_sub_coord[-self.dec_layers: -1],
                       'pred_obj_boxes': outputs_obj_coord[-self.dec_layers: -1]}
        if outputs_inter_hs is not None:
            aux_outputs['inter_memory'] = outputs_inter_hs[-self.dec_layers: -1]
        outputs_auxes = []
        for i in range(self.dec_layers - 1):
            output_aux = {}
            for aux_key in aux_outputs.keys():
                output_aux[aux_key] = aux_outputs[aux_key][i]
            outputs_auxes.append(output_aux)
        return outputs_auxes


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class SetCriterionHOI(nn.Module):

    def __init__(self, num_obj_classes, num_queries, num_verb_classes, matcher, weight_dict, eos_coef, losses, args):
        super().__init__()

        self.num_obj_classes = num_obj_classes
        self.num_queries = num_queries
        self.num_verb_classes = num_verb_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_obj_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, _ = clip.load(args.clip_model, device=device)
        self.alpha = args.alpha
        # ---- Soft-label distillation (Teacher CaDiff/CLIP) ----
        # If outputs contains 'teacher_hoi_probs', we will split HOI loss into hard/soft parts.
        self.lambda_soft = getattr(args, "lambda_soft", 0.0)
        self.kd_eps = getattr(args, "kd_eps", 1e-6)
        self.kd_gamma = getattr(args, "kd_gamma", 2.0)

    def loss_obj_labels(self, outputs, targets, indices, num_interactions, log=True):
        assert 'pred_obj_logits' in outputs
        src_logits = outputs['pred_obj_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['obj_labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_obj_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_obj_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_obj_ce': loss_obj_ce}

        if log:
            losses['obj_class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_obj_cardinality(self, outputs, targets, indices, num_interactions):
        pred_logits = outputs['pred_obj_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v['obj_labels']) for v in targets], device=device)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'obj_cardinality_error': card_err}
        return losses

    def loss_verb_labels(self, outputs, targets, indices, num_interactions):
        assert 'pred_verb_logits' in outputs
        src_logits = outputs['pred_verb_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['verb_labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.zeros_like(src_logits)
        target_classes[idx] = target_classes_o

        src_logits = src_logits.sigmoid()
        loss_verb_ce = self._neg_loss(src_logits, target_classes, weights=None, alpha=self.alpha)
        losses = {'loss_verb_ce': loss_verb_ce}
        return losses

    def loss_hoi_labels(self, outputs, targets, indices, num_interactions, topk=5):
        assert 'pred_hoi_logits' in outputs
        src_logits = outputs['pred_hoi_logits']  # [bs, nq, num_hoi]

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['hoi_labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.zeros_like(src_logits)
        target_classes[idx] = target_classes_o  # hard GT in {0,1}

        # student prob (clamped for log stability)
        src_prob = _sigmoid(src_logits)

        teacher_probs = outputs.get('teacher_hoi_probs', None)
        use_kd = (teacher_probs is not None) and (self.lambda_soft > 0)

        if not use_kd:
            loss_hoi = self._neg_loss(src_prob, target_classes, weights=None, alpha=self.alpha)
            losses = {'loss_hoi_labels': loss_hoi}
        else:
            with torch.no_grad():
                t = teacher_probs.detach()
                # SoftMask: 1 if teacher != 0 and GT != 1, else 0
                softmask = ((t > self.kd_eps).float()) * (1.0 - target_classes)
                hardmask = 1.0 - softmask

            loss_hard = self._neg_loss_masked(src_prob, target_classes, hardmask, alpha=self.alpha)
            loss_soft = self._sigmoid_focal_soft_with_logits_masked(
                src_logits, t, softmask, alpha=self.alpha, gamma=self.kd_gamma
            )

            loss_hoi = loss_hard + self.lambda_soft * loss_soft

            losses = {
                'loss_hoi_labels': loss_hoi,
                'loss_hoi_kd_hard': loss_hard,
                'loss_hoi_kd_soft': loss_soft,
            }

        # keep the original top-k error metric
        _, pred = src_prob[idx].topk(topk, 1, True, True)
        acc = 0.0
        for tid, target in enumerate(target_classes_o):
            tgt_idx = torch.where(target == 1)[0]
            if len(tgt_idx) == 0:
                continue
            acc_pred = 0.0
            for tgt_rel in tgt_idx:
                acc_pred += (tgt_rel in pred[tid])
            acc += acc_pred / len(tgt_idx)
        rel_labels_error = 100 - 100 * acc / max(len(target_classes_o), 1)
        losses['hoi_class_error'] = torch.from_numpy(np.array(rel_labels_error)).to(src_prob.device).float()
        return losses

    def loss_sub_obj_boxes(self, outputs, targets, indices, num_interactions):
        assert 'pred_sub_boxes' in outputs and 'pred_obj_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_sub_boxes = outputs['pred_sub_boxes'][idx]
        src_obj_boxes = outputs['pred_obj_boxes'][idx]
        target_sub_boxes = torch.cat([t['sub_boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_obj_boxes = torch.cat([t['obj_boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        exist_obj_boxes = (target_obj_boxes != 0).any(dim=1)

        losses = {}
        if src_sub_boxes.shape[0] == 0:
            losses['loss_sub_bbox'] = src_sub_boxes.sum()
            losses['loss_obj_bbox'] = src_obj_boxes.sum()
            losses['loss_sub_giou'] = src_sub_boxes.sum()
            losses['loss_obj_giou'] = src_obj_boxes.sum()
        else:
            loss_sub_bbox = F.l1_loss(src_sub_boxes, target_sub_boxes, reduction='none')
            loss_obj_bbox = F.l1_loss(src_obj_boxes, target_obj_boxes, reduction='none')
            losses['loss_sub_bbox'] = loss_sub_bbox.sum() / num_interactions
            losses['loss_obj_bbox'] = (loss_obj_bbox * exist_obj_boxes.unsqueeze(1)).sum() / (
                    exist_obj_boxes.sum() + 1e-4)
            loss_sub_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_sub_boxes),
                                                               box_cxcywh_to_xyxy(target_sub_boxes)))
            loss_obj_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_obj_boxes),
                                                               box_cxcywh_to_xyxy(target_obj_boxes)))
            losses['loss_sub_giou'] = loss_sub_giou.sum() / num_interactions
            losses['loss_obj_giou'] = (loss_obj_giou * exist_obj_boxes).sum() / (exist_obj_boxes.sum() + 1e-4)
        return losses

    def mimic_loss(self, outputs, targets, indices, num_interactions):
        src_feats = outputs['inter_memory']
        src_feats = torch.mean(src_feats, dim=1)

        target_clip_inputs = torch.cat([t['clip_inputs'].unsqueeze(0) for t in targets])
        with torch.no_grad():
            target_clip_feats = self.clip_model.encode_image(target_clip_inputs)
        loss_feat_mimic = F.l1_loss(src_feats, target_clip_feats)
        losses = {'loss_feat_mimic': loss_feat_mimic}
        return losses

    def _neg_loss(self, pred, gt, weights=None, alpha=0.25):
        ''' Modified focal loss. Exactly the same as CornerNet.
          Runs faster and costs a little bit more memory
        '''
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        loss = 0

        pos_loss = alpha * torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
        if weights is not None:
            pos_loss = pos_loss * weights[:-1]

        neg_loss = (1 - alpha) * torch.log(1 - pred) * torch.pow(pred, 2) * neg_inds

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
        return loss

    def _neg_loss_masked(self, pred, gt, mask, alpha=0.25):
        """CornerNet-style focal loss with an element-wise mask.
        pred: sigmoid probs in (0,1), gt: hard labels in {0,1}, mask: {0,1}
        """
        pos_inds = (gt.eq(1).float()) * mask
        neg_inds = (gt.lt(1).float()) * mask

        pos_loss = alpha * torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
        neg_loss = (1 - alpha) * torch.log(1 - pred) * torch.pow(pred, 2) * neg_inds

        num_pos = pos_inds.sum()
        if num_pos < 1:
            return -neg_loss.sum()
        return -(pos_loss.sum() + neg_loss.sum()) / num_pos

    def _sigmoid_focal_soft_with_logits_masked(self, logits, targets, mask, alpha=0.25, gamma=2.0):
        """Sigmoid focal loss with *soft targets* and an element-wise mask.
        logits: student logits, targets: soft labels in [0,1], mask: {0,1}
        """
        # BCE with logits supports soft targets
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce * torch.pow(1 - p_t, gamma)

        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = loss * alpha_t

        loss = loss * mask
        denom = mask.sum().clamp_min(1.0)
        return loss.sum() / denom

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def get_loss(self, loss, outputs, targets, indices, num, **kwargs):
        if 'pred_hoi_logits' in outputs.keys():
            loss_map = {
                'hoi_labels': self.loss_hoi_labels,
                'obj_labels': self.loss_obj_labels,
                'sub_obj_boxes': self.loss_sub_obj_boxes,
                'feats_mimic': self.mimic_loss
            }
        else:
            loss_map = {
                'obj_labels': self.loss_obj_labels,
                'obj_cardinality': self.loss_obj_cardinality,
                'verb_labels': self.loss_verb_labels,
                'sub_obj_boxes': self.loss_sub_obj_boxes
            }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num, **kwargs)

    def forward(self, outputs, targets):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        num_interactions = sum(len(t['hoi_labels']) for t in targets)
        num_interactions = torch.as_tensor([num_interactions], dtype=torch.float,
                                           device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_interactions)
        num_interactions = torch.clamp(num_interactions / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_interactions))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    kwargs = {}
                    if loss == 'obj_labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_interactions, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


class PostProcessHOITriplet(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.subject_category_id = args.subject_category_id

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        out_hoi_logits = outputs['pred_hoi_logits']
        out_obj_logits = outputs['pred_obj_logits']
        out_sub_boxes = outputs['pred_sub_boxes']
        out_obj_boxes = outputs['pred_obj_boxes']

        assert len(out_hoi_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        hoi_scores = out_hoi_logits.sigmoid()
        obj_scores = out_obj_logits.sigmoid()
        obj_labels = F.softmax(out_obj_logits, -1)[..., :-1].max(-1)[1]

        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(hoi_scores.device)
        sub_boxes = box_cxcywh_to_xyxy(out_sub_boxes)
        sub_boxes = sub_boxes * scale_fct[:, None, :]
        obj_boxes = box_cxcywh_to_xyxy(out_obj_boxes)
        obj_boxes = obj_boxes * scale_fct[:, None, :]

        results = []
        for index in range(len(hoi_scores)):
            hs, os, ol, sb, ob = hoi_scores[index], obj_scores[index], obj_labels[index], sub_boxes[index], obj_boxes[
                index]
            sl = torch.full_like(ol, self.subject_category_id)
            l = torch.cat((sl, ol))
            b = torch.cat((sb, ob))
            results.append({'labels': l.to('cpu'), 'boxes': b.to('cpu')})

            ids = torch.arange(b.shape[0])

            results[-1].update({'hoi_scores': hs.to('cpu'), 'obj_scores': os.to('cpu'),
                                'sub_ids': ids[:ids.shape[0] // 2], 'obj_ids': ids[ids.shape[0] // 2:]})

        return results


def build_cadm_l(args):
    device = torch.device(args.device)

    num_classes = args.num_classes
    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_deformable_transformer(args)

    try:
        match_unstable_error = args.match_unstable_error
        dn_labelbook_size = args.dn_labelbook_size
    except:
        match_unstable_error = True
        dn_labelbook_size = num_classes

    try:
        dec_pred_class_embed_share = args.dec_pred_class_embed_share
    except:
        dec_pred_class_embed_share = True
    try:
        dec_pred_bbox_embed_share = args.dec_pred_bbox_embed_share
    except:
        dec_pred_bbox_embed_share = True

    model = DiffHOI_L(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=True,
        iter_update=True,
        query_dim=4,
        random_refpoints_xy=args.random_refpoints_xy,
        fix_refpoints_hw=args.fix_refpoints_hw,
        num_feature_levels=args.num_feature_levels,
        nheads=args.nheads,
        dec_pred_class_embed_share=dec_pred_class_embed_share,
        dec_pred_bbox_embed_share=dec_pred_bbox_embed_share,
        # two stage
        two_stage_type=args.two_stage_type,
        # box_share
        two_stage_bbox_embed_share=args.two_stage_bbox_embed_share,
        two_stage_class_embed_share=args.two_stage_class_embed_share,
        decoder_sa_type=args.decoder_sa_type,
        num_patterns=args.num_patterns,
        dn_number=args.dn_number if args.use_dn else 0,
        dn_box_noise_scale=args.dn_box_noise_scale,
        dn_label_noise_ratio=args.dn_label_noise_ratio,
        dn_labelbook_size=dn_labelbook_size,
        args=args
    )
    matcher = build_matcher(args)
    weight_dict = {}
    if args.with_clip_label:
        weight_dict['loss_hoi_labels'] = args.hoi_loss_coef
        weight_dict['loss_obj_ce'] = args.obj_loss_coef
    else:
        weight_dict['loss_hoi_labels'] = args.hoi_loss_coef
        weight_dict['loss_obj_ce'] = args.obj_loss_coef

    weight_dict['loss_sub_bbox'] = args.bbox_loss_coef
    weight_dict['loss_obj_bbox'] = args.bbox_loss_coef
    weight_dict['loss_sub_giou'] = args.giou_loss_coef
    weight_dict['loss_obj_giou'] = args.giou_loss_coef
    # if args.with_mimic:
    #     weight_dict['loss_feat_mimic'] = args.mimic_loss_coef
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)
    losses = ['hoi_labels', 'obj_labels', 'sub_obj_boxes']
    if args.with_mimic:
        losses.append('feats_mimic')

    criterion = SetCriterionHOI(args.num_obj_classes, args.num_queries, args.num_verb_classes, matcher=matcher,
                                weight_dict=weight_dict, eos_coef=args.eos_coef, losses=losses,
                                args=args)
    criterion.to(device)
    postprocessors = {'hoi': PostProcessHOITriplet(args)}

    return model, criterion, postprocessors
