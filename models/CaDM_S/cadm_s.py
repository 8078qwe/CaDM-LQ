import torch
from torch import nn
import torch.nn.functional as F
import os
from typing import Tuple

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size,
                       is_dist_avail_and_initialized)
import numpy as np
import clip
from datasets.hico_text_label import hico_text_label, hico_obj_text_label, hico_unseen_index
from datasets.vcoco_text_label import vcoco_hoi_text_label, vcoco_obj_text_label

from .backbone import build_backbone
from .matcher import build_matcher
from .cadm_s_transformer import build_cadm_s_transformer

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


class CaDM_S(nn.Module):
    def __init__(self, backbone, transformer, num_queries, aux_loss=False, args=None, unet_config=dict()):
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
        sd_model.model = None
        sd_model.first_stage_model = None
        del sd_model.cond_stage_model
        del self.encoder_vq.decoder
        self.sd_model = sd_model
        self.text_adapter = nn.Linear(args.clip_embed_dim, 768)
        self.clip_adapter = nn.Linear(args.clip_embed_dim, args.clip_embed_dim)

        self.args = args
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.query_embed_h = nn.Embedding(num_queries, hidden_dim)
        self.query_embed_o = nn.Embedding(num_queries, hidden_dim)
        self.pos_guided_embedd = nn.Embedding(num_queries, hidden_dim)
        self.hum_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.obj_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.dec_layers = self.args.dec_layers

        # ---------------- Stage II: Human-object Pair Learner ----------------
        self.use_pair_learner = bool(getattr(args, "use_pair_learner", True))
        self.num_pair_queries = int(getattr(args, "num_pair_queries", num_queries))
        self.pair_mapper = getattr(args, "pair_mapper", "identity")
        self.return_pair_info = bool(getattr(args, "return_pair_info", False))
        self.pair_learner = HumanObjectPairLearner(
            hidden_dim=hidden_dim,
            num_obj_query=num_queries,
            num_pair_query=self.num_pair_queries,
            mapper=self.pair_mapper,
        )

        # ---------------- Stage II: Locality-aware Interaction Decoder ----------------
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
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.obj_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.image_adapter = nn.Conv2d(1280, hidden_dim, kernel_size=1)

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
            self.register_buffer(
                "hoi_text_emb",
                train_clip_label / (train_clip_label.norm(dim=-1, keepdim=True) + 1e-6)
            )
            self.kd_eps = getattr(args, "kd_eps", 1e-6)
            self.use_cadiff_kd = getattr(args, "use_cadiff_kd", False)
            if self.args.dataset_file == 'hico' and self.args.zero_shot_type != 'default':
                self.eval_visual_projection = nn.Linear(args.clip_embed_dim, 600)
                self.eval_visual_projection.weight.data = clip_label / clip_label.norm(dim=-1, keepdim=True)
        else:
            self.hoi_class_embedding = nn.Linear(args.clip_embed_dim, len(hoi_text))

        if args.with_obj_clip_label:
            self.obj_class_fc = nn.Sequential(
                nn.Linear(hidden_dim, args.clip_embed_dim),
                nn.LayerNorm(args.clip_embed_dim),
            )
            self.obj_visual_projection = nn.Linear(args.clip_embed_dim, num_obj_classes + 1)
            self.obj_visual_projection.weight.data = obj_clip_label / obj_clip_label.norm(dim=-1, keepdim=True)
        else:
            self.obj_class_embed = nn.Linear(hidden_dim, num_obj_classes + 1)

        self.hidden_dim = hidden_dim
        self.reset_parameters()

        # device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, _ = clip.load(args.clip_model, device=device)
        self.freeze()

    def freeze(self):
        self.unet = self.unet.eval()
        for param in self.unet.parameters():
            param.requires_grad = False

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

    def reset_parameters(self):
        nn.init.uniform_(self.pos_guided_embedd.weight)

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


def forward(self, samples: NestedTensor, targets, is_training=True):
    if not isinstance(samples, NestedTensor):
        samples = nested_tensor_from_tensor_list(samples)

    features, pos = self.backbone(samples)
    src, mask = features[-1].decompose()

    # Stable-Diffusion feature (frozen) used as additional conditioning
    sd_feat = self.extract_feat(samples.decompose()[0], targets)
    sd_decoder = self.image_adapter(sd_feat[-2])
    sd_decoder = torch.nn.functional.interpolate(sd_decoder, size=[src.shape[-2], src.shape[-1]])

    assert mask is not None
    h_hs, o_hs, inter_hs = self.transformer(
        self.input_proj(src), mask,
        self.query_embed_h.weight,
        self.query_embed_o.weight,
        self.pos_guided_embedd.weight,
        pos[-1], sd_decoder
    )[:3]  # each: [dec, B, N, C]

    # Boxes per decoder layer (subject/person and object)
    outputs_sub_coord = self.hum_bbox_embed(h_hs).sigmoid()  # [dec, B, N, 4]
    outputs_obj_coord = self.obj_bbox_embed(o_hs).sigmoid()  # [dec, B, N, 4]

    # CLIP image embedding for (a) HOI classification bias and (b) KD teacher
    target_clip_inputs = torch.cat([t['clip_inputs'].unsqueeze(0) for t in targets])
    with torch.no_grad():
        target_clip_feats = self.clip_model.encode_image(target_clip_inputs)
    target_clip_feats = self.clip_adapter(target_clip_feats.float())  # [B, clip_dim]

    # Object logits (per layer)
    if self.args.with_obj_clip_label:
        obj_logit_scale = self.obj_logit_scale.exp()
        o_hs_cls = self.obj_class_fc(o_hs)
        o_hs_cls = o_hs_cls / (o_hs_cls.norm(dim=-1, keepdim=True) + 1e-6)
        outputs_obj_class = obj_logit_scale * self.obj_visual_projection(o_hs_cls)  # [dec,B,N,num_obj+1]
    else:
        outputs_obj_class = self.obj_class_embed(o_hs)  # [dec,B,N,num_obj+1]

    dec, B, N, C = h_hs.shape

    # ---------------- Pair Learner: select Top-K (human_idx, obj_idx) pairs ----------------
    if self.use_pair_learner:
        # Build an interleaved [sub_0,obj_0, sub_1,obj_1, ...] tensor for compatibility with the PairLearner
        hs_last = torch.stack([h_hs[-1], o_hs[-1]], dim=2).reshape(B, N * 2, C)  # [B,2N,C]
        pair_sub_pos, pair_obj_pos, pair_importance = self.pair_learner.select_topk_indices(hs_last)

        # Pair queries per decoder layer
        pair_hs_layers = []
        for lid in range(dec):
            hs_layer = torch.stack([h_hs[lid], o_hs[lid]], dim=2).reshape(B, N * 2, C)  # [B,2N,C]
            pair_hs_layers.append(self.pair_learner.build_pair_queries(hs_layer, pair_sub_pos, pair_obj_pos))
        hs_interaction = torch.stack(pair_hs_layers)  # [dec,B,K,C]
        K = hs_interaction.shape[2]

        # Gather boxes/logits to K pairs per layer
        def _gather_layer(x, idx, last_dim):
            # x: [dec,B,N,last_dim] or [dec,B,N,*]; idx: [B,K]
            idx_ex = idx.unsqueeze(0).unsqueeze(-1).expand(dec, B, K, last_dim)
            return torch.gather(x, 2, idx_ex)

        outputs_sub_coord_pair = _gather_layer(outputs_sub_coord, pair_sub_pos, 4)
        outputs_obj_coord_pair = _gather_layer(outputs_obj_coord, pair_obj_pos, 4)
        outputs_obj_class_pair = _gather_layer(outputs_obj_class, pair_obj_pos, outputs_obj_class.shape[-1])
    else:
        hs_interaction = inter_hs
        K = inter_hs.shape[2]
        outputs_sub_coord_pair = outputs_sub_coord
        outputs_obj_coord_pair = outputs_obj_coord
        outputs_obj_class_pair = outputs_obj_class
        pair_sub_pos = pair_obj_pos = pair_importance = None

    # ---------------- Locality-aware Interaction Decoder (optional) ----------------
    if self.use_locality_decoder:
        # Use fused image memory: DETR image proj + SD decoder feature
        mem_map = self.input_proj(src) + sd_decoder  # [B,C,H,W]
        mem = mem_map.flatten(2).transpose(1, 2)  # [B,HW,C]
        mem_pos = pos[-1].flatten(2).transpose(1, 2)  # [B,HW,C]

        # Pairwise spatial priors from gathered boxes
        sub_flat = outputs_sub_coord_pair.reshape(dec * B, K, 4)
        obj_flat = outputs_obj_coord_pair.reshape(dec * B, K, 4)
        sp_flat = self.pair_spatial_encoder(sub_flat, obj_flat)  # [dec*B,K,C]
        sp = sp_flat.reshape(dec, B, K, C)

        refined_layers = []
        for lid in range(dec):
            refined_layers.append(self.locality_decoder(hs_interaction[lid], mem, mem_pos, sp[lid]))
        hs_interaction = torch.stack(refined_layers)  # [dec,B,K,C]

    # ---------------- HOI logits from refined interaction queries ----------------
    if self.args.with_clip_label:
        logit_scale = self.logit_scale.exp()
        inter_hs_clip = self.hoi_class_fc(hs_interaction)  # [dec,B,K,clip_dim]
        outputs_inter_hs = inter_hs_clip.clone()  # for mimic loss (optional)
        inter_hs_clip = inter_hs_clip + target_clip_feats[None, :, None, :].repeat(dec, 1, K, 1)
        inter_hs_clip = inter_hs_clip / (inter_hs_clip.norm(dim=-1, keepdim=True) + 1e-6)

        if self.args.dataset_file == 'hico' and self.args.zero_shot_type != 'default' and (
                self.args.eval or not is_training):
            outputs_hoi_class = logit_scale * self.eval_visual_projection(inter_hs_clip)
        else:
            outputs_hoi_class = logit_scale * self.visual_projection(inter_hs_clip)
    else:
        inter_hs_clip = self.hoi_class_fc(hs_interaction)
        outputs_inter_hs = inter_hs_clip.clone()
        outputs_hoi_class = self.hoi_class_embedding(inter_hs_clip)

    # ---- Teacher soft labels (mechanism verification: CLIP image embedding teacher; can be replaced by CaDiff) ----
    teacher_hoi_probs = None
    if getattr(self, "use_cadiff_kd", False) and hasattr(self, "hoi_text_emb"):
        with torch.no_grad():
            img_emb = target_clip_feats / (target_clip_feats.norm(dim=-1, keepdim=True) + 1e-6)  # [B,clip_dim]
            txt_emb = self.hoi_text_emb  # [num_hoi, clip_dim]
            t_logits = self.logit_scale.exp() * (img_emb @ txt_emb.t())  # [B,num_hoi]
            t_probs = torch.sigmoid(t_logits)  # [B,num_hoi]
            t_probs = torch.where(t_probs > self.kd_eps, t_probs, torch.zeros_like(t_probs))
        teacher_hoi_probs = t_probs[:, None, :].expand(B, K, -1).contiguous()  # [B,K,num_hoi]

    out = {
        'pred_hoi_logits': outputs_hoi_class[-1],
        'pred_obj_logits': outputs_obj_class_pair[-1],
        'pred_sub_boxes': outputs_sub_coord_pair[-1],
        'pred_obj_boxes': outputs_obj_coord_pair[-1],
    }

    if teacher_hoi_probs is not None:
        out['teacher_hoi_probs'] = teacher_hoi_probs

    if self.return_pair_info and pair_sub_pos is not None:
        out['pair_info'] = {
            'sub_pos': pair_sub_pos,
            'obj_pos': pair_obj_pos,
            'importance': pair_importance,
        }

    if self.training and self.args.with_mimic:
        out['inter_memory'] = outputs_inter_hs[-1]

    if self.aux_loss:
        aux_mimic = outputs_inter_hs if (self.args.with_mimic and self.training) else None
        out['aux_outputs'] = self._set_aux_loss_triplet(
            outputs_hoi_class, outputs_obj_class_pair, outputs_sub_coord_pair, outputs_obj_coord_pair, aux_mimic
        )
        # propagate teacher probs to aux outputs (optional)
        if teacher_hoi_probs is not None:
            for _aux in out['aux_outputs']:
                _aux['teacher_hoi_probs'] = teacher_hoi_probs

    return out

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
        self.lambda_soft = getattr(args, 'lambda_soft', 0.0)
        self.kd_eps = getattr(args, 'kd_eps', 1e-6)
        self.kd_gamma = getattr(args, 'kd_gamma', 2.0)

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
    src_logits = outputs['pred_hoi_logits']  # [B, Q, num_hoi]

    idx = self._get_src_permutation_idx(indices)
    target_classes_o = torch.cat([t['hoi_labels'][J] for t, (_, J) in zip(targets, indices)])
    target_classes = torch.zeros_like(src_logits)
    target_classes[idx] = target_classes_o  # hard GT in {0,1}

    # student prob (clamped)
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

    # top-k error metric (keep original)
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
    """Sigmoid focal loss with soft targets and an element-wise mask.
    logits: student logits, targets: soft labels in [0,1], mask: {0,1}
    """
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


def build_cadm_s(args):
    device = torch.device(args.device)

    backbone = build_backbone(args)

    gen = build_cadm_s_transformer(args)

    model = CaDM_S(
        backbone,
        gen,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
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
    if args.with_mimic:
        weight_dict['loss_feat_mimic'] = args.mimic_loss_coef
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
