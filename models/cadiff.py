
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from PIL import Image


try:
    import clip  # openai/CLIP
except Exception as e:
    raise RuntimeError("CLIP package is required. Install openai-clip or ensure 'clip' is importable.") from e

try:
    from omegaconf import OmegaConf
    from ldm.util import instantiate_from_config
except Exception as e:
    raise RuntimeError("Stable Diffusion (LDM) dependencies missing: omegaconf + ldm.util.instantiate_from_config") from e

try:
    from SD_Extractor import UNetWrapper
except Exception as e:
    raise RuntimeError("Cannot import UNetWrapper from SD_Extractor. Ensure it exists in your project.") from e


# ----------------------------
# Utilities
# ----------------------------

def set_seed(seed: int) -> None:
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def xyxy_union(box1: List[float], box2: List[float]) -> List[float]:
    x0 = min(box1[0], box2[0])
    y0 = min(box1[1], box2[1])
    x1 = max(box1[2], box2[2])
    y1 = max(box1[3], box2[3])
    return [x0, y0, x1, y1]


def clamp_box_xyxy(box: List[float], w: int, h: int) -> List[float]:
    x0, y0, x1, y1 = box
    x0 = max(0.0, min(float(w - 1), x0))
    y0 = max(0.0, min(float(h - 1), y0))
    x1 = max(0.0, min(float(w), x1))
    y1 = max(0.0, min(float(h), y1))
    # keep valid ordering
    if x1 <= x0:
        x1 = min(float(w), x0 + 1.0)
    if y1 <= y0:
        y1 = min(float(h), y0 + 1.0)
    return [x0, y0, x1, y1]


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def pil_crop(img: Image.Image, box_xyxy: List[float]) -> Image.Image:
    x0, y0, x1, y1 = box_xyxy
    return img.crop((int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))))


def sd_image_transform(img: Image.Image, size: int = 512) -> torch.Tensor:
    """
    SD first-stage encoder typically expects float tensor in [-1, 1].
    """
    import numpy as np
    img = img.convert("RGB")
    img = img.resize((size, size), resample=Image.BICUBIC)
    arr = np.asarray(img).astype("float32") / 255.0  # HWC in [0,1]
    x = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # CHW
    x = x * 2.0 - 1.0
    return x


@dataclass
class Batch:
    clip_inputs: torch.Tensor  # [B,3,224,224]
    sd_inputs: torch.Tensor    # [B,3,512,512], in [-1,1]
    labels: torch.Tensor       # [B,C] float or [B] long


class UnionCropJsonlDataset(Dataset):
    """
    A minimal dataset for Stage I teacher pretraining.

    Each sample corresponds to ONE human-object union crop and its HOI label(s).
    JSONL supports either:
      - on-the-fly cropping from original image path
      - loading precomputed tensors (.pt) for clip_inputs / sd_inputs
    """

    def __init__(
        self,
        jsonl_path: str,
        num_hoi_classes: int,
        clip_preprocess,
        sd_size: int = 512,
        boxes_normalized: bool = False,
        image_root: str = "",
        use_precomputed_inputs: bool = False,
    ) -> None:
        super().__init__()
        self.items = load_jsonl(jsonl_path)
        self.num_hoi_classes = num_hoi_classes
        self.clip_preprocess = clip_preprocess
        self.sd_size = sd_size
        self.boxes_normalized = boxes_normalized
        self.image_root = image_root
        self.use_precomputed_inputs = use_precomputed_inputs

    def __len__(self) -> int:
        return len(self.items)

    def _make_label(self, it: Dict[str, Any]) -> torch.Tensor:
        if "label" in it:
            # single-label
            y = torch.zeros(self.num_hoi_classes, dtype=torch.float32)
            y[int(it["label"])] = 1.0
            return y
        elif "labels" in it:
            y = torch.zeros(self.num_hoi_classes, dtype=torch.float32)
            for k in it["labels"]:
                y[int(k)] = 1.0
            return y
        else:
            raise KeyError("JSONL item must contain 'label' or 'labels'.")

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        it = self.items[idx]
        label = self._make_label(it)

        if self.use_precomputed_inputs:
            clip_inputs = torch.load(it["clip_pt"], map_location="cpu")
            sd_inputs = torch.load(it["sd_pt"], map_location="cpu")
            return {"clip_inputs": clip_inputs, "sd_inputs": sd_inputs, "label": label}

        img_path = it.get("img", None)
        if img_path is None:
            raise KeyError("When not using precomputed inputs, JSONL item must contain 'img'.")
        if self.image_root and not os.path.isabs(img_path):
            img_path = os.path.join(self.image_root, img_path)

        box = it.get("union_box", None)
        if box is None:
            raise KeyError("JSONL item must contain 'union_box' for cropping.")
        box = [float(x) for x in box]

        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        if self.boxes_normalized:
            box = [box[0] * w, box[1] * h, box[2] * w, box[3] * h]

        box = clamp_box_xyxy(box, w=w, h=h)
        crop = pil_crop(img, box)

        clip_inputs = self.clip_preprocess(crop)  # tensor CHW normalized as CLIP expects
        sd_inputs = sd_image_transform(crop, size=self.sd_size)  # tensor CHW in [-1,1]

        return {"clip_inputs": clip_inputs, "sd_inputs": sd_inputs, "label": label}


def collate_fn(batch: List[Dict[str, Any]]) -> Batch:
    clip_inputs = torch.stack([b["clip_inputs"] for b in batch], dim=0)
    sd_inputs = torch.stack([b["sd_inputs"] for b in batch], dim=0)
    labels = torch.stack([b["label"] for b in batch], dim=0)
    return Batch(clip_inputs=clip_inputs, sd_inputs=sd_inputs, labels=labels)



class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        layers: List[nn.Module] = []
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU(inplace=True))
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MixConvBlock(nn.Module):
    """
    Paper: "two 3x3 conv layers with residual connections and batch norm".
    """
    def __init__(self, dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(dim)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(dim)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.act(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = self.act(x + residual)
        return x


class FusionEncoder(nn.Module):

    def __init__(self, out_dim: int = 512, fuse_dim: int = 256, upsample_mode: str = "bilinear"):
        super().__init__()
        self.out_dim = out_dim
        self.fuse_dim = fuse_dim
        self.upsample_mode = upsample_mode

        self.proj_convs: Optional[nn.ModuleList] = None
        self.mixconv: Optional[MixConvBlock] = None
        self.out_proj: Optional[nn.Linear] = None

    def _lazy_init(self, feats: List[torch.Tensor]) -> None:
        # Build one 1x1 conv per scale
        self.proj_convs = nn.ModuleList([
            nn.Conv2d(f.shape[1], self.fuse_dim, kernel_size=1, bias=False) for f in feats
        ])
        self.mixconv = MixConvBlock(dim=self.fuse_dim * len(feats))
        self.out_proj = nn.Linear(self.fuse_dim * len(feats), self.out_dim)

        # register modules on correct device
        self.proj_convs.to(feats[0].device)
        self.mixconv.to(feats[0].device)
        self.out_proj.to(feats[0].device)

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        # Keep only feature maps (B,C,H,W)
        feats = [f for f in feats if f.dim() == 4]
        if len(feats) == 0:
            raise ValueError("FusionEncoder expects a non-empty list of 4D feature maps.")

        if self.proj_convs is None:
            self._lazy_init(feats)

        assert self.proj_convs is not None and self.mixconv is not None and self.out_proj is not None

        # Choose the largest spatial size as target
        max_h = max([f.shape[2] for f in feats])
        max_w = max([f.shape[3] for f in feats])

        fused: List[torch.Tensor] = []
        for f, conv in zip(feats, self.proj_convs):
            x = conv(f)
            if x.shape[2] != max_h or x.shape[3] != max_w:
                x = F.interpolate(x, size=(max_h, max_w), mode=self.upsample_mode, align_corners=False)
            fused.append(x)

        x = torch.cat(fused, dim=1)  # [B, fuse_dim*n, H, W]
        x = self.mixconv(x)
        x = x.mean(dim=[2, 3])       # GAP -> [B, fuse_dim*n]
        x = self.out_proj(x)         # [B, out_dim]
        return x


class CaDiffStage1(nn.Module):

    def __init__(
        self,
        dataset_file: str,
        sd_config: str,
        sd_ckpt: Optional[str],
        clip_model_name: str,
        num_hoi_classes: int,
        sd_attn_selector: str = "",
        sd_use_attn: bool = True,
        sd_base_size: int = 64,
        sd_max_size: int = 512,
        sd_timestep: int = 0,
        embed_dim: int = 512,
        implicit_hidden_dim: int = 1024,
        implicit_out_dim: int = 768,
        hoi_adapter_hidden: int = 512,
        fusion_dim: int = 256,
        device: str = "cuda",
    ) -> None:
        super().__init__()
        self.dataset_file = dataset_file
        self.num_hoi_classes = num_hoi_classes
        self.sd_timestep = int(sd_timestep)

        # --- CLIP encoders (frozen) ---
        self.clip_model, _ = clip.load(clip_model_name, device=device)
        self.clip_model.eval()
        for p in self.clip_model.parameters():
            p.requires_grad = False

        # Determine CLIP image embedding dim
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224, device=device)
            clip_dim = self.clip_model.encode_image(dummy).shape[-1]

        # MLP adapter: CLIPimg(I_crop) -> implicit caption embedding (Eq. 2)
        self.implicit_adapter = MLP(clip_dim, implicit_hidden_dim, implicit_out_dim, num_layers=2, dropout=0.0)

        # --- Stable Diffusion UNet + VAE (frozen) ---
        ckpt = sd_ckpt or os.environ.get("SD_ckpt", None)
        if ckpt is None:
            raise ValueError("Stable Diffusion checkpoint not provided. Set --sd_ckpt or env SD_ckpt.")
        config = OmegaConf.load(sd_config)
        sd_model = instantiate_from_config(config.model)
        sd_state = torch.load(ckpt, map_location="cpu")
        sd_model.load_state_dict(sd_state["state_dict"], strict=False)

        # UNetWrapper wraps sd_model.model.diffusion_model
        self.unet = UNetWrapper(
            sd_model.model.diffusion_model,
            use_attn=sd_use_attn,
            base_size=sd_base_size,
            max_size=sd_max_size,
            attn_selector=sd_attn_selector,
        )
        self.encoder_vq = sd_model.first_stage_model

        self.unet.eval()
        self.encoder_vq.eval()
        for p in self.unet.parameters():
            p.requires_grad = False
        for p in self.encoder_vq.parameters():
            p.requires_grad = False

        # --- Fusion encoder Φ_F (trainable) ---
        self.fusion_encoder = FusionEncoder(out_dim=embed_dim, fuse_dim=fusion_dim)

        # --- HOI text bank W and HOI adapter -> W' (trainable adapter) ---
        hoi_text_label = self._load_hoi_texts(dataset_file)
        # Ensure ordering is consistent with class indices (keys are expected 0..C-1)
        prompts = [hoi_text_label[k] for k in sorted(hoi_text_label.keys())]
        if len(prompts) != num_hoi_classes:
            # If mismatch, still proceed but warn
            print(f"[WARN] #prompts={len(prompts)} != num_hoi_classes={num_hoi_classes}. "
                  f"Check your label bank alignment.")
        text_inputs = torch.cat([clip.tokenize(p) for p in prompts], dim=0).to(device)
        with torch.no_grad():
            text_emb = self.clip_model.encode_text(text_inputs).float()  # [C, D]
        self.register_buffer("hoi_text_emb", text_emb, persistent=False)

        text_dim = text_emb.shape[-1]
        self.hoi_adapter = MLP(text_dim, hoi_adapter_hidden, text_dim, num_layers=2, dropout=0.0)

        # CLIP-style temperature
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))

    @staticmethod
    def _load_hoi_texts(dataset_file: str) -> Dict[int, str]:
        if dataset_file == "hico":
            from datasets.hico_text_label import hico_text_label
            return hico_text_label
        if dataset_file == "vcoco":
            from datasets.vcoco_text_label import vcoco_hoi_text_label
            # vcoco_hoi_text_label might be list; normalize to dict
            if isinstance(vcoco_hoi_text_label, dict):
                return vcoco_hoi_text_label
            return {i: t for i, t in enumerate(vcoco_hoi_text_label)}
        raise ValueError(f"Unsupported dataset_file={dataset_file}. Use 'hico' or 'vcoco'.")

    @torch.no_grad()
    def _encode_sd_latents(self, sd_inputs: torch.Tensor) -> torch.Tensor:
        # sd_inputs: [B,3,H,W] in [-1,1]
        posterior = self.encoder_vq.encode(sd_inputs)
        if hasattr(self.encoder_vq, "get_first_stage_encoding"):
            latents = self.encoder_vq.get_first_stage_encoding(posterior)
        else:
            # fallback: some implementations store .mode() or .sample()
            latents = posterior.mode()
        return latents

    def forward(self, clip_inputs: torch.Tensor, sd_inputs: torch.Tensor) -> torch.Tensor:

        device = clip_inputs.device

        # CLIP image embedding (frozen)
        with torch.no_grad():
            clip_feat = self.clip_model.encode_image(clip_inputs).float()  # [B, D]
        # Implicit caption embedding for SD cross-attn
        implicit = self.implicit_adapter(clip_feat)  # [B, 768]
        implicit = implicit.unsqueeze(1)             # [B, 1, 768]

        # SD latents + UNet features (single forward pass)
        with torch.no_grad():
            latents = self._encode_sd_latents(sd_inputs)
            t = torch.full((latents.shape[0],), self.sd_timestep, device=device, dtype=torch.long)
            feats = self.unet.forward(latents, t, c_crossattn=[implicit])

        # Fuse multi-scale UNet features -> v'
        v = self.fusion_encoder(list(feats))  # [B, embed_dim]
        v = F.normalize(v, dim=-1)

        # HOI knowledge bank W' (trainable adapter)
        w = self.hoi_adapter(self.hoi_text_emb)  # [C, D]
        w = F.normalize(w, dim=-1)

        logit_scale = self.logit_scale.exp().clamp(max=100.0)
        logits = logit_scale * (v @ w.t())  # [B, C]
        return logits


# ----------------------------
# Train / Eval
# ----------------------------

def compute_loss(logits: torch.Tensor, labels: torch.Tensor, use_softmax: bool = True) -> torch.Tensor:

    if use_softmax:
        probs = logits.softmax(dim=-1)
        return F.binary_cross_entropy(probs, labels)
    else:
        return F.binary_cross_entropy_with_logits(logits, labels)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str, use_softmax: bool) -> float:
    model.eval()
    total_loss = 0.0
    n = 0
    for batch in loader:
        clip_in = batch.clip_inputs.to(device, non_blocking=True)
        sd_in = batch.sd_inputs.to(device, non_blocking=True)
        y = batch.labels.to(device, non_blocking=True)
        logits = model(clip_in, sd_in)
        loss = compute_loss(logits, y, use_softmax=use_softmax)
        total_loss += float(loss.item()) * clip_in.shape[0]
        n += clip_in.shape[0]
    return total_loss / max(1, n)


def save_checkpoint(path: str, model: CaDiffStage1, optimizer: torch.optim.Optimizer, epoch: int, args: argparse.Namespace) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Only save trainable modules (adapters + fusion encoder + logit_scale)
    ckpt = {
        "epoch": epoch,
        "args": vars(args),
        "state_dict": {
            "implicit_adapter": model.implicit_adapter.state_dict(),
            "fusion_encoder": model.fusion_encoder.state_dict(),
            "hoi_adapter": model.hoi_adapter.state_dict(),
            "logit_scale": model.logit_scale.detach().cpu(),
        },
        "optimizer": optimizer.state_dict(),
    }
    torch.save(ckpt, path)


def main() -> None:
    parser = argparse.ArgumentParser("Stage I CaDiff teacher pretraining")

    # Data
    parser.add_argument("--dataset_file", default="hico", choices=["hico", "vcoco"])
    parser.add_argument("--train_jsonl", required=True, help="JSONL of union-crop samples for training")
    parser.add_argument("--val_jsonl", default="", help="Optional JSONL for validation")
    parser.add_argument("--image_root", default="", help="Prefix for relative image paths in JSONL")
    parser.add_argument("--boxes_normalized", action="store_true", help="If union_box coordinates are normalized in [0,1]")
    parser.add_argument("--use_precomputed_inputs", action="store_true", help="Load clip_inputs/sd_inputs from .pt paths")

    # Model / SD / CLIP
    parser.add_argument("--clip_model", default="ViT-B/16")
    parser.add_argument("--sd_config", required=True, help="LDM config yaml for Stable Diffusion")
    parser.add_argument("--sd_ckpt", default="", help="Stable Diffusion ckpt path (or set env SD_ckpt)")
    parser.add_argument("--sd_timestep", type=int, default=0)
    parser.add_argument("--sd_use_attn", action="store_true", help="Enable attention feature extraction in UNetWrapper")
    parser.add_argument("--sd_attn_selector", default="", help="UNetWrapper attn selector (project-specific)")

    # Dimensions
    parser.add_argument("--num_hoi_classes", type=int, default=600, help="600 for HICO-DET, 26 (or #verbs) for V-COCO etc")
    parser.add_argument("--embed_dim", type=int, default=512, help="Output dim for v' and w'")
    parser.add_argument("--implicit_hidden_dim", type=int, default=1024)
    parser.add_argument("--implicit_out_dim", type=int, default=768)
    parser.add_argument("--hoi_adapter_hidden", type=int, default=512)
    parser.add_argument("--fusion_dim", type=int, default=256)

    # Training
    parser.add_argument("--output_dir", default="outputs/cadiff_stage1")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--use_softmax_bce", action="store_true", help="Use paper-style Softmax->BCE (default).")
    parser.add_argument("--use_bce_logits", action="store_true", help="Use BCEWithLogits (ignores softmax).")
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp mixed precision")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_every", type=int, default=1)

    args = parser.parse_args()

    if args.use_softmax_bce and args.use_bce_logits:
        raise ValueError("Choose only one of --use_softmax_bce or --use_bce_logits.")
    use_softmax = True
    if args.use_bce_logits:
        use_softmax = False

    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # CLIP preprocess (same as clip.load)
    _, clip_preprocess = clip.load(args.clip_model, device=device)

    train_set = UnionCropJsonlDataset(
        jsonl_path=args.train_jsonl,
        num_hoi_classes=args.num_hoi_classes,
        clip_preprocess=clip_preprocess,
        sd_size=512,
        boxes_normalized=args.boxes_normalized,
        image_root=args.image_root,
        use_precomputed_inputs=args.use_precomputed_inputs,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True,
    )

    val_loader = None
    if args.val_jsonl:
        val_set = UnionCropJsonlDataset(
            jsonl_path=args.val_jsonl,
            num_hoi_classes=args.num_hoi_classes,
            clip_preprocess=clip_preprocess,
            sd_size=512,
            boxes_normalized=args.boxes_normalized,
            image_root=args.image_root,
            use_precomputed_inputs=args.use_precomputed_inputs,
        )
        val_loader = DataLoader(
            val_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            drop_last=False,
        )

    model = CaDiffStage1(
        dataset_file=args.dataset_file,
        sd_config=args.sd_config,
        sd_ckpt=args.sd_ckpt if args.sd_ckpt else None,
        clip_model_name=args.clip_model,
        num_hoi_classes=args.num_hoi_classes,
        sd_use_attn=args.sd_use_attn,
        sd_attn_selector=args.sd_attn_selector,
        sd_timestep=args.sd_timestep,
        embed_dim=args.embed_dim,
        implicit_hidden_dim=args.implicit_hidden_dim,
        implicit_out_dim=args.implicit_out_dim,
        hoi_adapter_hidden=args.hoi_adapter_hidden,
        fusion_dim=args.fusion_dim,
        device=device,
    ).to(device)

    # Only train adapters + fusion + logit_scale
    train_params = []
    for name, p in model.named_parameters():
        if p.requires_grad:
            train_params.append(p)

    optimizer = torch.optim.AdamW(train_params, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n = 0

        for batch in train_loader:
            clip_in = batch.clip_inputs.to(device, non_blocking=True)
            sd_in = batch.sd_inputs.to(device, non_blocking=True)
            y = batch.labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=args.amp):
                logits = model(clip_in, sd_in)
                loss = compute_loss(logits, y, use_softmax=use_softmax)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running += float(loss.item()) * clip_in.shape[0]
            n += clip_in.shape[0]

        train_loss = running / max(1, n)

        msg = f"[Epoch {epoch:03d}/{args.epochs}] train_loss={train_loss:.6f}"
        if val_loader is not None:
            val_loss = evaluate(model, val_loader, device=device, use_softmax=use_softmax)
            msg += f"  val_loss={val_loss:.6f}"
        print(msg, flush=True)

        if epoch % args.save_every == 0:
            ckpt_path = os.path.join(args.output_dir, f"cadiff_stage1_epoch{epoch:03d}.pth")
            save_checkpoint(ckpt_path, model, optimizer, epoch, args)

    # final
    final_path = os.path.join(args.output_dir, "cadiff_stage1_final.pth")
    save_checkpoint(final_path, model, optimizer, args.epochs, args)
    print(f"Saved final checkpoint to: {final_path}")


if __name__ == "__main__":
    main()
