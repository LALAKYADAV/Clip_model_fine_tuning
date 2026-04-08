"""
model.py
CLIP ViT-L/14 with LoRA fine-tuning for face recognition.

Design:
  - All base CLIP weights are frozen
  - LoRA A/B matrices injected into out_proj of every attention block
    (compatible with CLIP's fused QKV design)
  - rsLoRA scaling: alpha / sqrt(rank)  — prevents gradient collapse at
    higher ranks, as recommended by FRoundation paper
  - Classification head for ArcFace / softmax training
  - merge_lora() merges weights back → zero inference latency, clean ONNX
"""

import math
import torch
import torch.nn as nn
import clip
# Add this at the top of model.py
import torch.nn.functional as F

# ══════════════════════════════════════════════════════════════════════════════
# LoRA layer
# ══════════════════════════════════════════════════════════════════════════════

# class LoRALinear(nn.Module):
#     """
#     Wraps an nn.Linear with Low-Rank Adaptation:
#         output = W·x  +  (B·A·x) × (alpha / sqrt(rank))
#     Only A and B are trained; W stays frozen.
#     """

#     def __init__(self, linear: nn.Linear, rank: int = 16,
#                  alpha: float = 32.0, use_rslora: bool = True):
#         super().__init__()
#         self.linear    = linear
#         self.rank      = rank
#         self.alpha     = alpha
#         self.use_rslora = use_rslora

#         in_f  = linear.in_features
#         out_f = linear.out_features

#         self.lora_A = nn.Parameter(torch.empty(rank, in_f))
#         self.lora_B = nn.Parameter(torch.zeros(out_f, rank))   # zero-init → identity at start

#         nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

#         # Freeze base weight
#         self.linear.weight.requires_grad_(False)
#         if self.linear.bias is not None:
#             self.linear.bias.requires_grad_(False)

#     @property
#     def scaling(self):
#         return (self.alpha / math.sqrt(self.rank) if self.use_rslora
#                 else self.alpha / self.rank)

#     def forward(self, x):
#         return self.linear(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scaling

#     def merge(self):
#         """Fold LoRA into base weight. Call before ONNX export."""
#         with torch.no_grad():
#             self.linear.weight.data += (self.lora_B @ self.lora_A) * self.scaling
#         self.linear.weight.requires_grad_(False)


class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, rank: int, alpha: float, use_rslora: bool = False):
        super().__init__()
        self.linear  = linear
        self.rank    = rank
        self.scaling = (alpha / math.sqrt(rank)) if use_rslora else (alpha / rank)

        in_f, out_f = linear.in_features, linear.out_features
        self.lora_A = nn.Parameter(torch.randn(rank, in_f)  * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank))

        # Freeze the original weight
        linear.weight.requires_grad_(False)
        if linear.bias is not None:
            linear.bias.requires_grad_(False)

    @property
    def weight(self):
        """Expose .weight so nn.MultiheadAttention's internals don't break."""
        return self.linear.weight + (self.lora_B @ self.lora_A) * self.scaling

    @property
    def bias(self):
        return self.linear.bias

    @property
    def in_features(self):
        return self.linear.in_features

    @property
    def out_features(self):
        return self.linear.out_features

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)
    

    # ← ADD merge() RIGHT HERE, same indentation as forward()
    def merge(self):
        with torch.no_grad():
            self.linear.weight.data += (self.lora_B @ self.lora_A) * self.scaling



# ══════════════════════════════════════════════════════════════════════════════
# CLIP + LoRA face model
# ══════════════════════════════════════════════════════════════════════════════

class CLIPFaceModel(nn.Module):
    """
    CLIP ViT-L/14 image encoder + LoRA + optional classification head.

    Args:
        num_classes : training identities for softmax/ArcFace head.
                      0 = embedding-only mode (inference / ONNX export).
        rank        : LoRA rank  (paper: 16 best accuracy/cost tradeoff)
        alpha       : LoRA alpha scaling
        use_rslora  : rank-stabilised LoRA (recommended True)
        embed_dim   : output dimension of ViT-L/14 = 768
    """

    CLIP_MODEL = "ViT-L/14"
    EMBED_DIM  = 768

    def __init__(self, num_classes: int = 0, rank: int = 16,
                 alpha: float = 32.0, use_rslora: bool = True):
        super().__init__()

        # Load CLIP image encoder in float32
        clip_model, _ = clip.load(self.CLIP_MODEL, device="cpu", jit=False)
        self.encoder  = clip_model.visual.float()

        # Freeze everything
        for p in self.encoder.parameters():
            p.requires_grad_(False)

        # Inject LoRA into out_proj of every transformer block
        self._inject_lora(rank, alpha, use_rslora)

        # Classification head
        if num_classes > 0:
            self.head = nn.Linear(self.EMBED_DIM, num_classes, bias=False)
        else:
            self.head = nn.Identity()

        self._print_param_summary()

    # ── LoRA injection ────────────────────────────────────────────────────────
    def _inject_lora(self, rank, alpha, use_rslora):
        for block in self.encoder.transformer.resblocks:
            block.attn.out_proj = LoRALinear(
                block.attn.out_proj, rank, alpha, use_rslora
            )

    # ── Forward ───────────────────────────────────────────────────────────────
    def forward(self, x):
        """
        Returns:
            logits    : (B, num_classes) — raw class scores (or identity if head=0)
            embeddings: (B, 768)         — L2-normalised face embeddings
        """
        features   = self.encoder(x)
        embeddings = features / features.norm(dim=-1, keepdim=True)
        logits     = self.head(embeddings)
        return logits, embeddings

    def get_embeddings(self, x):
        """Embedding-only pass — used for evaluation and ONNX export."""
        _, emb = self.forward(x)
        return emb

    # ── LoRA merge ────────────────────────────────────────────────────────────
    def merge_lora(self):
        """
        Merge all LoRA A/B into base weights.
        Call this BEFORE exporting to ONNX — eliminates extra ops in the graph.
        """
        for module in self.modules():
            if isinstance(module, LoRALinear):
                module.merge()
        print("[Model] LoRA weights merged into base model.")

    # ── Checkpoint helpers ────────────────────────────────────────────────────
    def save(self, path: str, extra: dict = None):
        payload = {"model_state": self.state_dict()}
        if extra:
            payload.update(extra)
        torch.save(payload, path)
        print(f"[Model] Saved → {path}")

    @classmethod
    def load(cls, path: str, num_classes: int = 0,
             rank: int = 16, alpha: float = 32.0):
        model = cls(num_classes=num_classes, rank=rank, alpha=alpha)
        ckpt  = torch.load(path, map_location="cpu")
        model.load_state_dict(ckpt["model_state"], strict=False)
        print(f"[Model] Loaded ← {path}")
        return model

    # ── Utility ───────────────────────────────────────────────────────────────
    def _print_param_summary(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[Model] Total params     : {total:,}")
        print(f"[Model] Trainable params : {trainable:,}  "
              f"({100 * trainable / total:.3f}%)")
