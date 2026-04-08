"""
train.py
LoRA fine-tuning for CLIP ViT-L/14 face recognition.

Supports:
  - CASIA-WebFace  (RecordIO .rec format)
  - SFace2         (standard image folder)
  - Both combined with automatic label remapping

Usage examples:
    # CASIA only
    python train.py --casia_root data/casia-webface

    # SFace2 only
    python train.py --sface2_root data/sface2

    # Both datasets combined
    python train.py --casia_root data/casia-webface \
                    --sface2_root data/sface2

    # Quick subset test
    python train.py --casia_root data/casia-webface \
                    --max_ids 1000
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import build_loader          # ← updated import
from model   import CLIPFaceModel


# ══════════════════════════════════════════════════════════════════════════════
# ArcFace Loss
# ══════════════════════════════════════════════════════════════════════════════

class ArcFaceLoss(nn.Module):
    """
    Additive Angular Margin Loss (ArcFace).
    Ref: Deng et al., CVPR 2019.
    s=64, m=0.5 are standard competition settings.
    """

    def __init__(self, in_features: int, num_classes: int,
                 s: float = 64.0, m: float = 0.5):
        import math
        super().__init__()
        self.s      = s
        self.m      = m
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th    = math.cos(math.pi - m)
        self.mm    = math.sin(math.pi - m) * m
        self.ce    = nn.CrossEntropyLoss()

    def forward(self, embeddings, labels):
        import torch.nn.functional as F
        cosine  = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        sine    = torch.sqrt(1.0 - torch.clamp(cosine ** 2, 0.0, 1.0))
        phi     = cosine * self.cos_m - sine * self.sin_m
        phi     = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        output  = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return self.ce(output, labels)


# ══════════════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════════════

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Train] Device : {device}")
    if device.type == "cuda":
        print(f"[Train] GPU    : {torch.cuda.get_device_name(0)}")

    # ── Data ──────────────────────────────────────────────────────────────────
    loader, num_classes = build_loader(
        casia_root             = args.casia_root,
        sface2_root            = args.sface2_root,
        img_size               = 224,
        batch_size             = args.batch_size,
        num_workers            = args.num_workers,
        max_identities_casia   = args.max_ids,
        max_identities_sface2  = args.max_ids,
        mode                   = "train",
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = CLIPFaceModel(
        num_classes = num_classes,
        rank        = args.lora_rank,
        alpha       = args.lora_alpha,
        use_rslora  = True,
    ).to(device)

    # ── Loss ──────────────────────────────────────────────────────────────────
    criterion = ArcFaceLoss(
        in_features = 768,
        num_classes = num_classes,
        s           = 64.0,
        m           = 0.5,
    ).to(device)

    # ── Optimiser: only LoRA params + ArcFace weight ──────────────────────────
    trainable = (
        list(filter(lambda p: p.requires_grad, model.parameters())) +
        list(criterion.parameters())
    )
    optimizer = optim.AdamW(trainable, lr=args.lr,
                            weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    scaler = torch.amp.GradScaler('cuda')
    writer = SummaryWriter(log_dir=os.path.join(args.save_dir, "tb_logs"))
    os.makedirs(args.save_dir, exist_ok=True)

    # ── Resume from checkpoint ────────────────────────────────────────────────
    start_epoch = 0
    best_loss   = float("inf")
    if args.resume and os.path.isfile(args.resume):
        ckpt        = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"], strict=False)
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_loss   = ckpt.get("loss",  float("inf"))
        print(f"[Train] Resumed from {args.resume}  (epoch {start_epoch})")

    # ── Epoch loop ─────────────────────────────────────────────────────────────
    best_path    = os.path.join(args.save_dir, "best_model.pt")
    global_step  = 0

    for epoch in range(start_epoch, args.epochs):
        model.train()
        criterion.train()
        running_loss = 0.0

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=True)
        for imgs, labels in pbar:
            imgs   = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda'):
                _logits, embeddings = model(imgs)
                loss = criterion(embeddings, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            global_step  += 1

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                lr  =f"{scheduler.get_last_lr()[0]:.2e}"
            )
            writer.add_scalar("Loss/step", loss.item(), global_step)

        scheduler.step()
        epoch_loss = running_loss / len(loader)
        writer.add_scalar("Loss/epoch", epoch_loss, epoch)
        writer.add_scalar("LR/epoch",   scheduler.get_last_lr()[0], epoch)

        print(f"[Epoch {epoch+1:03d}] avg loss: {epoch_loss:.4f}  "
              f"lr: {scheduler.get_last_lr()[0]:.2e}")

        # ── Save best ──────────────────────────────────────────────────────
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save({
                "epoch"          : epoch,
                "model_state"    : model.state_dict(),
                "criterion_state": criterion.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "loss"           : best_loss,
                "num_classes"    : num_classes,
                "lora_rank"      : args.lora_rank,
                "lora_alpha"     : args.lora_alpha,
            }, best_path)
            print(f"  → Best checkpoint saved: {best_path}  (loss {best_loss:.4f})")

        # ── Periodic save ──────────────────────────────────────────────────
        if (epoch + 1) % args.save_every == 0:
            ep_path = os.path.join(args.save_dir, f"epoch_{epoch+1:03d}.pt")
            torch.save({"epoch": epoch, "model_state": model.state_dict()},
                       ep_path)

    writer.close()
    print(f"\n[Train] Finished.  Best loss: {best_loss:.4f}")
    print(f"[Train] Best model: {best_path}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LoRA fine-tuning of CLIP ViT-L/14 for face recognition"
    )

    # ── Dataset ───────────────────────────────────────────────────────────────
    g = parser.add_argument_group("Dataset")
    g.add_argument("--casia_root",  type=str, default=None,
                   help="Path to data/casia-webface/ (RecordIO format)")
    g.add_argument("--sface2_root", type=str, default=None,
                   help="Path to data/sface2/ (image folder format)")
    g.add_argument("--max_ids",     type=int, default=None,
                   help="Cap identities per dataset — useful for quick tests")

    # ── Training ──────────────────────────────────────────────────────────────
    g = parser.add_argument_group("Training")
    g.add_argument("--epochs",       type=int,   default=20)
    g.add_argument("--batch_size",   type=int,   default=128)
    g.add_argument("--lr",           type=float, default=1e-4)
    g.add_argument("--weight_decay", type=float, default=1e-4)
    g.add_argument("--num_workers",  type=int,   default=4,
                   help="Use 0 on Windows to avoid multiprocessing issues")
    g.add_argument("--resume",       type=str,   default=None,
                   help="Path to checkpoint to resume from")

    # ── LoRA ──────────────────────────────────────────────────────────────────
    g = parser.add_argument_group("LoRA")
    g.add_argument("--lora_rank",  type=int,   default=16)
    g.add_argument("--lora_alpha", type=float, default=32.0)

    # ── Saving ────────────────────────────────────────────────────────────────
    g = parser.add_argument_group("Saving")
    g.add_argument("--save_dir",   type=str, default="weights")
    g.add_argument("--save_every", type=int, default=5)

    args = parser.parse_args()

    if args.casia_root is None and args.sface2_root is None:
        parser.error("Provide at least one of --casia_root or --sface2_root")

    train(args)
