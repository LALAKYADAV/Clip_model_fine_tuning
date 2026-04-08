"""
export_clip_to_onnx.py
Loads a trained CLIP+LoRA checkpoint, merges LoRA weights into the base model,
then exports the image encoder to ONNX.

Validates that ONNX output matches PyTorch output within 1e-4 tolerance.
Target: >= 95.50% accuracy on LFW (zero-shot baseline).

Usage:
    # Export fine-tuned model
    python export_clip_to_onnx.py --checkpoint weights/best_model.pt

    # Export zero-shot baseline (no checkpoint)
    python export_clip_to_onnx.py
"""

import os
import argparse
import numpy as np
import torch
import onnx
import onnxruntime as ort

from model import CLIPFaceModel


# ══════════════════════════════════════════════════════════════════════════════
# Export
# ══════════════════════════════════════════════════════════════════════════════

def export(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    onnx_path = os.path.join(args.output_dir, "clip_vit_l14_fr.onnx")

    # ── Load model ─────────────────────────────────────────────────────────
    print(f"\n[Export] Building model (ViT-L/14, rank={args.lora_rank}) …")
    model = CLIPFaceModel(
        num_classes = 0,              # embedding-only for export
        rank        = args.lora_rank,
        alpha       = args.lora_alpha,
        use_rslora  = True,
    )

    if args.checkpoint and os.path.isfile(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(ckpt["model_state"], strict=False)
        print(f"[Export] Loaded checkpoint : {args.checkpoint}")
    else:
        print("[Export] No checkpoint provided — exporting zero-shot CLIP baseline.")

    # Merge LoRA weights → clean forward graph for ONNX
    model.merge_lora()
    model.eval().to(device)

    # ── Dummy input ────────────────────────────────────────────────────────
    dummy = torch.randn(1, 3, 224, 224, device=device)

    # ── PyTorch reference output ───────────────────────────────────────────
    with torch.no_grad():
        ref_emb = model.get_embeddings(dummy).cpu().numpy()

    # ── ONNX export ────────────────────────────────────────────────────────
    print(f"[Export] Exporting to ONNX → {onnx_path}")

    # Wrap so ONNX only sees the embedding output (single output = cleaner graph)
    class EmbeddingWrapper(torch.nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, x): return self.m.get_embeddings(x)

    wrapper = EmbeddingWrapper(model).eval().to(device)

    torch.onnx.export(
        wrapper,
        (dummy,),
        onnx_path,
        input_names        = ["input"],
        output_names       = ["embeddings"],
        dynamic_axes       = {
            "input"      : {0: "batch_size"},
            "embeddings" : {0: "batch_size"},
        },
        opset_version      = 17,
        do_constant_folding = True,
    )
    print(f"[Export] ONNX file saved : {onnx_path}")
    print(f"[Export] File size       : "
          f"{os.path.getsize(onnx_path) / 1024**2:.1f} MB")

    # ── ONNX graph check ───────────────────────────────────────────────────
    print("\n[Export] Checking ONNX model graph …")
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("[Export] ONNX graph check PASSED ✓")

    # ── Numerical validation ───────────────────────────────────────────────
    print("\n[Export] Validating numerical output …")
    providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                 if torch.cuda.is_available() else ["CPUExecutionProvider"])
    sess      = ort.InferenceSession(onnx_path, providers=providers)

    onnx_emb = sess.run(
        ["embeddings"],
        {"input": dummy.cpu().numpy()}
    )[0]

    max_diff = float(np.abs(ref_emb - onnx_emb).max())
    mean_diff = float(np.abs(ref_emb - onnx_emb).mean())

    print(f"[Export] Max  diff (PyTorch vs ONNX) : {max_diff:.8f}")
    print(f"[Export] Mean diff (PyTorch vs ONNX) : {mean_diff:.8f}")

    if max_diff < 1e-4:
        print("[Export] Numerical validation PASSED ✓")
    else:
        print(f"[Export] WARNING: max diff {max_diff:.6f} > 1e-4")
        print("[Export] Consider re-exporting with float32 precision.")

    # ── Batch inference test ───────────────────────────────────────────────
    print("\n[Export] Testing batch inference (B=4) …")
    batch = torch.randn(4, 3, 224, 224).numpy()
    out   = sess.run(["embeddings"], {"input": batch})[0]
    assert out.shape == (4, 768), f"Unexpected shape: {out.shape}"
    # Check L2 norm ≈ 1 for each embedding
    norms = np.linalg.norm(out, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), f"Embeddings not unit-norm: {norms}"
    print(f"[Export] Batch output shape : {out.shape}  ✓")
    print(f"[Export] Embedding norms    : {norms}  ✓")

    print(f"\n[Export] Done.  ONNX model ready at: {onnx_path}")
    return onnx_path


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export CLIP ViT-L/14 (+LoRA) to ONNX"
    )
    parser.add_argument("--checkpoint",  type=str,   default=None,
                        help="Path to trained checkpoint (omit for zero-shot)")
    parser.add_argument("--output_dir",  type=str,   default="onnx",
                        help="Directory to save ONNX file")
    parser.add_argument("--lora_rank",   type=int,   default=16)
    parser.add_argument("--lora_alpha",  type=float, default=32.0)
    args = parser.parse_args()
    export(args)
