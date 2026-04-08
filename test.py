"""
test.py
Evaluates the ONNX model on ALL evaluation .bin benchmarks:

    data/eval/
        lfw.bin         → target >= 95.50%
        agedb_30.bin
        calfw.bin
        cfp_ff.bin
        cfp_fp.bin
        cplfw.bin
        sllfw.bin
        talfw.bin

Each .bin is in MXNet InsightFace pickle format:
    bins, issame_list = pickle.load(f, encoding='bytes')

Reports per-benchmark accuracy + overall summary table.

Usage:
    python test.py
    python test.py --onnx_path onnx/clip_vit_l14_fr.onnx \
                   --eval_dir  data/eval
"""

import os
import io
import pickle
import argparse
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageFile
from torchvision import transforms
from tqdm import tqdm

# Allow slightly truncated images (common in eval .bin files)
ImageFile.LOAD_TRUNCATED_IMAGES = True


# ══════════════════════════════════════════════════════════════════════════════
# .bin reader  — InsightFace pickle format
# ══════════════════════════════════════════════════════════════════════════════

def read_bin(bin_path: str):
    """
    Load an InsightFace .bin verification file.

    Format:
        bins        — list of raw JPEG bytes, two per pair (img1, img2, img3, img4 …)
        issame_list — list of bool/int  (True = same identity)

    Returns:
        pairs  : list of (img1_bytes, img2_bytes)
        labels : list of int  (1 = same, 0 = different)
    """
    with open(bin_path, 'rb') as f:
        # Try numpy pickle first (most InsightFace bins), fall back to plain pickle
        try:
            bins, issame_list = pickle.load(f, encoding='bytes')
        except Exception:
            f.seek(0)
            bins, issame_list = pickle.load(f)

    pairs  = [(bins[i * 2],     bins[i * 2 + 1])
              for i in range(len(issame_list))]
    labels = [int(v) for v in issame_list]
    return pairs, labels


# ══════════════════════════════════════════════════════════════════════════════
# Preprocessing  (same CLIP stats as training)
# ══════════════════════════════════════════════════════════════════════════════

CLIP_MEAN = [0.48145466, 0.4578275,  0.40821073]
CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]

_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
])


def img_bytes_to_numpy(img_bytes: bytes) -> np.ndarray:
    """Decode raw JPEG bytes → normalised numpy array (1, 3, 224, 224)."""
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    return _transform(img).unsqueeze(0).numpy()


# ══════════════════════════════════════════════════════════════════════════════
# Cosine similarity + threshold sweep
# ══════════════════════════════════════════════════════════════════════════════

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def best_accuracy(sims: np.ndarray, labels: np.ndarray,
                  step: float = 0.001):
    """
    Sweep thresholds from min→max similarity.
    Returns (best_accuracy, best_threshold).
    """
    best_acc, best_thr = 0.0, 0.0
    for thr in np.arange(float(sims.min()), float(sims.max()), step):
        preds = (sims >= thr).astype(int)
        acc   = (preds == labels).mean()
        if acc > best_acc:
            best_acc, best_thr = float(acc), float(thr)
    return best_acc, best_thr


# ══════════════════════════════════════════════════════════════════════════════
# Evaluate one .bin file
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_bin(sess, input_name: str, emb_name: str,
                 bin_path: str, name: str):
    pairs, labels = read_bin(bin_path)
    labels_arr    = np.array(labels)

    sims = []
    for b1, b2 in tqdm(pairs, desc=f"  {name:<14}", leave=False):
        t1 = img_bytes_to_numpy(b1)
        t2 = img_bytes_to_numpy(b2)
        e1 = sess.run([emb_name], {input_name: t1})[0][0]
        e2 = sess.run([emb_name], {input_name: t2})[0][0]
        sims.append(cosine_sim(e1, e2))

    sims_arr      = np.array(sims)
    acc, thr      = best_accuracy(sims_arr, labels_arr)
    return acc, thr, len(pairs)


# ══════════════════════════════════════════════════════════════════════════════
# Benchmark registry
# ══════════════════════════════════════════════════════════════════════════════

# key = filename,  value = minimum accuracy target (None = informational only)
BENCHMARKS = {
    "lfw.bin"      : 0.9550,   # competition baseline target
    "agedb_30.bin" : None,
    "calfw.bin"    : None,
    "cfp_ff.bin"   : None,
    "cfp_fp.bin"   : None,
    "cplfw.bin"    : None,
    "sllfw.bin"    : None,
    "talfw.bin"    : None,
}


# ══════════════════════════════════════════════════════════════════════════════
# Main evaluation routine
# ══════════════════════════════════════════════════════════════════════════════

def run_evaluation(args):
    # ── Load ONNX session ─────────────────────────────────────────────────────
    providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                 if args.use_gpu else ["CPUExecutionProvider"])

    print(f"\n[Test] Loading ONNX model : {args.onnx_path}")
    if not os.path.isfile(args.onnx_path):
        raise FileNotFoundError(f"ONNX model not found: {args.onnx_path}")

    sess       = ort.InferenceSession(args.onnx_path, providers=providers)
    input_name = sess.get_inputs()[0].name
    emb_name   = sess.get_outputs()[0].name

    print(f"[Test] Input  : {input_name}  {sess.get_inputs()[0].shape}")
    print(f"[Test] Output : {emb_name}   {sess.get_outputs()[0].shape}")
    print(f"[Test] Eval   : {args.eval_dir}\n")

    results = {}

    for bin_file, target in BENCHMARKS.items():
        bin_path = os.path.join(args.eval_dir, bin_file)
        if not os.path.isfile(bin_path):
            print(f"  [skip] {bin_file} not found")
            continue
        name              = bin_file.replace(".bin", "")
        acc, thr, n_pairs = evaluate_bin(sess, input_name, emb_name,
                                         bin_path, name)
        results[name]     = (acc, thr, n_pairs, target)

    if not results:
        print("[Test] No .bin files found — check --eval_dir path.")
        return {}

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'═'*62}")
    print(f"  {'Benchmark':<14}  {'Pairs':>6}  {'Accuracy':>9}  {'Thresh':>7}  Status")
    print(f"{'─'*62}")

    all_passed = True
    for name, (acc, thr, n, target) in results.items():
        if target is not None:
            passed = acc >= target
            status = "✓ PASS" if passed else "✗ FAIL"
            if not passed:
                all_passed = False
        else:
            status = "—"
        print(f"  {name:<14}  {n:>6}  {acc*100:>8.2f}%  {thr:>7.3f}  {status}")

    print(f"{'═'*62}")

    lfw_acc = results.get("lfw", (0,))[0]
    print(f"\n  LFW accuracy : {lfw_acc*100:.2f}%  "
          f"({'MEETS' if lfw_acc >= 0.955 else 'BELOW'} 95.50% target)")

    if all_passed:
        print("  Overall      : ALL targets met ✓")
    else:
        print("  Overall      : Some targets not yet met — continue training.")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate CLIP ONNX model on all face verification benchmarks"
    )
    parser.add_argument("--onnx_path", type=str,
                        default="onnx/clip_vit_l14_fr.onnx",
                        help="Path to exported ONNX model")
    parser.add_argument("--eval_dir",  type=str,
                        default="data/eval",
                        help="Directory containing .bin evaluation files")
    parser.add_argument("--no_gpu",    action="store_true",
                        help="Force CPU-only inference")
    args        = parser.parse_args()
    args.use_gpu = not args.no_gpu

    run_evaluation(args)
