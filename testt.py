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

Each .bin is in MXNet InsightFace format (same as LFW).
Reports per-benchmark accuracy and overall summary table.

Usage:
    python test.py
    python test.py --onnx_path onnx/clip_vit_l14_fr.onnx \
                   --eval_dir  data/eval
"""

import os
import io
import struct
import argparse
import numpy as np
import onnxruntime as ort
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


# ══════════════════════════════════════════════════════════════════════════════
# .bin reader  (MXNet InsightFace format)
# ══════════════════════════════════════════════════════════════════════════════

def read_bin(bin_path: str):
    """
    Parse MXNet InsightFace .bin verification file.
    Record layout:  [uint32 len1][img1][uint32 len2][img2][uint8 label]
    Returns:
        pairs  : list of (img1_bytes, img2_bytes)
        labels : list of int  (1 = same identity, 0 = different)
    """
    pairs  = []
    labels = []
    with open(bin_path, "rb") as f:
        data = f.read()

    idx = 0
    while idx < len(data) - 1:
        if idx + 4 > len(data): break
        l1 = struct.unpack_from("<I", data, idx)[0]; idx += 4
        if idx + l1 > len(data): break
        b1 = data[idx: idx + l1];                   idx += l1

        if idx + 4 > len(data): break
        l2 = struct.unpack_from("<I", data, idx)[0]; idx += 4
        if idx + l2 > len(data): break
        b2 = data[idx: idx + l2];                   idx += l2

        if idx + 1 > len(data): break
        lb = struct.unpack_from("B", data, idx)[0];  idx += 1

        pairs.append((b1, b2))
        labels.append(lb)

    return pairs, labels


# ══════════════════════════════════════════════════════════════════════════════
# Preprocessing
# ══════════════════════════════════════════════════════════════════════════════

CLIP_MEAN = [0.48145466, 0.4578275,  0.40821073]
CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]

_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
])


def img_bytes_to_numpy(img_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return _transform(img).unsqueeze(0).numpy()   # (1, 3, 224, 224)


# ══════════════════════════════════════════════════════════════════════════════
# Cosine similarity + threshold sweep
# ══════════════════════════════════════════════════════════════════════════════

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def best_accuracy(sims: np.ndarray, labels: np.ndarray,
                  step: float = 0.001):
    """Sweep thresholds, return (best_acc, best_threshold)."""
    best_acc, best_thr = 0.0, 0.0
    for thr in np.arange(sims.min(), sims.max(), step):
        preds = (sims >= thr).astype(int)
        acc   = (preds == labels).mean()
        if acc > best_acc:
            best_acc, best_thr = acc, thr
    return best_acc, best_thr


# ══════════════════════════════════════════════════════════════════════════════
# Evaluate one .bin
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_bin(sess, input_name: str, emb_name: str,
                 bin_path: str, name: str):
    pairs, labels = read_bin(bin_path)
    labels_arr    = np.array(labels)

    sims = []
    for b1, b2 in tqdm(pairs, desc=f"  {name:<12}", leave=False):
        t1  = img_bytes_to_numpy(b1)
        t2  = img_bytes_to_numpy(b2)
        e1  = sess.run([emb_name], {input_name: t1})[0][0]
        e2  = sess.run([emb_name], {input_name: t2})[0][0]
        sims.append(cosine_sim(e1, e2))

    sims_arr         = np.array(sims)
    acc, thr         = best_accuracy(sims_arr, labels_arr)
    return acc, thr, len(pairs)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

# Benchmarks and their minimum pass targets (None = informational only)
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


def run_evaluation(args):
    # ── Load ONNX session ──────────────────────────────────────────────────
    providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                 if args.use_gpu else ["CPUExecutionProvider"])
    print(f"\n[Test] Loading ONNX model : {args.onnx_path}")
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
        name = bin_file.replace(".bin", "")
        acc, thr, n_pairs = evaluate_bin(sess, input_name, emb_name,
                                         bin_path, name)
        results[name] = (acc, thr, n_pairs, target)

    # ── Summary table ──────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  {'Benchmark':<14}  {'Pairs':>6}  {'Acc':>8}  {'Thresh':>7}  Status")
    print(f"{'─'*60}")

    all_passed = True
    for name, (acc, thr, n, target) in results.items():
        if target is not None:
            passed  = acc >= target
            status  = "✓ PASS" if passed else "✗ FAIL"
            if not passed:
                all_passed = False
        else:
            status = "—"
        print(f"  {name:<14}  {n:>6}  {acc*100:>7.2f}%  {thr:>7.3f}  {status}")

    print(f"{'═'*60}")

    lfw_acc = results.get("lfw", (0,))[0]
    print(f"\n  LFW accuracy : {lfw_acc*100:.2f}%  "
          f"({'MEETS' if lfw_acc >= 0.955 else 'BELOW'} 95.50% baseline)")

    if all_passed:
        print("  Overall: ALL targets MET ✓")
    else:
        print("  Overall: Some targets not met — continue training.")

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
                        help="Force CPU inference")
    args = parser.parse_args()
    args.use_gpu = not args.no_gpu

    run_evaluation(args)
