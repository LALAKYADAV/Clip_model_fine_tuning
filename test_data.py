import struct
import numpy as np
import io
import os
from collections import namedtuple
from PIL import Image, ImageFile
import matplotlib.pyplot as plt

# ── Fix for "image file is truncated" error ──
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ─────────────────────────────────────────────
# Pure Python RecordIO Reader
# ─────────────────────────────────────────────

MAGIC     = 0xced7230a
IR_FORMAT = 'IfQQ'
IR_SIZE   = struct.calcsize(IR_FORMAT)   # 24 bytes
IRHeader  = namedtuple('IRHeader', ['flag', 'label', 'id', 'id2'])


def _read_chunk(f):
    header_bytes = f.read(8)
    if len(header_bytes) < 8:
        return None
    magic, length = struct.unpack('<II', header_bytes)
    assert magic == MAGIC, f"Bad magic: 0x{magic:08x}"
    data = f.read(length - 8)
    pad  = (4 - (length % 4)) % 4
    f.read(pad)
    return data


def unpack(data):
    header = IRHeader(*struct.unpack(IR_FORMAT, data[:IR_SIZE]))
    s = data[IR_SIZE:]
    if header.flag > 0:
        label  = np.frombuffer(s[:header.flag * 4], dtype=np.float32).copy()
        header = header._replace(label=label)
        s      = s[header.flag * 4:]
    return header, s


def unpack_img(data):
    header, img_bytes = unpack(data)
    img = Image.open(io.BytesIO(img_bytes))
    img.load()          # force full decode here so errors surface clearly
    img = img.convert('RGB')   # normalise palette/grayscale → RGB
    return header, img


class MXIndexedRecordIO:
    def __init__(self, idx_path, rec_path):
        self.idx = {}
        with open(idx_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                key, offset = line.split('\t')
                self.idx[int(key)] = int(offset)
        self.f = open(rec_path, 'rb')

    def read_idx(self, idx):
        self.f.seek(self.idx[idx])
        return _read_chunk(self.f)

    def keys(self):
        return sorted(self.idx.keys())

    def close(self):
        self.f.close()

    def __enter__(self): return self
    def __exit__(self, *a): self.close()


# ─────────────────────────────────────────────
# Paths  ← adjust if needed
# ─────────────────────────────────────────────
REC_PATH = r"D:\clip\IJCB-AFMFR-2026\data\casia-webface\train.rec"
IDX_PATH = r"D:\clip\IJCB-AFMFR-2026\data\casia-webface\train.idx"

# ─────────────────────────────────────────────
# Verify + Plot
# ─────────────────────────────────────────────
print("Opening record file …")
record = MXIndexedRecordIO(IDX_PATH, REC_PATH)
print(f"  Index keys loaded: {len(record.idx):,}")

# Read dataset header (index 0)
try:
    hdr, _ = unpack(record.read_idx(0))
    lbl = hdr.label
    if hasattr(lbl, '__len__') and len(lbl) >= 2:
        print(f"  Total images : {int(lbl[0]):,}")
        print(f"  Num classes  : {int(lbl[1]):,}")
except Exception as e:
    print(f"  Header: {e}")

# Plot 16 samples
fig, axes = plt.subplots(4, 4, figsize=(12, 12))
fig.patch.set_facecolor('#111')

ok, errors = 0, 0
for i, ax in enumerate(axes.flatten()):
    try:
        h, img = unpack_img(record.read_idx(i + 1))
        lbl = h.label
        if hasattr(lbl, '__len__'): lbl = lbl[0]
        ax.imshow(img)
        ax.set_title(f"ID: {int(lbl)}", fontsize=9, color='white')
        ok += 1
    except Exception as e:
        ax.text(0.5, 0.5, f"ERR\n{str(e)[:30]}",
                ha='center', va='center', color='red',
                fontsize=7, transform=ax.transAxes)
        errors += 1
    ax.axis('off')

plt.suptitle("CASIA-WebFace — Sample Images", color='white', fontsize=14)
plt.tight_layout()
plt.savefig("sample_output.png", dpi=150, bbox_inches='tight', facecolor='#111')
plt.show()

print(f"\n✅ {ok}/16 images rendered successfully")
if errors:
    print(f"⚠️  {errors} failed — check paths or file integrity")

record.close()