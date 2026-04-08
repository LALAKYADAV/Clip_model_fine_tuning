"""
dataset.py
==========
Unified dataset loader for face recognition. No MXNet required.

Supported datasets
------------------
  1. CASIA-WebFace  — MXNet RecordIO format  (train.rec + train.idx)
  2. SFace2         — Standard image folder  (<id>/*.png)

Pure-Python RecordIO reader is built-in (reverse-engineered from MXNet source).
Labels are globally remapped so datasets can be safely combined.
"""

import os
import io
import struct
import numpy as np
from collections import namedtuple
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms

# Allow slightly truncated JPEG/PNG files (common in packed .rec files)
ImageFile.LOAD_TRUNCATED_IMAGES = True


# ══════════════════════════════════════════════════════════════════════════════
# Transforms  (CLIP ViT-L/14 normalisation stats)
# ══════════════════════════════════════════════════════════════════════════════

CLIP_MEAN = [0.48145466, 0.4578275,  0.40821073]
CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]


def get_train_transforms(img_size: int = 224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2,
                               saturation=0.2, hue=0.05),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])


def get_eval_transforms(img_size: int = 224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])


# ══════════════════════════════════════════════════════════════════════════════
# Pure-Python MXNet RecordIO reader  (no MXNet installation needed)
# ══════════════════════════════════════════════════════════════════════════════

_MX_MAGIC  = 0xced7230a          # magic number stamped on every RecordIO chunk
_IR_FORMAT = 'IfQQ'              # flag(uint32) label(float32) id(uint64) id2(uint64)
_IR_SIZE   = struct.calcsize(_IR_FORMAT)   # = 24 bytes
_IRHeader  = namedtuple('IRHeader', ['flag', 'label', 'id', 'id2'])


def _read_chunk(fp, offset: int):
    """
    Seek to `offset` and read one raw RecordIO chunk.
    RecordIO chunk layout:
        [uint32 magic][uint32 length][body: length-8 bytes][padding to 4-byte align]
    Returns raw body bytes, or None on failure.
    """
    fp.seek(offset)
    hdr = fp.read(8)
    if len(hdr) < 8:
        return None
    magic, length = struct.unpack('<II', hdr)
    if magic != _MX_MAGIC:
        return None
    body = fp.read(length - 8)
    pad  = (4 - (length % 4)) % 4
    fp.read(pad)
    return body


def _unpack(data: bytes):
    """
    Parse body bytes → (IRHeader, remaining_bytes).
    If header.flag > 0, label is a float32 array of length flag.
    """
    header = _IRHeader(*struct.unpack(_IR_FORMAT, data[:_IR_SIZE]))
    rest   = data[_IR_SIZE:]
    if header.flag > 0:
        arr    = np.frombuffer(rest[:header.flag * 4], dtype=np.float32).copy()
        header = header._replace(label=arr)
        rest   = rest[header.flag * 4:]
    return header, rest


def _unpack_img(data: bytes):
    """Parse body bytes → (IRHeader, PIL Image in RGB)."""
    header, img_bytes = _unpack(data)
    img = Image.open(io.BytesIO(img_bytes))
    img.load()
    img = img.convert('RGB')
    return header, img


def _get_label(header_label):
    """Safely extract a scalar int label from IRHeader.label."""
    lbl = header_label
    if hasattr(lbl, '__len__'):
        lbl = lbl[0]
    return int(float(lbl))


class _MXIndexedRecordIO:
    """
    Pure-Python indexed RecordIO reader.
    Reads the text .idx file (tab-separated: key offset) that ships
    alongside CASIA-WebFace / InsightFace datasets.
    """

    def __init__(self, idx_path: str, rec_path: str):
        self.rec_path = rec_path
        self.idx: dict[int, int] = {}

        with open(idx_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                key, offset = line.split('\t')
                self.idx[int(key)] = int(offset)

        self._fp = open(rec_path, 'rb')

    def read_idx(self, idx: int):
        return _read_chunk(self._fp, self.idx[idx])

    def sorted_keys(self):
        return sorted(self.idx.keys())

    def close(self):
        self._fp.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ══════════════════════════════════════════════════════════════════════════════
# Dataset 1 — CASIA-WebFace  (MXNet RecordIO)
# ══════════════════════════════════════════════════════════════════════════════

class CASIAWebFaceDataset(Dataset):
    """
    Reads CASIA-WebFace from MXNet RecordIO files without MXNet.

    Expected folder layout::

        <root>/
            property          ← "num_classes,h,w"  (optional but recommended)
            train.idx         ← text index: "key\\toffset\\n" per line
            train.rec         ← packed JPEG images

    Args:
        root           : path to the casia-webface folder
        transform      : torchvision transform applied to each image
        max_identities : cap the number of identities loaded (None = all)
        label_offset   : added to every label (for multi-dataset merging)
    """

    def __init__(
        self,
        root: str,
        transform=None,
        max_identities: int = None,
        label_offset: int = 0,
    ):
        self.rec_path     = os.path.join(root, 'train.rec')
        self.idx_path     = os.path.join(root, 'train.idx')
        self.transform    = transform
        self.label_offset = label_offset

        if not os.path.isfile(self.rec_path):
            raise FileNotFoundError(f'train.rec not found: {self.rec_path}')
        if not os.path.isfile(self.idx_path):
            raise FileNotFoundError(f'train.idx not found: {self.idx_path}')

        # ── Read property file ──────────────────────────────────────────────
        self.num_classes = 10575   # CASIA-WebFace default
        prop_path = os.path.join(root, 'property')
        if os.path.isfile(prop_path):
            with open(prop_path) as f:
                parts = f.read().strip().split(',')
            try:
                self.num_classes = int(parts[0])
                print(f'[CASIA] property → {self.num_classes} classes, '
                      f'{parts[1]}×{parts[2]} px')
            except Exception:
                pass

        # ── Load index ──────────────────────────────────────────────────────
        print(f'[CASIA] Loading index from {self.idx_path} …')
        record = _MXIndexedRecordIO(self.idx_path, self.rec_path)
        all_keys = record.sorted_keys()
        print(f'[CASIA] Index entries: {len(all_keys):,}')

        # Key 0 is a metadata header record — read dataset info then skip it
        try:
            hdr_data = record.read_idx(0)
            hdr, _   = _unpack(hdr_data)
            lbl       = hdr.label
            if hasattr(lbl, '__len__') and len(lbl) >= 2:
                print(f'[CASIA] Header → total images: {int(lbl[0]):,}, '
                      f'classes: {int(lbl[1]):,}')
        except Exception as e:
            print(f'[CASIA] Could not read header record: {e}')

        image_keys = [k for k in all_keys if k > 0]

        # ── Build sample list (offset, remapped_label) ──────────────────────
        print(f'[CASIA] Scanning {len(image_keys):,} records …')
        self.samples: list[tuple[int, int]] = []
        seen_ids:     dict[int, int]        = {}

        for key in image_keys:
            try:
                data   = record.read_idx(key)
                hdr, _ = _unpack(data)
                orig   = _get_label(hdr.label)
            except Exception:
                continue

            if max_identities is not None:
                if orig not in seen_ids:
                    if len(seen_ids) >= max_identities:
                        continue
                    seen_ids[orig] = len(seen_ids)
                mapped = seen_ids[orig]
            else:
                mapped = orig

            offset = record.idx[key]
            self.samples.append((offset, mapped + label_offset))

            if len(self.samples) % 100_000 == 0:
                print(f'[CASIA]   … scanned {len(self.samples):,}')

        record.close()

        n_ids = len(seen_ids) if max_identities else self.num_classes
        print(f'[CASIA] Identities loaded : {n_ids:,}')
        print(f'[CASIA] Total samples     : {len(self.samples):,}')

        if not self.samples:
            raise RuntimeError('[CASIA] No valid samples found — '
                               'check file integrity.')

        # Keep a persistent file handle for __getitem__
        self._fp = open(self.rec_path, 'rb')

    # ── PyTorch Dataset interface ───────────────────────────────────────────

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        offset, label = self.samples[index]
        data          = _read_chunk(self._fp, offset)
        _, img        = _unpack_img(data)
        if self.transform:
            img = self.transform(img)
        return img, label

    def __del__(self):
        if hasattr(self, '_fp') and self._fp:
            try:
                self._fp.close()
            except Exception:
                pass


# ══════════════════════════════════════════════════════════════════════════════
# Dataset 2 — SFace2 / any image-folder dataset
# ══════════════════════════════════════════════════════════════════════════════

class SFace2Dataset(Dataset):
    """
    Loads any image-folder dataset where sub-folders are identity IDs.

    Expected layout::

        <root>/
            000001/  img_001.png  img_002.png …
            000002/  …
            …

    Args:
        root           : path to the dataset root folder
        transform      : torchvision transform
        max_identities : cap the number of identity folders (None = all)
        label_offset   : added to every label (for multi-dataset merging)
    """

    VALID_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    def __init__(
        self,
        root: str,
        transform=None,
        max_identities: int = None,
        label_offset: int = 0,
    ):
        self.transform = transform
        self.samples: list[tuple[str, int]] = []

        if not os.path.isdir(root):
            raise FileNotFoundError(f'SFace2 root not found: {root}')

        print(f'\n[SFace2] Loading from {root} …')

        identities = sorted([
            d for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d))
        ])

        if max_identities is not None:
            identities = identities[:max_identities]

        for idx, identity in enumerate(identities):
            id_dir = os.path.join(root, identity)
            for fname in os.listdir(id_dir):
                if os.path.splitext(fname)[1].lower() in self.VALID_EXTS:
                    self.samples.append(
                        (os.path.join(id_dir, fname), idx + label_offset)
                    )

        print(f'[SFace2] Identities loaded : {len(identities):,}')
        print(f'[SFace2] Total samples     : {len(self.samples):,}')

        if not self.samples:
            raise RuntimeError(f'No images found in SFace2 root: {root}')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


# ══════════════════════════════════════════════════════════════════════════════
# Combined DataLoader factory
# ══════════════════════════════════════════════════════════════════════════════

def build_loader(
    casia_root:              str  = None,
    sface2_root:             str  = None,
    img_size:                int  = 224,
    batch_size:              int  = 128,
    num_workers:             int  = 4,
    max_identities_casia:    int  = None,
    max_identities_sface2:   int  = None,
    mode:                    str  = 'train',   # 'train' | 'eval'
):
    """
    Build a DataLoader from any combination of CASIA-WebFace and SFace2.
    Labels are globally remapped so class indices never collide.

    Returns
    -------
    loader      : torch.utils.data.DataLoader
    num_classes : total unique identities across all datasets loaded
    """
    transform = (get_train_transforms(img_size)
                 if mode == 'train' else get_eval_transforms(img_size))

    datasets = []
    offset   = 0

    # ── CASIA-WebFace ────────────────────────────────────────────────────────
    if casia_root:
        if not os.path.isdir(casia_root):
            print(f'[Warning] CASIA root not found, skipping: {casia_root}')
        else:
            print('\n── Loading CASIA-WebFace ──────────────────────────')
            ds = CASIAWebFaceDataset(
                root           = casia_root,
                transform      = transform,
                max_identities = max_identities_casia,
                label_offset   = offset,
            )
            used = len({lbl for _, lbl in ds.samples})
            offset += used
            datasets.append(ds)

    # ── SFace2 ───────────────────────────────────────────────────────────────
    if sface2_root:
        if not os.path.isdir(sface2_root):
            print(f'[Warning] SFace2 root not found, skipping: {sface2_root}')
        else:
            print('\n── Loading SFace2 ─────────────────────────────────')
            ds = SFace2Dataset(
                root           = sface2_root,
                transform      = transform,
                max_identities = max_identities_sface2,
                label_offset   = offset,
            )
            used = len({lbl for _, lbl in ds.samples})
            offset += used
            datasets.append(ds)

    if not datasets:
        raise ValueError(
            'No valid dataset found. Pass at least one of '
            'casia_root or sface2_root.'
        )

    combined    = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
    num_classes = offset

    loader = DataLoader(
        combined,
        batch_size         = batch_size,
        shuffle            = (mode == 'train'),
        num_workers        = num_workers,
        pin_memory         = True,
        drop_last          = (mode == 'train'),
        persistent_workers = (num_workers > 0),
    )

    print(f"\n{'═'*55}")
    print(f"  DataLoader ready")
    print(f"  Mode        : {mode}")
    print(f"  Datasets    : {len(datasets)}")
    print(f"  Total IDs   : {num_classes:,}")
    print(f"  Total imgs  : {len(combined):,}")
    print(f"  Batch size  : {batch_size}")
    print(f"{'═'*55}\n")

    return loader, num_classes


# ══════════════════════════════════════════════════════════════════════════════
# Verification / smoke-test
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import argparse
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description='Dataset verification')
    parser.add_argument('--casia_root',   type=str, default=None,
                        help='Path to casia-webface folder')
    parser.add_argument('--sface2_root',  type=str, default=None,
                        help='Path to SFace2 root folder')
    parser.add_argument('--batch_size',   type=int, default=8)
    parser.add_argument('--num_workers',  type=int, default=0,
                        help='0 recommended for Windows debugging')
    parser.add_argument('--max_ids',      type=int, default=None,
                        help='Limit identities per dataset for quick test')
    parser.add_argument('--plot_samples', action='store_true',
                        help='Save a 4×4 sample grid as sample_output.png')
    args = parser.parse_args()

    if not args.casia_root and not args.sface2_root:
        # ── Default quick-test: CASIA only, first 10 identities ─────────────
        print('[INFO] No args given — running internal RecordIO unit test.')
        print('[INFO] Pass --casia_root or --sface2_root for a real test.\n')

        # Minimal round-trip test with the RecordIO helpers themselves
        import tempfile, random

        def _make_fake_jpeg():
            """Create a tiny valid JPEG in memory."""
            buf = io.BytesIO()
            Image.new('RGB', (32, 32), color=(
                random.randint(0,255),
                random.randint(0,255),
                random.randint(0,255),
            )).save(buf, format='JPEG')
            return buf.getvalue()

        def _pack(header, data: bytes) -> bytes:
            hdr = _IRHeader(*header)
            if isinstance(hdr.label, (int, float)):
                flag  = 0
                label = float(hdr.label)
                extra = b''
            else:
                arr   = np.asarray(hdr.label, dtype=np.float32)
                flag  = len(arr)
                label = 0.0
                extra = arr.tobytes()
            packed_hdr = struct.pack(_IR_FORMAT, flag, label, hdr.id, hdr.id2)
            body       = packed_hdr + extra + data
            length     = 8 + len(body)
            pad        = (4 - (length % 4)) % 4
            return struct.pack('<II', _MX_MAGIC, length) + body + b'\x00' * pad

        with tempfile.TemporaryDirectory() as tmpdir:
            rec_path = os.path.join(tmpdir, 'test.rec')
            idx_path = os.path.join(tmpdir, 'test.idx')

            # Write 5 fake records
            idx_lines = []
            with open(rec_path, 'wb') as rf:
                for i in range(1, 6):
                    offset = rf.tell()
                    idx_lines.append(f'{i}\t{offset}\n')
                    chunk  = _pack((0, float(i * 10), i, 0), _make_fake_jpeg())
                    rf.write(chunk)

            with open(idx_path, 'w') as idf:
                idf.writelines(idx_lines)

            rec = _MXIndexedRecordIO(idx_path, rec_path)
            for key in rec.sorted_keys():
                data        = rec.read_idx(key)
                hdr, img    = _unpack_img(data)
                label       = _get_label(hdr.label)
                print(f'  key={key}  label={label}  img={img.size} {img.mode}  ✅')
            rec.close()

        print('\n✅ RecordIO unit-test PASSED — internal reader works correctly.')

    else:
        # ── Real dataset test ────────────────────────────────────────────────
        loader, num_classes = build_loader(
            casia_root  = args.casia_root,
            sface2_root = args.sface2_root,
            batch_size  = args.batch_size,
            num_workers = args.num_workers,
            max_identities_casia  = args.max_ids,
            max_identities_sface2 = args.max_ids,
            mode        = 'train',
        )

        print('[TEST] Fetching one batch …')
        imgs, labels = next(iter(loader))

        print(f'  Batch shape  : {imgs.shape}')
        print(f'  Label range  : {labels.min().item()} – {labels.max().item()}')
        print(f'  Num classes  : {num_classes:,}')
        print('✅ DataLoader test PASSED')

        if args.plot_samples:
            # Denormalise and plot
            mean = torch.tensor(CLIP_MEAN).view(3,1,1)
            std  = torch.tensor(CLIP_STD).view(3,1,1)
            import torch
            denorm = (imgs[:16] * std + mean).clamp(0, 1)

            fig, axes = plt.subplots(4, 4, figsize=(12, 12))
            fig.patch.set_facecolor('#111')
            for i, ax in enumerate(axes.flatten()):
                if i < len(denorm):
                    ax.imshow(denorm[i].permute(1,2,0).numpy())
                    ax.set_title(f'ID: {labels[i].item()}',
                                 fontsize=9, color='white')
                ax.axis('off')
            plt.suptitle('Sample Images', color='white', fontsize=14)
            plt.tight_layout()
            plt.savefig('sample_output.png', dpi=150,
                        bbox_inches='tight', facecolor='#111')
            print('💾 Saved → sample_output.png')





































# import os
# import io
# from PIL import Image
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms

# # ══════════════════════════════════════════════════════════════════════════════
# # Transforms (CLIP ViT-L/14 stats)
# # ══════════════════════════════════════════════════════════════════════════════
# CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
# CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

# def get_train_transforms(img_size: int = 224):
#     return transforms.Compose([
#         transforms.Resize((img_size, img_size)),
#         transforms.RandomHorizontalFlip(),
#         transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
#         transforms.RandomGrayscale(p=0.05),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
#     ])

# def get_eval_transforms(img_size: int = 224):
#     return transforms.Compose([
#         transforms.Resize((img_size, img_size)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
#     ])

# # ══════════════════════════════════════════════════════════════════════════════
# # SFace2 Dataset (Standard Image Folder Structure)
# # ══════════════════════════════════════════════════════════════════════════════
# class SFace2Dataset(Dataset):
#     """
#     Loads SFace2 dataset from folder structure:
#     data/sface2/
#         000001/   *.png
#         000002/   *.png
#         ...
#     """
#     VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

#     def __init__(self, root: str, transform=None, max_identities: int = None):
#         self.transform = transform
#         self.samples = []  # list of (image_path, label)

#         if not os.path.isdir(root):
#             raise FileNotFoundError(f"SFace2 root not found: {root}")

#         print(f"\n[INFO] Loading SFace2 from {root} ...")

#         # Get sorted identity folders
#         identities = sorted([
#             d for d in os.listdir(root)
#             if os.path.isdir(os.path.join(root, d))
#         ])

#         if max_identities is not None:
#             identities = identities[:max_identities]

#         for idx, identity_folder in enumerate(identities):
#             id_dir = os.path.join(root, identity_folder)
#             for fname in os.listdir(id_dir):
#                 ext = os.path.splitext(fname)[1].lower()
#                 if ext in self.VALID_EXTS:
#                     self.samples.append((os.path.join(id_dir, fname), idx))

#         print(f"[INFO] Identities used : {len(identities)}")
#         print(f"[INFO] Total images    : {len(self.samples)}")

#         if len(self.samples) == 0:
#             raise RuntimeError("No images found in SFace2 dataset.")

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, index):
#         img_path, label = self.samples[index]
#         img = Image.open(img_path).convert("RGB")
#         if self.transform:
#             img = self.transform(img)
#         return img, label


# # ══════════════════════════════════════════════════════════════════════════════
# # Dataloader Builder
# # ══════════════════════════════════════════════════════════════════════════════
# def build_loader(
#     sface2_root: str = "data/sface2",
#     img_size: int = 224,
#     batch_size: int = 128,
#     num_workers: int = 4,
#     max_identities: int = None,
#     mode: str = "train",   # "train" or "eval"
# ):
#     transform = get_train_transforms(img_size) if mode == "train" else get_eval_transforms(img_size)

#     dataset = SFace2Dataset(
#         root=sface2_root,
#         transform=transform,
#         max_identities=max_identities
#     )

#     loader = DataLoader(
#         dataset,
#         batch_size=batch_size,
#         shuffle=(mode == "train"),
#         num_workers=num_workers,
#         pin_memory=True,
#         drop_last=(mode == "train"),
#         persistent_workers=(num_workers > 0),
#     )

#     num_classes = len({lbl for _, lbl in dataset.samples})

#     print(f"\n{'═'*60}")
#     print(f" SFace2 DataLoader Ready")
#     print(f" Mode          : {mode}")
#     print(f" Total IDs     : {num_classes}")
#     print(f" Total images  : {len(dataset)}")
#     print(f" Batch size    : {batch_size}")
#     print(f"{'═'*60}\n")

#     return loader, num_classes


# # ══════════════════════════════════════════════════════════════════════════════
# # Quick Test
# # ══════════════════════════════════════════════════════════════════════════════
# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--root", type=str, default="data/sface2")
#     parser.add_argument("--batch_size", type=int, default=8)
#     parser.add_argument("--max_identities", type=int, default=None)
#     args = parser.parse_args()

#     loader, num_classes = build_loader(
#         sface2_root=args.root,
#         batch_size=args.batch_size,
#         max_identities=args.max_identities,
#         mode="train",
#         num_workers=0   # 0 for debugging on Windows
#     )

#     imgs, labels = next(iter(loader))
#     print(f"Batch shape     : {imgs.shape}")
#     print(f"Labels range    : {labels.min().item()} – {labels.max().item()}")
#     print(f"Num classes     : {num_classes}")
#     print("✅ SFace2 loader test PASSED")


























































# # # """
# # # dataset.py
# # # Unified dataset loader for:
# # #   - CASIA-WebFace  → MXNet RecordIO format  (data/casia-webface/)
# # #   - DigiFace-1M    → Standard image folder  (data/digiface/<id>/*.png)

# # # No MXNet installation required — pure Python RecordIO reader.
# # # Labels are globally remapped to avoid collisions when combining datasets.
# # # """

# # # import os
# # # import io
# # # import struct
# # # import numpy as np
# # # from PIL import Image
# # # from torch.utils.data import Dataset, DataLoader, ConcatDataset
# # # from torchvision import transforms


# # # # ══════════════════════════════════════════════════════════════════════════════
# # # # Transforms  (CLIP ViT-L/14 normalisation stats)
# # # # ══════════════════════════════════════════════════════════════════════════════

# # # CLIP_MEAN = [0.48145466, 0.4578275,  0.40821073]
# # # CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]


# # # def get_train_transforms(img_size: int = 224):
# # #     return transforms.Compose([
# # #         transforms.Resize((img_size, img_size)),
# # #         transforms.RandomHorizontalFlip(),
# # #         transforms.ColorJitter(brightness=0.2, contrast=0.2,
# # #                                saturation=0.2,  hue=0.05),
# # #         transforms.RandomGrayscale(p=0.05),
# # #         transforms.ToTensor(),
# # #         transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
# # #     ])


# # # def get_eval_transforms(img_size: int = 224):
# # #     return transforms.Compose([
# # #         transforms.Resize((img_size, img_size)),
# # #         transforms.ToTensor(),
# # #         transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
# # #     ])


# # # # ══════════════════════════════════════════════════════════════════════════════
# # # # Pure-Python MXNet RecordIO helpers
# # # # ══════════════════════════════════════════════════════════════════════════════

# # # _MX_MAGIC = 0xced7230a   # magic number in every RecordIO record header


# # # def _read_idx(idx_path: str):
# # #     """
# # #     Parse MXNet .idx binary file.
# # #     Each entry: uint64 record_index  +  uint64 byte_offset  (little-endian).
# # #     Returns list of byte offsets (one per record, in order).
# # #     """
# # #     offsets = []
# # #     with open(idx_path, "rb") as f:
# # #         while True:
# # #             buf = f.read(16)
# # #             if len(buf) < 16:
# # #                 break
# # #             _rec_idx, byte_off = struct.unpack("<QQ", buf)
# # #             offsets.append(byte_off)
# # #     return offsets


# # # def _read_body_at(fp, offset: int):
# # #     """
# # #     Read one RecordIO record body (image bytes + label) at `offset`.
# # #     RecordIO record layout:
# # #         [uint32 length][uint32 magic][body: length-8 bytes]
# # #     Body layout (ImageRecordIter):
# # #         [uint32 flag][float32 × n_labels][JPEG/PNG bytes]
# # #         flag == 0  →  1 label
# # #         flag == n  →  n labels  (n > 0)
# # #     """
# # #     fp.seek(offset)
# # #     hdr = fp.read(8)
# # #     if len(hdr) < 8:
# # #         return None, None
# # #     length, magic = struct.unpack("<II", hdr)
# # #     if magic != _MX_MAGIC:
# # #         return None, None
# # #     body = fp.read(length - 8)
# # #     if len(body) < 8:
# # #         return None, None

# # #     flag = struct.unpack_from("<I", body, 0)[0]
# # #     n_labels = flag if flag > 0 else 1
# # #     label = int(struct.unpack_from("<f", body, 4)[0])   # first label = class id
# # #     img_bytes = body[4 + 4 * n_labels:]
# # #     return label, img_bytes


# # # # ══════════════════════════════════════════════════════════════════════════════
# # # # Dataset 1 — CASIA-WebFace  (RecordIO)
# # # # ══════════════════════════════════════════════════════════════════════════════

# # # class CASIAWebFaceDataset(Dataset):
# # #     """
# # #     Reads CASIA-WebFace from MXNet RecordIO files.

# # #     Folder layout expected:
# # #         data/casia-webface/
# # #             property      ← "num_classes,h,w"
# # #             train.idx     ← byte offsets
# # #             train.lst     ← (unused at runtime, kept for reference)
# # #             train.rec     ← packed JPEG images

# # #     Args:
# # #         root            : path to data/casia-webface/
# # #         transform       : torchvision transform
# # #         max_identities  : keep only first N identities (None = all 10 575)
# # #         label_offset    : added to every label for multi-dataset merging
# # #     """

# # #     def __init__(self, root: str, transform=None,
# # #                  max_identities: int = None, label_offset: int = 0):
# # #         self.rec_path     = os.path.join(root, "train.rec")
# # #         self.idx_path     = os.path.join(root, "train.idx")
# # #         self.transform    = transform
# # #         self.label_offset = label_offset

# # #         # ── Read property file ──────────────────────────────────────────────
# # #         prop_path = os.path.join(root, "property")
# # #         self.num_classes = 10575   # CASIA-WebFace default
# # #         if os.path.isfile(prop_path):
# # #             with open(prop_path, "r") as f:
# # #                 parts = f.read().strip().split(",")
# # #                 try:
# # #                     self.num_classes = int(parts[0])
# # #                 except ValueError:
# # #                     pass

# # #         # ── Read byte offsets from .idx ─────────────────────────────────────
# # #         print(f"[CASIA] Reading index file …")
# # #         all_offsets = _read_idx(self.idx_path)
# # #         print(f"[CASIA] Total records in .idx : {len(all_offsets)}")

# # #         # ── Scan records → build (offset, label) sample list ────────────────
# # #         # We do a single sequential pass to collect labels without decoding images.
# # #         print(f"[CASIA] Scanning records to map labels (one-time) …")
# # #         self.samples = []   # list of (byte_offset, global_label)
# # #         seen_ids     = {}   # orig_label → remapped contiguous id

# # #         with open(self.rec_path, "rb") as fp:
# # #             for byte_off in all_offsets:
# # #                 label, _ = _read_body_at(fp, byte_off)
# # #                 if label is None:
# # #                     continue
# # #                 if max_identities is not None:
# # #                     # assign contiguous ids in order of first appearance
# # #                     if label not in seen_ids:
# # #                         if len(seen_ids) >= max_identities:
# # #                             continue          # skip identities beyond cap
# # #                         seen_ids[label] = len(seen_ids)
# # #                     mapped = seen_ids[label]
# # #                 else:
# # #                     mapped = label
# # #                 self.samples.append((byte_off, mapped + label_offset))

# # #         # Keep file handle open for __getitem__
# # #         self._fp = open(self.rec_path, "rb")

# # #         n_ids = (len(seen_ids) if max_identities is not None
# # #                  else self.num_classes)
# # #         print(f"[CASIA] Identities used : {n_ids}")
# # #         print(f"[CASIA] Samples         : {len(self.samples)}")

# # #     def __len__(self):
# # #         return len(self.samples)

# # #     def __getitem__(self, index):
# # #         byte_off, label = self.samples[index]
# # #         _, img_bytes = _read_body_at(self._fp, byte_off)
# # #         img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
# # #         if self.transform:
# # #             img = self.transform(img)
# # #         return img, label

# # #     def __del__(self):
# # #         if hasattr(self, "_fp") and self._fp:
# # #             self._fp.close()


# # # # ══════════════════════════════════════════════════════════════════════════════
# # # # Dataset 2 — DigiFace-1M  (image folder)
# # # # ══════════════════════════════════════════════════════════════════════════════

# # # class DigiFaceDataset(Dataset):
# # #     """
# # #     Reads DigiFace-1M from a standard image-folder layout.

# # #     Folder layout expected:
# # #         data/digiface/
# # #             000001/   img_001.png  img_002.png …
# # #             000002/   …
# # #             …

# # #     Args:
# # #         root            : path to data/digiface/
# # #         transform       : torchvision transform
# # #         max_identities  : keep only first N identity folders (None = all)
# # #         label_offset    : added to every label for multi-dataset merging
# # #     """

# # #     VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# # #     def __init__(self, root: str, transform=None,
# # #                  max_identities: int = None, label_offset: int = 0):
# # #         self.transform    = transform
# # #         self.label_offset = label_offset
# # #         self.samples      = []   # list of (path, global_label)

# # #         identities = sorted([
# # #             d for d in os.listdir(root)
# # #             if os.path.isdir(os.path.join(root, d))
# # #         ])

# # #         if max_identities is not None:
# # #             identities = identities[:max_identities]

# # #         for idx, identity in enumerate(identities):
# # #             id_dir = os.path.join(root, identity)
# # #             for fname in os.listdir(id_dir):
# # #                 if os.path.splitext(fname)[1].lower() in self.VALID_EXTS:
# # #                     self.samples.append(
# # #                         (os.path.join(id_dir, fname), idx + label_offset)
# # #                     )

# # #         print(f"[DigiFace] Identities used : {len(identities)}")
# # #         print(f"[DigiFace] Samples         : {len(self.samples)}")

# # #     def __len__(self):
# # #         return len(self.samples)

# # #     def __getitem__(self, index):
# # #         path, label = self.samples[index]
# # #         img = Image.open(path).convert("RGB")
# # #         if self.transform:
# # #             img = self.transform(img)
# # #         return img, label


# # # # ══════════════════════════════════════════════════════════════════════════════
# # # # Combined DataLoader factory
# # # # ══════════════════════════════════════════════════════════════════════════════

# # # def build_dataloader(
# # #     casia_root:              str  = None,
# # #     digiface_root:           str  = None,
# # #     img_size:                int  = 224,
# # #     batch_size:              int  = 128,
# # #     num_workers:             int  = 8,
# # #     max_identities_casia:    int  = None,
# # #     max_identities_digiface: int  = None,
# # #     mode:                    str  = "train",   # "train" | "eval"
# # # ):
# # #     """
# # #     Build a combined DataLoader from any combination of CASIA and DigiFace.
# # #     Labels are globally remapped so there are zero class-index collisions.

# # #     Returns:
# # #         loader      : torch DataLoader
# # #         num_classes : total unique identities across all datasets
# # #     """
# # #     transform = (get_train_transforms(img_size) if mode == "train"
# # #                  else get_eval_transforms(img_size))

# # #     datasets = []
# # #     offset   = 0   # running label offset

# # #     # ── CASIA-WebFace ────────────────────────────────────────────────────────
# # #     if casia_root and os.path.isdir(casia_root):
# # #         print("\n── Loading CASIA-WebFace ──────────────────────────────────")
# # #         casia_ds = CASIAWebFaceDataset(
# # #             root            = casia_root,
# # #             transform       = transform,
# # #             max_identities  = max_identities_casia,
# # #             label_offset    = offset,
# # #         )
# # #         # Count unique labels actually used
# # #         used_labels = len({lbl for _, lbl in casia_ds.samples})
# # #         offset += used_labels
# # #         datasets.append(casia_ds)
# # #     else:
# # #         if casia_root:
# # #             print(f"[Warning] CASIA root not found: {casia_root}")

# # #     # ── DigiFace-1M ──────────────────────────────────────────────────────────
# # #     if digiface_root and os.path.isdir(digiface_root):
# # #         print("\n── Loading DigiFace-1M ────────────────────────────────────")
# # #         digi_ds = DigiFaceDataset(
# # #             root            = digiface_root,
# # #             transform       = transform,
# # #             max_identities  = max_identities_digiface,
# # #             label_offset    = offset,
# # #         )
# # #         used_labels = len({lbl for _, lbl in digi_ds.samples})
# # #         offset += used_labels
# # #         datasets.append(digi_ds)
# # #     else:
# # #         if digiface_root:
# # #             print(f"[Warning] DigiFace root not found: {digiface_root}")

# # #     if not datasets:
# # #         raise ValueError(
# # #             "No valid dataset found. Provide at least one of "
# # #             "casia_root or digiface_root."
# # #         )

# # #     combined    = ConcatDataset(datasets)
# # #     num_classes = offset

# # #     loader = DataLoader(
# # #         combined,
# # #         batch_size  = batch_size,
# # #         shuffle     = (mode == "train"),
# # #         num_workers = num_workers,
# # #         pin_memory  = True,
# # #         drop_last   = (mode == "train"),
# # #         persistent_workers = (num_workers > 0),
# # #     )

# # #     print(f"\n{'═'*55}")
# # #     print(f"  DataLoader ready")
# # #     print(f"  Mode        : {mode}")
# # #     print(f"  Datasets    : {len(datasets)}")
# # #     print(f"  Total IDs   : {num_classes}")
# # #     print(f"  Total imgs  : {len(combined)}")
# # #     print(f"  Batch size  : {batch_size}")
# # #     print(f"{'═'*55}\n")

# # #     return loader, num_classes


# # # # ══════════════════════════════════════════════════════════════════════════════
# # # # Quick smoke-test
# # # # ══════════════════════════════════════════════════════════════════════════════

# # # if __name__ == "__main__":
# # #     import argparse, torch

# # #     parser = argparse.ArgumentParser()
# # #     parser.add_argument("--casia_root",    type=str, default=None)
# # #     parser.add_argument("--digiface_root", type=str, default=None)
# # #     parser.add_argument("--batch_size",    type=int, default=8)
# # #     args = parser.parse_args()

# # #     loader, num_classes = build_dataloader(
# # #         casia_root    = args.casia_root,
# # #         digiface_root = args.digiface_root,
# # #         batch_size    = args.batch_size,
# # #         num_workers   = 0,    # 0 for smoke-test (avoids multiprocess issues)
# # #         mode          = "train",
# # #     )

# # #     imgs, labels = next(iter(loader))
# # #     print(f"Batch shape  : {imgs.shape}")
# # #     print(f"Label range  : {labels.min().item()} – {labels.max().item()}")
# # #     print(f"Num classes  : {num_classes}")
# # #     print("Smoke-test PASSED ✓")
# # import os
# # import io
# # import struct
# # from PIL import Image
# # from torch.utils.data import Dataset, DataLoader
# # from torchvision import transforms

# # # ─────────────────────────────────────────────────────────────
# # # Constants
# # # ─────────────────────────────────────────────────────────────

# # _MX_MAGIC = 0xced7230a

# # CLIP_MEAN = [0.48145466, 0.4578275,  0.40821073]
# # CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]


# # # ─────────────────────────────────────────────────────────────
# # # Transforms
# # # ─────────────────────────────────────────────────────────────

# # def get_transforms(train=True, size=224):
# #     if train:
# #         return transforms.Compose([
# #             transforms.Resize((size, size)),
# #             transforms.RandomHorizontalFlip(),
# #             transforms.ToTensor(),
# #             transforms.Normalize(CLIP_MEAN, CLIP_STD),
# #         ])
# #     else:
# #         return transforms.Compose([
# #             transforms.Resize((size, size)),
# #             transforms.ToTensor(),
# #             transforms.Normalize(CLIP_MEAN, CLIP_STD),
# #         ])


# # # ─────────────────────────────────────────────────────────────
# # # IDX Reader
# # # ─────────────────────────────────────────────────────────────

# # def read_idx(idx_path):
# #     offsets = []
# #     with open(idx_path, "rb") as f:
# #         while True:
# #             buf = f.read(16)
# #             if len(buf) < 16:
# #                 break
# #             _, offset = struct.unpack("<QQ", buf)
# #             offsets.append(offset)
# #     return offsets


# # # ─────────────────────────────────────────────────────────────
# # # RecordIO Reader (ROBUST)
# # # ─────────────────────────────────────────────────────────────

# # # def read_record(fp, offset):
# # #     fp.seek(offset)

# # #     header = fp.read(8)
# # #     if len(header) < 8:
# # #         return None, None

# # #     length, magic = struct.unpack("<II", header)

# # #     if magic != _MX_MAGIC:
# # #         return None, None

# # #     body = fp.read(length - 8)
# # #     if len(body) < 16:
# # #         return None, None

# # #     try:
# # #         flag = struct.unpack_from("<I", body, 0)[0]
# # #         n_labels = flag if flag > 0 else 1

# # #         label = int(struct.unpack_from("<f", body, 4)[0])

# # #         # Try multiple possible image offsets (robust fallback)
# # #         possible_offsets = [
# # #             4 + 4 * n_labels,           # minimal
# # #             4 + 4 * n_labels + 4,       # + id
# # #             4 + 4 * n_labels + 8,       # + id + id2 (most common)
# # #         ]

# # #         for start in possible_offsets:
# # #             img_bytes = body[start:]
# # #             if img_bytes.startswith(b'\xff\xd8') or img_bytes.startswith(b'\x89PNG'):
# # #                 return label, img_bytes

# # #         return None, None

# # #     except Exception:
# # #         return None, None
# # def read_record(fp, offset):
# #     try:
# #         fp.seek(offset)
# #         header = fp.read(8)
# #         if len(header) < 8:
# #             return None, None

# #         rec_len, magic = struct.unpack("<II", header)
# #         if magic != _MX_MAGIC:
# #             if offset < 200:
# #                 print(f"[ERROR] Bad magic at offset {offset}: {hex(magic)} (expected {hex(_MX_MAGIC)})")
# #             return None, None

# #         body = fp.read(rec_len - 8)
# #         if len(body) < 20:
# #             if offset < 200:
# #                 print(f"[ERROR] Body too short at offset {offset}: only {len(body)} bytes")
# #             return None, None

# #         # Parse header
# #         flag, label_f, id1, id2 = struct.unpack_from("<Ifff", body, 0)
# #         label = int(label_f)

# #         # === Force diagnostic on first record ===
# #         if offset < 100 and not hasattr(read_record, "has_printed"):
# #             read_record.has_printed = True
# #             print(f"\n=== DETAILED DIAGNOSTIC - FIRST RECORD (offset={offset}) ===")
# #             print(f"Record length      : {rec_len}")
# #             print(f"IRHeader           : flag={flag}, label={label}, id1={id1}, id2={id2}")
# #             print(f"Body size          : {len(body)} bytes")
# #             print(f"First 100 bytes hex: {body[:100].hex()}")
# #             print(f"First 100 bytes raw: {repr(body[:100])}")

# #             jpeg_pos = body.find(b'\xff\xd8')
# #             if jpeg_pos != -1:
# #                 print(f"✅ JPEG signature found at byte {jpeg_pos}")
# #                 return label, body[jpeg_pos:]
# #             else:
# #                 print("❌ JPEG signature (FF D8) NOT found anywhere in body!")

# #             print("=" * 80)

# #         # Try common image start positions
# #         for start in [16, 20, 24, 28, 32, 36, 40, 44, 48]:
# #             if start + 2 <= len(body) and body[start:start+2] == b'\xff\xd8':
# #                 return label, body[start:]

# #         # Full search fallback
# #         pos = body.find(b'\xff\xd8')
# #         if pos != -1 and pos >= 16:
# #             return label, body[pos:]

# #         return None, None

# #     except Exception as e:
# #         # ← This is what you asked for: print the real exception
# #         if offset < 200:
# #             print(f"[EXCEPTION] at offset {offset}: {type(e).__name__}: {e}")
# #         return None, None
    

# # # ─────────────────────────────────────────────────────────────
# # # CASIA Dataset
# # # ─────────────────────────────────────────────────────────────

# # # class CASIAWebFaceDataset(Dataset):
# # #     def __init__(self, root, transform=None, max_samples=None):
# # #         self.rec_path = os.path.join(root, "train.rec")
# # #         self.idx_path = os.path.join(root, "train.idx")
# # #         self.transform = transform

# # #         print("\n[INFO] Loading CASIA-WebFace...")
# # #         print(self.rec_path)

# # #         self.offsets = read_idx(self.idx_path)
# # #         print(f"[INFO] Total records: {len(self.offsets)}")

# # #         self.samples = []

# # #         with open(self.rec_path, "rb") as fp:
# # #             valid = 0
# # #             checked = 0

# # #             for offset in self.offsets:
# # #                 label, _ = read_record(fp, offset)


# # #                 checked += 1

# # #                 if label is not None:
# # #                     self.samples.append((offset, label))
# # #                     valid += 1

# # #                 # limit for debug
# # #                 if max_samples and valid >= max_samples:
# # #                     break

# # #                 # progress log
# # #                 if checked % 50000 == 0:
# # #                     print(f"[SCAN] checked={checked} valid={valid}")

# # #         print(f"[INFO] Final valid samples: {len(self.samples)}")

# # #         if len(self.samples) == 0:
# # #             raise RuntimeError("❌ No valid samples found. RecordIO parsing failed.")

# # #         self.fp = open(self.rec_path, "rb")

# # class CASIAWebFaceDataset(Dataset):
# #     def __init__(self, root, transform=None, max_samples=None):
# #         self.rec_path = os.path.join(root, "train.rec")
# #         self.idx_path = os.path.join(root, "train.idx")
# #         self.transform = transform

# #         print("\n[INFO] Loading CASIA-WebFace...")
# #         print(f"Record file: {self.rec_path}")

# #         # ── ADD PROPERTY READING HERE ─────────────────────────────────────
# #         prop_path = os.path.join(root, "property")
# #         self.num_classes = 10575  # default fallback
# #         if os.path.isfile(prop_path):
# #             with open(prop_path) as f:
# #                 line = f.read().strip()
# #                 try:
# #                     num_classes, h, w = map(int, line.split(','))
# #                     self.num_classes = num_classes
# #                     print(f"[INFO] CASIA property: {num_classes} classes, {h}x{w}")
# #                 except Exception as e:
# #                     print(f"[WARNING] Could not parse property file: {e}")
# #         else:
# #             print(f"[WARNING] property file not found at {prop_path}. Using default {self.num_classes} classes.")

# #         # ── Continue with index loading ───────────────────────────────────
# #         self.offsets = read_idx(self.idx_path)
# #         print(f"[INFO] Total records in .idx: {len(self.offsets)}")

# #         self.samples = []
# #         with open(self.rec_path, "rb") as fp:
# #             valid = 0
# #             checked = 0
# #             for offset in self.offsets:
# #                 label, _ = read_record(fp, offset)
               
# #                 checked += 1
# #                 if label is not None:
# #                     self.samples.append((offset, label))
# #                     valid += 1

# #                 # limit for debug
# #                 if max_samples and valid >= max_samples:
# #                     break

# #                 # progress log
# #                 if checked % 50000 == 0:
# #                     print(f"[SCAN] checked={checked:,} valid={valid:,}")

# #         print(f"[INFO] Final valid samples: {len(self.samples):,}")

# #         if len(self.samples) == 0:
# #             raise RuntimeError("❌ No valid samples found. RecordIO parsing failed.")

# #         self.fp = open(self.rec_path, "rb")



# #     def __len__(self):
# #         return len(self.samples)

# #     def __getitem__(self, idx):
# #         offset, label = self.samples[idx]
# #         _, img_bytes = read_record(self.fp, offset)

# #         img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

# #         if self.transform:
# #             img = self.transform(img)

# #         return img, label

# #     def __del__(self):
# #         if hasattr(self, "fp"):
# #             self.fp.close()


# # # ─────────────────────────────────────────────────────────────
# # # Dataloader Builder
# # # ─────────────────────────────────────────────────────────────

# # def build_loader(root, batch_size=32, train=True):
# #     dataset = CASIAWebFaceDataset(
# #         root=root,
# #         transform=get_transforms(train)
# #     )

# #     loader = DataLoader(
# #         dataset,
# #         batch_size=batch_size,
# #         shuffle=train,
# #         num_workers=0  # keep 0 for debugging
# #     )

# #     return loader


# # # ─────────────────────────────────────────────────────────────
# # # Test
# # # ─────────────────────────────────────────────────────────────

# # if __name__ == "__main__":
# #     import argparse

# #     parser = argparse.ArgumentParser()
# #     parser.add_argument("--casia_root", type=str, required=True)
# #     args = parser.parse_args()

# #     loader = build_loader(args.casia_root, batch_size=8)

# #     print("\n[TEST] Fetching batch...")
# #     imgs, labels = next(iter(loader))

# #     print("Batch shape:", imgs.shape)
# #     print("Labels:", labels[:10])
# #     print("✅ SUCCESS")