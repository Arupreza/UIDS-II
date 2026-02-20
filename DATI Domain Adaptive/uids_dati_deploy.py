# dati_ids_kia_train.py
# Replicates DATI-IDS pipeline (Tan et al., IEEE T-ITS 2025) for Kia CSVs:
# CAN-ID only -> windows of 128 IDs -> GASF 128x128 -> RGB stack -> 4-layer CNN
# + MK-MMD domain adaptation (train=source labeled, test=target unlabeled) -> joint loss.

import os
import glob
import math
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ---- GASF (pyts) ----
# pip install pyts
from pyts.image import GramianAngularField


# -----------------------------
# Config
# -----------------------------
@dataclass
class CFG:
    train_dir: str
    test_dir: str
    out_dir: str = "./dati_kia_runs"

    window: int = 128
    stride: int = 128        # start with non-overlap; change to 1 for sliding (much larger dataset)
    batch_size: int = 128
    epochs: int = 100
    lr: float = 3e-3
    lambda_mmd: float = 0.1
    num_workers: int = 4

    # cache
    use_cache: bool = True
    cache_name: str = "kia_gasf_cache"

    # training
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

    # MK-MMD kernel bandwidths (multi-kernel)
    sigmas: Tuple[float, ...] = (1, 2, 4, 8, 16)


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# Utilities: CSV -> CAN_ID stream
# -----------------------------
def list_csvs(folder: str) -> List[str]:
    return sorted(glob.glob(os.path.join(folder, "*.csv")))

def normalize_can_id(x) -> str:
    # Keep as string; preserve leading zeros
    s = str(x).strip()
    # common: "340" vs "0340"
    if all(c in "0123456789abcdefABCDEF" for c in s) and len(s) <= 8:
        s = s.upper().zfill(4)
    return s

def is_attack_label(lbl) -> int:
    # Customize if your label schema differs
    s = str(lbl).strip().lower()
    return 0 if s == "normal" else 1

def iter_rows(csv_path: str, chunksize: int = 200_000) -> Iterable[pd.DataFrame]:
    # Use chunked read for large logs
    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        yield chunk

def collect_can_id_vocab(csv_paths: List[str]) -> Dict[str, int]:
    """Build mapping CAN_ID -> integer [0..Q-1] over union of train+test (labels not used)."""
    vocab_set = set()
    for fp in csv_paths:
        for chunk in iter_rows(fp):
            if "CAN_ID" not in chunk.columns:
                raise ValueError(f"Missing CAN_ID column in {fp}")
            ids = chunk["CAN_ID"].map(normalize_can_id).unique().tolist()
            vocab_set.update(ids)
    vocab = sorted(vocab_set)
    return {cid: i for i, cid in enumerate(vocab)}

def count_windows_in_file(csv_path: str, window: int, stride: int) -> int:
    # Count rows quickly (streaming) then compute number of windows
    n = 0
    for chunk in iter_rows(csv_path):
        n += len(chunk)
    if n < window:
        return 0
    return 1 + (n - window) // stride

def count_total_windows(csv_paths: List[str], window: int, stride: int) -> int:
    return sum(count_windows_in_file(fp, window, stride) for fp in csv_paths)


# -----------------------------
# Windowing + GASF
# -----------------------------
def minmax_scale_01(x: np.ndarray) -> np.ndarray:
    # Eq.(1) in paper: scale within [0,1] per window
    x = x.astype(np.float32)
    mn = float(x.min())
    mx = float(x.max())
    if mx - mn < 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)

class GASFMaker:
    def __init__(self, image_size: int):
        self.gaf = GramianAngularField(image_size=image_size, method="summation")

    def to_rgb(self, seq_scaled_01: np.ndarray) -> np.ndarray:
        # seq_scaled_01: (window,)
        img = self.gaf.fit_transform(seq_scaled_01[None, :])[0].astype(np.float32)  # (H,W)
        rgb = np.stack([img, img, img], axis=0)  # (3,H,W) per paper (3 channels)
        return rgb


def build_cache(
    csv_paths: List[str],
    mapping: Dict[str, int],
    cache_dir: str,
    split_name: str,
    window: int,
    stride: int,
    chunksize: int = 200_000,
):
    os.makedirs(cache_dir, exist_ok=True)
    X_path = os.path.join(cache_dir, f"{split_name}_X.npy")
    y_path = os.path.join(cache_dir, f"{split_name}_y.npy")
    meta_path = os.path.join(cache_dir, f"{split_name}_meta.json")

    total = count_total_windows(csv_paths, window, stride)
    if total == 0:
        raise ValueError(f"No windows produced for {split_name}. Check window/stride or CSV length.")

    # memmap for big data
    X_mm = np.memmap(X_path, dtype="float16", mode="w+", shape=(total, 3, window, window))
    y_mm = np.memmap(y_path, dtype="int8", mode="w+", shape=(total,))

    maker = GASFMaker(image_size=window)

    # ESTABLISH GLOBAL SCALING MAXIMUM (Q)
    Q = len(mapping)
    max_val = float(Q - 1) if Q > 1 else 1.0

    w_ids = []
    w_lbl = []
    write_idx = 0

    for fp in csv_paths:
        buf_ids: List[int] = []
        buf_attack: List[int] = []
        
        for chunk in iter_rows(fp, chunksize=chunksize):
            if "CAN_ID" not in chunk.columns or "Label" not in chunk.columns:
                raise ValueError(f"{fp} must contain CAN_ID and Label columns.")

            ids = chunk["CAN_ID"].map(normalize_can_id).map(lambda s: mapping.get(s, None)).to_numpy()
            if np.any(pd.isna(ids)):
                raise ValueError(f"Found CAN_ID not in mapping in {fp}. This should not happen if mapping built on union.")
            labels = chunk["Label"].map(is_attack_label).to_numpy(dtype=np.int8)

            for cid_i, lab_i in zip(ids.tolist(), labels.tolist()):
                buf_ids.append(int(cid_i))
                buf_attack.append(int(lab_i))

                if len(buf_ids) == window:
                    seq = np.asarray(buf_ids, dtype=np.float32)
                    
                    # APPLY GLOBAL SCALING
                    seq01 = seq / max_val
                    
                    rgb = maker.to_rgb(seq01)  # (3,128,128)
                    X_mm[write_idx] = rgb.astype(np.float16)

                    # Window label: if any attack in 128 lines -> Attack (paper rule)
                    y_mm[write_idx] = 1 if any(buf_attack) else 0
                    write_idx += 1

                    # advance
                    if stride >= window:
                        buf_ids = []
                        buf_attack = []
                    else:
                        buf_ids = buf_ids[stride:]
                        buf_attack = buf_attack[stride:]

    X_mm.flush()
    y_mm.flush()

    meta = {
        "split": split_name,
        "total_windows": int(total),
        "window": int(window),
        "stride": int(stride),
        "X_path": X_path,
        "y_path": y_path,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[CACHE] Built {split_name}: {total} windows -> {cache_dir}")


class CachedGASFDataset(Dataset):
    def __init__(self, cache_dir: str, split_name: str, window: int):
        meta_path = os.path.join(cache_dir, f"{split_name}_meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Missing cache meta: {meta_path}")
        with open(meta_path, "r") as f:
            meta = json.load(f)
        self.n = int(meta["total_windows"])
        self.window = window
        self.X = np.memmap(meta["X_path"], dtype="float16", mode="r", shape=(self.n, 3, window, window))
        self.y = np.memmap(meta["y_path"], dtype="int8", mode="r", shape=(self.n,))

    def __len__(self):
        return self.n

    def __getitem__(self, idx: int):
        x = torch.from_numpy(np.array(self.X[idx], dtype=np.float32))  # float32 for training
        y = torch.tensor(int(self.y[idx]), dtype=torch.long)
        return x, y


# -----------------------------
# DATI-IDS Model (CNN + feature taps)
# -----------------------------
class DATI_CNN(nn.Module):
    """
    4 conv blocks: (Conv3x3 + ReLU + MaxPool2x2) with filters 64,32,16,8
    Then FC: 512->256->128->2 (binary logits)
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)  # same
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 128 -> 64 -> 32 -> 16 -> 8, channels=8 => 8*8*8=512
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.cls = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(x.size(0), -1)             # (B,512)

        f256 = F.relu(self.fc1(x))            # feature for MMD
        f128 = F.relu(self.fc2(f256))         # feature for MMD
        logits = self.cls(f128)
        return logits, f256, f128


# -----------------------------
# MK-MMD (multi-kernel RBF)
# -----------------------------
def pairwise_sq_dists(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # ||x-y||^2
    x2 = (x ** 2).sum(dim=1, keepdim=True)   # (n,1)
    y2 = (y ** 2).sum(dim=1, keepdim=True).t()  # (1,m)
    return x2 + y2 - 2.0 * (x @ y.t())

def rbf_kernel(x: torch.Tensor, y: torch.Tensor, sigma: float) -> torch.Tensor:
    d2 = pairwise_sq_dists(x, y).clamp_min(0.0)
    gamma = 1.0 / (2.0 * (sigma ** 2))
    return torch.exp(-gamma * d2)

def mkmmd(x: torch.Tensor, y: torch.Tensor, sigmas: Tuple[float, ...]) -> torch.Tensor:
    # MMD^2 = E[Kxx] + E[Kyy] - 2E[Kxy], averaged across kernels
    Kxx = 0.0
    Kyy = 0.0
    Kxy = 0.0
    for s in sigmas:
        Kxx = Kxx + rbf_kernel(x, x, s)
        Kyy = Kyy + rbf_kernel(y, y, s)
        Kxy = Kxy + rbf_kernel(x, y, s)

    Kxx = Kxx / len(sigmas)
    Kyy = Kyy / len(sigmas)
    Kxy = Kxy / len(sigmas)

    # exclude diagonal terms for stability (optional)
    n = x.size(0)
    m = y.size(0)
    if n > 1:
        Kxx = (Kxx.sum() - Kxx.diag().sum()) / (n * (n - 1))
    else:
        Kxx = Kxx.mean()
    if m > 1:
        Kyy = (Kyy.sum() - Kyy.diag().sum()) / (m * (m - 1))
    else:
        Kyy = Kyy.mean()

    Kxy = Kxy.mean()
    return Kxx + Kyy - 2.0 * Kxy


# -----------------------------
# Train / Eval
# -----------------------------
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> Tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    ys, ps = [], []
    total_loss = 0.0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits, _, _ = model(x)
        loss = F.cross_entropy(logits, y)
        total_loss += float(loss.item()) * x.size(0)
        total += x.size(0)
        pred = torch.argmax(logits, dim=1)
        ys.append(y.cpu().numpy())
        ps.append(pred.cpu().numpy())
    ys = np.concatenate(ys)
    ps = np.concatenate(ps)
    return total_loss / max(total, 1), ys, ps

def train(cfg: CFG):
    set_seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)

    train_csvs = list_csvs(cfg.train_dir)
    test_csvs  = list_csvs(cfg.test_dir)
    if len(train_csvs) == 0 or len(test_csvs) == 0:
        raise ValueError("train_dir and test_dir must each contain at least one CSV.")

    # ---- Build CAN_ID mapping on union (labels not used) ----
    mapping = collect_can_id_vocab(train_csvs + test_csvs)
    cache_dir = os.path.join(cfg.out_dir, cfg.cache_name)
    os.makedirs(cache_dir, exist_ok=True)
    with open(os.path.join(cache_dir, "can_id_mapping.json"), "w") as f:
        json.dump(mapping, f)

    # ---- Cache GASF for speed ----
    if cfg.use_cache:
        if not os.path.exists(os.path.join(cache_dir, "train_meta.json")):
            build_cache(train_csvs, mapping, cache_dir, "train", cfg.window, cfg.stride)
        if not os.path.exists(os.path.join(cache_dir, "test_meta.json")):
            build_cache(test_csvs, mapping, cache_dir, "test", cfg.window, cfg.stride)

        ds_src = CachedGASFDataset(cache_dir, "train", cfg.window)  # source labeled
        ds_tgt = CachedGASFDataset(cache_dir, "test",  cfg.window)  # target (use labels only for eval)

    else:
        raise NotImplementedError("For large CAN logs, caching is strongly recommended.")

    # ---- Sanity: label distribution ----
    y_src = np.array(ds_src.y, dtype=np.int64)
    y_tgt = np.array(ds_tgt.y, dtype=np.int64)
    print(f"[SRC] windows={len(ds_src)}  normal={(y_src==0).sum()}  attack={(y_src==1).sum()}")
    print(f"[TGT] windows={len(ds_tgt)}  normal={(y_tgt==0).sum()}  attack={(y_tgt==1).sum()}")

    src_loader = DataLoader(ds_src, batch_size=cfg.batch_size, shuffle=True,
                            num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    tgt_loader = DataLoader(ds_tgt, batch_size=cfg.batch_size, shuffle=True,
                            num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(ds_tgt, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=cfg.num_workers, pin_memory=True)

    model = DATI_CNN().to(cfg.device)

    opt = torch.optim.NAdam(model.parameters(), lr=cfg.lr)
    # Paper uses LambdaLR; here a simple linear decay to 10% of lr by final epoch
    def lr_lambda(epoch):
        return 0.1 + 0.9 * (1.0 - epoch / max(cfg.epochs, 1))
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)

    best_f1 = -1.0
    best_path = os.path.join(cfg.out_dir, "best_dati_cnn.pt")

    tgt_iter = iter(tgt_loader)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0
        seen = 0

        for x_s, y_s in src_loader:
            try:
                x_t, _ = next(tgt_iter)
            except StopIteration:
                tgt_iter = iter(tgt_loader)
                x_t, _ = next(tgt_iter)

            x_s = x_s.to(cfg.device)
            y_s = y_s.to(cfg.device)
            x_t = x_t.to(cfg.device)

            opt.zero_grad(set_to_none=True)

            logits_s, f256_s, f128_s = model(x_s)
            _,        f256_t, f128_t = model(x_t)

            cls_loss = F.cross_entropy(logits_s, y_s)
            # MK-MMD between source/target features (between FC layers as described)
            mmd_loss = mkmmd(f256_s, f256_t, cfg.sigmas) + mkmmd(f128_s, f128_t, cfg.sigmas)

            loss = cls_loss + cfg.lambda_mmd * mmd_loss
            loss.backward()
            opt.step()

            running += float(loss.item()) * x_s.size(0)
            seen += x_s.size(0)

        sch.step()

        # ---- Eval on target (test) ----
        test_loss, y_true, y_pred = evaluate(model, test_loader, cfg.device)
        rep = classification_report(y_true, y_pred, digits=4, output_dict=True, zero_division=0)
        f1 = rep["weighted avg"]["f1-score"]

        print(f"Epoch {epoch:03d}/{cfg.epochs} | train_loss={running/max(seen,1):.4f} | "
              f"test_loss={test_loss:.4f} | test_weighted_f1={f1:.4f} | lr={sch.get_last_lr()[0]:.6f}")

        if f1 > best_f1:
            best_f1 = f1
            torch.save({"model": model.state_dict(), "cfg": cfg.__dict__}, best_path)

    # Final report
    ckpt = torch.load(best_path, map_location=cfg.device)
    model.load_state_dict(ckpt["model"])
    _, y_true, y_pred = evaluate(model, test_loader, cfg.device)

    print("\n=== BEST MODEL REPORT (on target/test) ===")
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))
    print("Confusion Matrix [rows=true 0/1, cols=pred 0/1]:\n", confusion_matrix(y_true, y_pred))

    print(f"\nSaved: {best_path}")