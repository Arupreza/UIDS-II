"""
uids_deepmlp_deploy.py

A Jupyter-friendly deployment + evaluation helper for the Aljabri-style DeepMLP CAN IDS.

What you get:
- Load trained artifacts (PyTorch model + sklearn scalers + LabelEncoder)
- End-to-end inference (raw rows -> preprocess -> predict label/prob)
- Streaming evaluation on large CSV folders (classification report + confusion matrix)
- Profiling: total inference time (end-to-end), params, analytic FLOPs (MFLOPs/TFLOPs), optional THOP FLOPs/MACs

Expected feature columns:
['Time_Offset','CAN_ID','Data_Length','One','Two','Three','Four','Five','Six','Seven','Eight']

Artifacts expected in artifact_dir:
- deepmlp_best.pt
- minmax.joblib
- standard.joblib
- label_encoder.joblib
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix
import joblib


FEATURE_COLS: List[str] = [
    "Time_Offset", "CAN_ID", "Data_Length",
    "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight"
]
LABEL_COL: str = "Label"


class DeepMLP(nn.Module):
    """11 -> 128 -> 64 -> C MLP, with optional Softmax for probability output."""
    def __init__(self, in_dim: int, num_classes: int, with_softmax: bool = True):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        self.with_softmax = with_softmax
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        if self.with_softmax:
            x = self.softmax(x)
        return x


def parse_can_id(x) -> float:
    """Parse CAN_ID robustly: int/float, '0x..', or hex like '18DAF110'."""
    try:
        if pd.isna(x):
            return np.nan
    except Exception:
        pass

    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return int(x)

    s = str(x).strip()
    if s == "":
        return np.nan

    try:
        return int(s)
    except Exception:
        pass

    s2 = s.lower()
    if s2.startswith("0x"):
        try:
            return int(s2, 16)
        except Exception:
            return np.nan

    if any(c in "abcdef" for c in s2):
        try:
            return int(s2, 16)
        except Exception:
            return np.nan

    return np.nan


def ensure_feature_columns(df: pd.DataFrame) -> None:
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")


def preprocess_raw_df(df: pd.DataFrame) -> pd.DataFrame:
    """CAN_ID parsing + numeric coercion + fillna(0.0)."""
    df = df.copy()
    ensure_feature_columns(df)
    df["CAN_ID"] = df["CAN_ID"].apply(parse_can_id)
    for c in FEATURE_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0.0)
    return df


def apply_scalers(X: np.ndarray, mm: MinMaxScaler, ss: StandardScaler) -> np.ndarray:
    """MinMax -> StandardScaler."""
    return ss.transform(mm.transform(X))


def to_tensor(X: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.tensor(X, dtype=torch.float32, device=device)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@dataclass
class DeployBundle:
    model: nn.Module
    mm: MinMaxScaler
    ss: StandardScaler
    le: LabelEncoder
    device: torch.device


def load_deploy_bundle(
    artifact_dir: Union[str, Path],
    *,
    model_filename: str = "deepmlp_best.pt",
    minmax_filename: str = "minmax.joblib",
    standard_filename: str = "standard.joblib",
    labelenc_filename: str = "label_encoder.joblib",
    device: Optional[torch.device] = None,
    with_softmax: bool = True,
) -> DeployBundle:
    """
    Load model + preprocessors from artifact_dir.
    """
    artifact_dir = Path(artifact_dir)

    mm = joblib.load(artifact_dir / minmax_filename)
    ss = joblib.load(artifact_dir / standard_filename)
    le = joblib.load(artifact_dir / labelenc_filename)

    num_classes = len(le.classes_)
    in_dim = len(FEATURE_COLS)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DeepMLP(in_dim=in_dim, num_classes=num_classes, with_softmax=with_softmax)
    state = torch.load(artifact_dir / model_filename, map_location="cpu")
    model.load_state_dict(state)
    model.to(device).eval()

    return DeployBundle(model=model, mm=mm, ss=ss, le=le, device=device)


@torch.no_grad()
def infer(
    bundle: DeployBundle,
    inp: Union[pd.DataFrame, pd.Series, Dict, List[Dict]],
    *,
    return_probs: bool = True
) -> Tuple[List[str], Optional[np.ndarray]]:
    """
    End-to-end inference:
      raw rows -> preprocess -> scale -> model -> predicted labels (+ probs)
    """
    if isinstance(inp, dict):
        df = pd.DataFrame([inp])
    elif isinstance(inp, list) and inp and isinstance(inp[0], dict):
        df = pd.DataFrame(inp)
    elif isinstance(inp, pd.Series):
        df = inp.to_frame().T
    else:
        df = inp.copy()

    df = preprocess_raw_df(df)
    X = df[FEATURE_COLS].values.astype(np.float32)
    X = apply_scalers(X, bundle.mm, bundle.ss)

    Xt = to_tensor(X, bundle.device)
    out = bundle.model(Xt)

    # If model returns logits (with_softmax=False), convert to probs.
    if out.min().item() < 0.0 or out.max().item() > 1.0:
        out = torch.softmax(out, dim=1)

    probs = out.detach().cpu().numpy()
    pred_idx = probs.argmax(axis=1)
    pred_labels = bundle.le.inverse_transform(pred_idx).tolist()

    return pred_labels, (probs if return_probs else None)


def deepmlp_flops_per_sample(in_dim: int, num_classes: int, *, count_softmax: bool = True) -> int:
    """
    Analytic FLOPs estimate per sample. Convention: 1 MAC = 2 FLOPs.
    """
    h1, h2 = 128, 64
    fc1 = 2 * in_dim * h1 + h1
    relu1 = h1
    fc2 = 2 * h1 * h2 + h2
    relu2 = h2
    fc3 = 2 * h2 * num_classes + num_classes
    softmax = (5 * num_classes) if count_softmax else 0
    return int(fc1 + relu1 + fc2 + relu2 + fc3 + softmax)


def try_thop_profile(model: nn.Module, in_dim: int, device: torch.device) -> Optional[Dict[str, float]]:
    """
    Optional THOP profiling. Returns dict or None if thop missing.
    Install in Jupyter: %pip install thop
    """
    try:
        from thop import profile
    except Exception:
        return None

    dummy = torch.randn(1, in_dim, device=device)
    macs, flops = profile(model, inputs=(dummy,), verbose=False)
    return {
        "thop_macs_per_sample": float(macs),
        "thop_flops_per_sample": float(flops),
        "thop_mflops_per_sample": float(flops) / 1e6,
        "thop_tflops_per_sample": float(flops) / 1e12,
    }


def _safe_label_transform(le: LabelEncoder, y_str: np.ndarray) -> np.ndarray:
    y_str = y_str.astype(str)
    unseen = set(np.unique(y_str)) - set(map(str, le.classes_))
    if unseen:
        raise ValueError(f"Unseen labels found in evaluation data: {sorted(unseen)}")
    return le.transform(y_str)


def report_from_confusion_matrix(cm: np.ndarray, class_names: List[str]) -> Tuple[str, Dict[str, float]]:
    """
    Build a sklearn-like classification report from confusion matrix (streaming-safe).
    """
    cm = cm.astype(np.int64)
    support = cm.sum(axis=1)
    pred_sum = cm.sum(axis=0)
    tp = np.diag(cm)

    precision = np.divide(tp, pred_sum, out=np.zeros_like(tp, dtype=float), where=pred_sum != 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp, dtype=float), where=support != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp, dtype=float), where=(precision + recall) != 0)

    accuracy = float(tp.sum() / max(1, cm.sum()))

    macro_p = float(np.mean(precision)) if len(precision) else 0.0
    macro_r = float(np.mean(recall)) if len(recall) else 0.0
    macro_f1 = float(np.mean(f1)) if len(f1) else 0.0

    weights = support / max(1, support.sum())
    weighted_p = float(np.sum(precision * weights))
    weighted_r = float(np.sum(recall * weights))
    weighted_f1 = float(np.sum(f1 * weights))

    lines = []
    lines.append("Classification Report (computed from streaming confusion matrix)\n")
    lines.append(f"{'class':<15} {'precision':>9} {'recall':>9} {'f1-score':>9} {'support':>9}")
    for i, name in enumerate(class_names):
        lines.append(f"{name:<15} {precision[i]:>9.4f} {recall[i]:>9.4f} {f1[i]:>9.4f} {int(support[i]):>9d}")
    lines.append("")
    lines.append(f"{'accuracy':<15} {'':>9} {'':>9} {accuracy:>9.4f} {int(support.sum()):>9d}")
    lines.append(f"{'macro avg':<15} {macro_p:>9.4f} {macro_r:>9.4f} {macro_f1:>9.4f} {int(support.sum()):>9d}")
    lines.append(f"{'weighted avg':<15} {weighted_p:>9.4f} {weighted_r:>9.4f} {weighted_f1:>9.4f} {int(support.sum()):>9d}")

    metrics = {
        "accuracy": accuracy,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_p,
        "weighted_recall": weighted_r,
        "weighted_f1": weighted_f1,
        "total_support": int(support.sum()),
    }
    return "\n".join(lines), metrics


def _iter_csv_paths(data_path: Union[str, Path, List[Union[str, Path]]]) -> List[Path]:
    """
    Accept a CSV file, a folder, or a list of them.
    Returns a list of CSV file paths.
    """
    if isinstance(data_path, list):
        paths = [Path(p) for p in data_path]
    else:
        paths = [Path(data_path)]

    files: List[Path] = []
    for p in paths:
        if p.is_dir():
            files.extend(sorted(p.glob("*.csv")))
        elif p.is_file() and p.suffix.lower() == ".csv":
            files.append(p)
        else:
            raise FileNotFoundError(f"Not a CSV file or folder: {p}")

    if not files:
        raise FileNotFoundError(f"No CSV files found in: {data_path}")
    return files


@torch.no_grad()
def evaluate_path_streaming(
    bundle: DeployBundle,
    data_path: Union[str, Path, List[Union[str, Path]]],
    *,
    chunksize: int = 200_000,
    batch_size: int = 8192,
    max_rows: Optional[int] = None,
    measure_time: bool = True,
) -> Dict:
    """
    Streaming evaluation over CSV(s) without loading everything into RAM.
    Produces:
      - confusion matrix
      - classification report
      - end-to-end timing (preprocess+scale+forward), throughput
      - params + analytic MFLOPs/TFLOPs + optional THOP
    """
    files = _iter_csv_paths(data_path)
    class_names = list(map(str, bundle.le.classes_))
    C = len(class_names)

    cm = np.zeros((C, C), dtype=np.int64)
    total_rows = 0

    t_total_start = time.perf_counter() if measure_time else None
    preprocess_time = 0.0
    forward_time = 0.0

    def _sync():
        if bundle.device.type == "cuda":
            torch.cuda.synchronize()

    for fp in files:
        for chunk in pd.read_csv(fp, chunksize=chunksize):
            if LABEL_COL not in chunk.columns:
                raise ValueError(f"Missing label column '{LABEL_COL}' in {fp.name}")

            if max_rows is not None and total_rows >= max_rows:
                break

            if max_rows is not None and total_rows + len(chunk) > max_rows:
                chunk = chunk.iloc[: (max_rows - total_rows)].copy()

            # preprocess + scale
            t0 = time.perf_counter() if measure_time else None
            chunk = preprocess_raw_df(chunk)
            X = chunk[FEATURE_COLS].values.astype(np.float32)
            X = apply_scalers(X, bundle.mm, bundle.ss)
            y_true = _safe_label_transform(bundle.le, chunk[LABEL_COL].values)
            t1 = time.perf_counter() if measure_time else None
            if measure_time:
                preprocess_time += (t1 - t0)

            # forward in batches
            n = X.shape[0]
            start = 0
            while start < n:
                end = min(n, start + batch_size)
                Xb = X[start:end]
                yb = y_true[start:end]

                Xt = to_tensor(Xb, bundle.device)

                _sync()
                t2 = time.perf_counter() if measure_time else None
                out = bundle.model(Xt)
                if out.min().item() < 0.0 or out.max().item() > 1.0:
                    out = torch.softmax(out, dim=1)
                pred = torch.argmax(out, dim=1).detach().cpu().numpy()
                _sync()
                t3 = time.perf_counter() if measure_time else None
                if measure_time:
                    forward_time += (t3 - t2)

                cm += confusion_matrix(yb, pred, labels=np.arange(C))
                start = end

            total_rows += n

        if max_rows is not None and total_rows >= max_rows:
            break

    t_total_end = time.perf_counter() if measure_time else None

    report_text, report_metrics = report_from_confusion_matrix(cm, class_names)

    params = count_parameters(bundle.model)
    in_dim = len(FEATURE_COLS)
    flops_per = deepmlp_flops_per_sample(in_dim, C, count_softmax=True)
    total_flops = flops_per * total_rows

    out = {
        "files": [str(f) for f in files],
        "total_rows_evaluated": int(total_rows),
        "confusion_matrix": cm,
        "class_names": class_names,
        "classification_report": report_text,
        "report_metrics": report_metrics,
        "parameters": int(params),
        "parameter_bytes_fp32": int(params * 4),
        "flops_per_sample": int(flops_per),
        "mflops_per_sample": flops_per / 1e6,
        "tflops_per_sample": flops_per / 1e12,
        "total_flops": int(total_flops),
        "total_mflops": total_flops / 1e6,
        "total_tflops": total_flops / 1e12,
    }

    if measure_time and t_total_start is not None:
        total_time = t_total_end - t_total_start
        out.update({
            "device": str(bundle.device),
            "total_time_sec_end2end": float(total_time),
            "time_sec_preprocess": float(preprocess_time),
            "time_sec_forward": float(forward_time),
            "avg_latency_sec_per_sample_end2end": float(total_time / max(1, total_rows)),
            "throughput_samples_per_sec_end2end": float(total_rows / max(1e-12, total_time)),
            "achieved_tflops_end2end": float((total_flops / max(1e-12, total_time)) / 1e12),
        })

    thop = try_thop_profile(bundle.model, in_dim, bundle.device)
    if thop is not None:
        out.update(thop)
        out["thop_available"] = True
    else:
        out["thop_available"] = False

    return out


def evaluate_and_profile(
    artifact_dir: Union[str, Path],
    data_path: Union[str, Path, List[Union[str, Path]]],
    *,
    chunksize: int = 200_000,
    batch_size: int = 8192,
    max_rows: Optional[int] = None,
    with_softmax: bool = True,
    device: Optional[torch.device] = None,
    print_report: bool = True,
) -> Dict:
    """
    One-call convenience function.
    """
    bundle = load_deploy_bundle(artifact_dir, device=device, with_softmax=with_softmax)
    results = evaluate_path_streaming(
        bundle, data_path, chunksize=chunksize, batch_size=batch_size, max_rows=max_rows, measure_time=True
    )
    if print_report:
        print(results["classification_report"])
        print("\nConfusion Matrix:\n", results["confusion_matrix"])

        if "total_time_sec_end2end" in results:
            print("\n--- End-to-end timing ---")
            print("Device:", results.get("device"))
            print("Total rows:", results["total_rows_evaluated"])
            print("Total time (s):", results["total_time_sec_end2end"])
            print("Avg latency (s/sample):", results["avg_latency_sec_per_sample_end2end"])
            print("Throughput (samples/s):", results["throughput_samples_per_sec_end2end"])
            print("Achieved TFLOPs (end-to-end):", results["achieved_tflops_end2end"])

        print("\n--- Model complexity ---")
        print("Parameters:", results["parameters"])
        print("Param bytes (fp32):", results["parameter_bytes_fp32"])
        print("FLOPs/sample:", results["flops_per_sample"])
        print("MFLOPs/sample:", results["mflops_per_sample"])
        print("TFLOPs/sample:", results["tflops_per_sample"])

        if not results.get("thop_available", False):
            print("\nTHOP not installed. To enable THOP FLOPs/MACs:  %pip install thop")
        else:
            print("\n--- THOP (per sample) ---")
            print("THOP MACs:", results["thop_macs_per_sample"])
            print("THOP FLOPs:", results["thop_flops_per_sample"])
            print("THOP MFLOPs:", results["thop_mflops_per_sample"])
            print("THOP TFLOPs:", results["thop_tflops_per_sample"])

    return results