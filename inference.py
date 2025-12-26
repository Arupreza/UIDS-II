import os
import glob

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support
)
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)
import onnxruntime as ort

# Local imports
from utils import SegmentForValidation, SegmentForInference


#####################################################
########## RAW pt MODEL EVALUATION FUNCTION #########
#####################################################


# ==================================================
# Confusion Matrix Heatmap (Display Only)
# ==================================================
# import os
# import glob
# import torch
# import numpy as np
# import matplotlib.pyplot as plt

# from tqdm import tqdm
# from sklearn.metrics import (
#     accuracy_score,
#     precision_recall_fscore_support,
#     confusion_matrix
# )
# from transformers import (
#     AutoTokenizer,
#     AutoModelForSequenceClassification
# )

# import os
# import glob
# import torch
# import numpy as np
# import matplotlib.pyplot as plt

# from tqdm import tqdm
# from sklearn.metrics import (
#     accuracy_score,
#     precision_recall_fscore_support,
#     confusion_matrix
# )
# from transformers import (
#     AutoTokenizer,
#     AutoModelForSequenceClassification
# )

# ==================================================
# Binary Confusion Matrix (ROW-normalized, IDS style)
# ==================================================
def plot_binary_confusion_matrix_rowwise(cm, class_names):
    """
    Binary confusion matrix with:
    - True labels on X-axis
    - Predicted labels on Y-axis
    - ROW-normalized percentages (per predicted class)
    - Count + percentage per cell
    - Highlighted diagonal
    """

    cm = cm.astype(float)

    # ROW-wise normalization (Predicted labels)
    row_sum = cm.sum(axis=1, keepdims=True)
    cm_pct = np.divide(cm, row_sum, where=row_sum != 0) * 100

    fig, ax = plt.subplots(figsize=(7, 6))

    # Light background
    ax.imshow(cm_pct, cmap="Greys", alpha=0.15)

    # Axis labels
    ax.set_xticks(range(2))
    ax.set_yticks(range(2))
    ax.set_xticklabels(class_names, fontsize=14, fontweight="bold")
    ax.set_yticklabels(class_names, fontsize=14, fontweight="bold")

    ax.set_xlabel("True Labels", fontsize=16, fontweight="bold")
    ax.set_ylabel("Predicted Labels", fontsize=16, fontweight="bold")
    #ax.set_title("Confusion Matrix (Row-normalized %)", fontsize=18, fontweight="bold")

    # Strong grid (paper style)
    ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
    ax.grid(which="minor", color="black", linestyle="-", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Annotate cells
    for i in range(2):
        for j in range(2):
            count = int(cm[i, j])
            pct = cm_pct[i, j]

            # Highlight diagonal (precision of each class)
            if i == j:
                ax.add_patch(
                    plt.Rectangle(
                        (j - 0.5, i - 0.5),
                        1, 1,
                        color="#ccff66",
                        alpha=0.85,
                        zorder=0
                    )
                )

            ax.text(
                j, i,
                f"{count}\n{pct:.2f}%",
                ha="center",
                va="center",
                fontsize=13,
                fontweight="bold",
                color="black"
            )

    plt.tight_layout()
    plt.show()


# ==================================================
# End-to-End Binary Model Evaluation
# ==================================================
def EvaluateModel(
    model_path,
    validation_data_directory,
    time_gap,
    device="cuda",
    batch_size=16
):
    """
    End-to-end evaluation for binary IDS classifier
    with ROW-normalized confusion matrix.
    """

    print(f"\n--- Evaluating Binary Model: {model_path} ---")

    # -----------------------------
    # 1. Load model & tokenizer
    # -----------------------------
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()

    # -----------------------------
    # 2. Load validation data
    # -----------------------------
    all_chunks = []
    all_labels = []

    csv_files = glob.glob(os.path.join(validation_data_directory, "*.csv"))
    if not csv_files:
        raise FileNotFoundError("No validation CSV files found.")

    for file_path in tqdm(csv_files, desc="Loading validation data"):
        fname = os.path.basename(file_path)

        # MUST exist in your project
        chunks, labels = SegmentForValidation(
            validation_data_directory,
            fname,
            time_gap=time_gap
        )

        all_chunks.extend(chunks)
        all_labels.extend(labels)

    print(f"Total validation samples: {len(all_chunks)}")

    # -----------------------------
    # 3. Format input (training-compatible)
    # -----------------------------
    def format_chunk(chunk):
        tokens = []
        for t, g in chunk:
            tokens.append(f"T{int(t)}")
            tokens.append(f"G{int(g)}")
        return " ".join(tokens)

    texts = [format_chunk(c) for c in all_chunks]

    # -----------------------------
    # 4. Inference
    # -----------------------------
    all_preds = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Inference"):
        batch_texts = texts[i:i + batch_size]

        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits
            preds = torch.argmax(logits, dim=-1)

        all_preds.extend(preds.cpu().numpy())

    # -----------------------------
    # 5. Metrics
    # -----------------------------
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels,
        all_preds,
        average="binary"
    )

    print("\n=== Binary IDS Metrics ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")

    # -----------------------------
    # 6. Confusion Matrix (ROW-wise)
    # -----------------------------
    cm = confusion_matrix(all_labels, all_preds)

    plot_binary_confusion_matrix_rowwise(
        cm,
        class_names=["Attack Free", "Attack"]
    )

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }





# TIME_GAP_TEST = 83.0
# VALIDATION_DATA_DIRECTORY = "Split_data/Test/Tesla/DoS/Low"
# checkpoint_to_evaluate = "SFTSrc/bert/TrainKia/mobilebert-can-attack-classifier-experimental"

# print(f"\n--- Evaluating specific checkpoint: {checkpoint_to_evaluate} ---")
# EvaluateModel(
#     model_path=checkpoint_to_evaluate,
#     validation_data_directory=VALIDATION_DATA_DIRECTORY, # From your script's config
#     time_gap=TIME_GAP_TEST                             # From your script's config
# )

#####################################################
########### ONNX MODEL EVALUATION FUNCTION ##########
#####################################################


import os
import glob
import numpy as np
from tqdm import tqdm

import onnxruntime as ort
from transformers import AutoTokenizer

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from scipy.stats import norm


# ==================================================
# Confidence Interval Utilities
# ==================================================
def wilson_ci(k, n, confidence=0.95):
    """
    Wilson score confidence interval for a proportion.
    """
    if n == 0:
        return (0.0, 0.0)

    z = norm.ppf(1 - (1 - confidence) / 2)
    phat = k / n

    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    margin = z * np.sqrt(
        (phat * (1 - phat) + z**2 / (4 * n)) / n
    ) / denom

    return center - margin, center + margin


def bootstrap_ci(y_true, y_pred, metric_fn, n_bootstrap=1000, confidence=0.95):
    """
    Bootstrap confidence interval for metrics without closed form (e.g., F1).
    """
    rng = np.random.default_rng(42)
    scores = []
    n = len(y_true)

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        scores.append(metric_fn(y_true[idx], y_pred[idx]))

    lower = np.percentile(scores, (1 - confidence) / 2 * 100)
    upper = np.percentile(scores, (1 + confidence) / 2 * 100)

    return lower, upper


# ==================================================
# End-to-End ONNX Binary Evaluation
# ==================================================
def EvaluateModelOnnx(
    model_path,
    validation_data_directory,
    time_gap,
    device="cuda",
    batch_size=16
):
    """
    End-to-end evaluation for ONNX binary IDS classifier
    with 95% confidence intervals.
    """

    onnx_model_path = os.path.join(model_path, "model.onnx")
    print(f"\n--- Evaluating ONNX Model: {onnx_model_path} ---")

    # --------------------------------------------------
    # 1. Load tokenizer & ONNX model
    # --------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if device == "cuda" and ort.get_device() == "GPU":
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    session = ort.InferenceSession(onnx_model_path, providers=providers)
    input_names = [i.name for i in session.get_inputs()]
    output_name = session.get_outputs()[0].name

    # --------------------------------------------------
    # 2. Load validation data
    # --------------------------------------------------
    all_chunks, all_labels = [], []

    csv_files = glob.glob(os.path.join(validation_data_directory, "*.csv"))
    if not csv_files:
        raise FileNotFoundError("No validation CSV files found.")

    for file_path in tqdm(csv_files, desc="Loading validation data"):
        fname = os.path.basename(file_path)

        chunks, labels = SegmentForValidation(
            validation_data_directory,
            fname,
            time_gap=time_gap
        )

        all_chunks.extend(chunks)
        all_labels.extend(labels)

    print(f"Total validation samples: {len(all_chunks)}")

    # --------------------------------------------------
    # 3. Format input (same as training)
    # --------------------------------------------------
    def format_chunk(chunk):
        tokens = []
        for t, g in chunk:
            tokens.append(f"T{int(t)}")
            tokens.append(f"G{int(g)}")
        return " ".join(tokens)

    texts = [format_chunk(c) for c in all_chunks]

    # --------------------------------------------------
    # 4. ONNX inference
    # --------------------------------------------------
    all_preds = []

    for i in tqdm(range(0, len(texts), batch_size), desc="ONNX Inference"):
        batch_texts = texts[i:i + batch_size]

        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="np"
        )

        onnx_inputs = {
            name: inputs[name].astype(np.int64)
            for name in input_names
            if name in inputs
        }

        logits = session.run([output_name], onnx_inputs)[0]
        preds = np.argmax(logits, axis=-1)
        all_preds.extend(preds.tolist())

    # --------------------------------------------------
    # 5. Metrics + 95% Confidence Intervals
    # --------------------------------------------------
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Wilson CIs
    acc_ci = wilson_ci(np.sum(y_true == y_pred), len(y_true))
    prec_ci = wilson_ci(
        np.sum((y_pred == 1) & (y_true == 1)),
        np.sum(y_pred == 1)
    )
    rec_ci = wilson_ci(
        np.sum((y_pred == 1) & (y_true == 1)),
        np.sum(y_true == 1)
    )

    # Bootstrap CI for F1
    f1_ci = bootstrap_ci(y_true, y_pred, f1_score)

    print("\n=== Binary IDS Metrics (ONNX) with 95% CI ===")
    print(f"Accuracy : {acc:.4f}  (95% CI: {acc_ci[0]:.4f}, {acc_ci[1]:.4f})")
    print(f"Precision: {precision:.4f}  (95% CI: {prec_ci[0]:.4f}, {prec_ci[1]:.4f})")
    print(f"Recall   : {recall:.4f}  (95% CI: {rec_ci[0]:.4f}, {rec_ci[1]:.4f})")
    print(f"F1-score : {f1:.4f}  (95% CI: {f1_ci[0]:.4f}, {f1_ci[1]:.4f})")

    # --------------------------------------------------
    # 6. Confusion Matrix
    # --------------------------------------------------
    cm = confusion_matrix(y_true, y_pred)

    plot_binary_confusion_matrix_rowwise(
        cm,
        class_names=["Attack Free", "Attack"]
    )

    # --------------------------------------------------
    # 7. Return results
    # --------------------------------------------------
    return {
        "accuracy": acc,
        "accuracy_ci": acc_ci,
        "precision": precision,
        "precision_ci": prec_ci,
        "recall": recall,
        "recall_ci": rec_ci,
        "f1": f1,
        "f1_ci": f1_ci,
        "confusion_matrix": cm
    }