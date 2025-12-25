import os
import glob
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support
)
from transformers import AutoTokenizer
import onnxruntime as ort
from utils import SegmentForValidation, SegmentForInference


def EvaluateModelOnnx(model_path, data_directory, time_gap, mode="validation", device="cuda"):
    """
    Evaluate or run inference with an ONNX MobileBERT classifier on CAN data.

    Args:
        model_path (str): Directory containing model.onnx and tokenizer files.
        data_directory (str): Directory with CAN CSV files.
        time_gap (float): Time window (seconds) for segmentation.
        mode (str): "validation" (with labels) or "inference" (no labels).
        device (str): "cuda" or "cpu" – ONNX Runtime execution provider.

    Returns:
        For mode="validation":
            dict: {'accuracy', 'f1', 'precision', 'recall', 'cm'}
        For mode="inference":
            dict: {'segments', 'predictions'}
    """

    ONNX_FILE_NAME = "model.onnx"
    onnx_model_path = os.path.join(model_path, ONNX_FILE_NAME)
    print(f"\n--- Starting {mode.upper()} with ONNX Model: {onnx_model_path} ---")

    # --------------------- 1. Load model & tokenizer ---------------------
    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"Model directory not found: {model_path}")
    if not os.path.exists(onnx_model_path):
        raise FileNotFoundError(f"ONNX model file not found: {onnx_model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if device == "cuda" and ort.get_device() == "GPU":
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    session = ort.InferenceSession(onnx_model_path, providers=providers)

    input_names = [i.name for i in session.get_inputs()]
    output_names = [o.name for o in session.get_outputs()]
    print(f"ONNX Inputs: {input_names}")
    print(f"ONNX Outputs: {output_names}")

    # --------------------- 2. Load & segment CAN CSVs ---------------------
    csv_files = glob.glob(os.path.join(data_directory, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {data_directory}")

    all_chunks = []
    all_labels = []

    for file_path in tqdm(csv_files, desc=f"Processing files ({mode})"):
        filename = os.path.basename(file_path)
        if mode == "validation":
            chunks, labels = SegmentForValidation(data_directory, filename, time_gap=time_gap)
            all_chunks.extend(chunks)
            all_labels.extend(labels)
        else:  # inference
            chunks = SegmentForInference(data_directory, filename, time_gap=time_gap)
            all_chunks.extend(chunks)

    if not all_chunks:
        raise RuntimeError("No segments were generated from the input data.")

    print(f"Total segments loaded: {len(all_chunks)}")

    # --------------------- 3. Convert segments to text ---------------------
    def format_chunk_to_string(chunk):
        tokens = []
        for pair in chunk:
            tokens.append(f"T{int(pair[0])}")
            tokens.append(f"G{int(pair[1])}")
        return " ".join(tokens)

    texts = [format_chunk_to_string(ch) for ch in tqdm(all_chunks, desc="Formatting segments")]

    # --------------------- 4. Run ONNX inference ---------------------
    print("Running ONNX inference...")
    batch_size = 16
    all_preds = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Inference"):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="np",
            padding=True,
            truncation=True,
            max_length=512,
        )

        # Ensure int64 dtype for ONNX inputs
        onnx_inputs = {
            name: inputs[name].astype(np.int64)
            for name in input_names
            if name in inputs
        }

        outputs = session.run(output_names, onnx_inputs)
        logits = outputs[0]
        preds = np.argmax(logits, axis=-1)
        all_preds.extend(preds.tolist())

    # --------------------- 5. Post-process results ---------------------
    if mode == "validation":
        all_labels_np = np.array(all_labels)
        all_preds_np = np.array(all_preds)

        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels_np, all_preds_np, average="binary", zero_division=0
        )
        acc = accuracy_score(all_labels_np, all_preds_np)
        cm = confusion_matrix(all_labels_np, all_preds_np)

        metrics = {
            "accuracy": acc,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "cm": cm,
        }

        print("\n=== Validation Results (ONNX) ===")
        print(f"Accuracy:  {acc:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print("\nConfusion Matrix:\n", cm)
        print("=" * 50)

        return metrics

    else:  # inference mode
        print(f"\n✅ Inference completed on {len(all_chunks)} segments.")
        print("Sample predictions:", all_preds[:10])
        return {
            "segments": len(all_chunks),
            "predictions": all_preds,
        }


# # Optional: keep a simple CLI entry point for direct testing
# if __name__ == "__main__":
#     TIME_GAP = 100.0
#     DATA_DIR = "/home/lisa/Arupreza/UIDS-II/Split_data/Test/Kia/Lower Low"
#     MODEL_DIR = "/home/lisa/Arupreza/UIDS-II/UIDSApp/OnnxModels/TrainKiaOnnx"

#     device_to_use = "cuda" if ort.get_device() == "GPU" else "cpu"

#     EvaluateModelOnnx(
#         model_path=MODEL_DIR,
#         data_directory=DATA_DIR,
#         time_gap=TIME_GAP,
#         mode="validation",
#         device=device_to_use,
#     )