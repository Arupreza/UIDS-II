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
from utils import SegmentForValidation, SegmentForInference, ChunkSizeMatch
import argparse


# ==========================================================
#  ONNX MODEL EVALUATION / INFERENCE FUNCTION
# ==========================================================
def EvaluateModelOnnx(model_path, data_directory, mode="validation", device="cuda"):
    """
    Evaluate or infer using an ONNX MobileBERT classifier on CAN data.

    Args:
        model_path (str): Directory containing model.onnx and tokenizer config.
        data_directory (str): Directory containing CAN CSV files.
        time_gap (float): Time gap for segmenting data.
        mode (str): 'validation' or 'inference'.
        device (str): 'cuda' or 'cpu'.

    Returns:
        dict: For validation → metrics dict
            For inference → predictions summary
    """
    ONNX_FILE_NAME = "model.onnx"
    onnx_model_path = os.path.join(model_path, ONNX_FILE_NAME)
    print(f"\n--- Starting {mode.upper()} with ONNX Model: {onnx_model_path} ---")

    # --- 1. Load model & tokenizer ---
    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"Model directory not found: {model_path}")
    if not os.path.exists(onnx_model_path):
        raise FileNotFoundError(f"ONNX model file not found: {onnx_model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == "cuda" else ['CPUExecutionProvider']
    session = ort.InferenceSession(onnx_model_path, providers=providers)

    input_names = [i.name for i in session.get_inputs()]
    output_names = [o.name for o in session.get_outputs()]
    print(f"ONNX Inputs: {input_names}")
    print(f"ONNX Outputs: {output_names}")

    # --- 2. Load and preprocess CAN CSV data ---
    csv_files = glob.glob(os.path.join(data_directory, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {data_directory}")
    time_gap = ChunkSizeMatch(csv_files)
    all_chunks, all_labels = [], []
    for file_path in tqdm(csv_files, desc=f"Processing files ({mode})"):
        filename = os.path.basename(file_path)
        if mode == "validation":
            chunks, labels = SegmentForValidation(data_directory, filename, time_gap=time_gap)
            all_chunks.extend(chunks)
            all_labels.extend(labels)
        else:
            chunks = SegmentForInference(data_directory, filename, time_gap=time_gap)
            all_chunks.extend(chunks)

    print(f"Total segments loaded: {len(all_chunks)}")

    # --- 3. Convert features to text format ---
    def format_chunk_to_string(chunk):
        tokens = []
        for pair in chunk:
            tokens += [f"T{int(pair[0])}", f"G{int(pair[1])}"]
        return " ".join(tokens)

    texts = [format_chunk_to_string(ch) for ch in tqdm(all_chunks, desc="Formatting segments")]

    # --- 4. Run inference ---
    print(f"Running inference on {len(texts)} segments ...")
    batch_size = 16
    all_preds = []

    for i in tqdm(range(0, len(texts), batch_size), desc="ONNX Inference"):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="np",
            padding=True,
            truncation=True,
            max_length=512
        )
        onnx_inputs = {name: inputs[name].astype(np.int64) for name in input_names if name in inputs}
        outputs = session.run(output_names, onnx_inputs)
        logits = outputs[0]
        preds = np.argmax(logits, axis=-1)
        all_preds.extend(preds.tolist())

    # --- 5. Postprocess ---
    if mode == "validation":
        all_labels_np = np.array(all_labels)
        all_preds_np = np.array(all_preds)

        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels_np, all_preds_np, average='binary', zero_division=0
        )
        acc = accuracy_score(all_labels_np, all_preds_np)
        cm = confusion_matrix(all_labels_np, all_preds_np)

        metrics = {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'cm': cm
        }

        print("\n=== Validation Results (ONNX) ===")
        print(f"Accuracy:  {acc:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print("\nConfusion Matrix:\n", cm)
        print("=" * 50)

        return metrics

    else:  # Inference mode
        print(f"\n✅ Inference completed on {len(all_chunks)} segments.")
        print(f"Sample predictions: {all_preds[:10]}")
        return {
            'segments': len(all_chunks),
            'predictions': all_preds
        }


# ==========================================================
#  SCRIPT ENTRY POINT
# ==========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate or run inference on ONNX CAN model.")
    parser.add_argument("--mode", type=str, default="validation",
                        choices=["validation", "inference"],
                        help="Select 'validation' (with labels) or 'inference' (real-life deployment).")
    parser.add_argument("--model", type=str, default="/home/lisa/Arupreza/UIDS-II/UIDSApp/OnnxModels/TrainKiaOnnx",
                        help="Path to ONNX model directory.")
    parser.add_argument("--data", type=str, default="/home/lisa/Arupreza/UIDS-II/Split_data/Test/Tesla/Lower Low",
                        help="Path to CAN CSV directory.")
    parser.add_argument("--gap", type=float, default=83.0, help="Time gap for segmentation.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"], help="Device to use for ONNX Runtime.")
    args = parser.parse_args()

    device_to_use = "cuda" if (args.device == "cuda" and ort.get_device() == 'GPU') else "cpu"

    print(f"\n--- Running {args.mode.upper()} ---")
    print(f"Model Path: {args.model}")
    print(f"Data Path:  {args.data}")
    print(f"Device:     {device_to_use}")

    EvaluateModelOnnx(
        model_path=args.model,
        data_directory=args.data,
        time_gap=args.gap,
        mode=args.mode,
        device=device_to_use
    )