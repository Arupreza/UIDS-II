import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import shutil
import json
from pathlib import Path
from transformers import PretrainedConfig, PreTrainedModel, AutoTokenizer

# ==========================================
# 1. CONFIGURATION
# ==========================================

# EXACT PATH to your existing trained model folder
CHECKPOINT_PATH = "/home/lisa/Arupreza/UIDS/UIDS-II/SFTSrc/titans/TrainTesla/titans-mac-can-classifier"

# Output folder
OUTPUT_PACKAGE_DIR = "Titans_ONNX_Deploy_512" 
ONNX_FILENAME = "model.onnx"

# ==========================================
# 2. DEFINITIONS (Standard Export Classes)
# ==========================================

class TitansConfig(PretrainedConfig):
    model_type = "titans"
    def __init__(self, vocab_size=30522, hidden_size=256, num_hidden_layers=4, num_attention_heads=4, 
                 intermediate_size=1024, num_persistent_tokens=16, segment_len=128, num_labels=2, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.num_persistent_tokens = num_persistent_tokens
        self.segment_len = segment_len
        self.num_labels = num_labels

class NeuralMemory(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.memory_cell = nn.GRUCell(input_dim, hidden_dim)
    def forward(self, x, h_prev):
        return self.memory_cell(x, h_prev)

# --- MANUAL ATTENTION (Fixed Attribute Error) ---
class ManualExportMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, batch_first=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # --- FIX: Store this attribute so ONNX can see it ---
        self.batch_first = batch_first 
        # ----------------------------------------------------

        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None, **kwargs):
        # 1. Linear Projection
        qkv = F.linear(query, self.in_proj_weight, self.in_proj_bias)
        
        # 2. Split
        batch_size, seq_len, _ = qkv.shape
        q, k, v = qkv.chunk(3, dim=-1)

        # 3. Reshape (Assuming batch_first=True logic)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 4. Attention
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale
        if attn_mask is not None: 
            attn_scores = attn_scores + attn_mask
        
        attn_probs = F.softmax(attn_scores, dim=-1)
        
        # 5. Output
        attn_output = (attn_probs @ v).transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(attn_output)
        return output, attn_probs

class ExportFriendlyTransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = ManualExportMultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.relu 

    def forward(self, src, src_mask=None, src_key_padding_mask=None, **kwargs):
        src_norm = self.norm1(src)
        attn_output, _ = self.self_attn(src_norm, src_norm, src_norm, key_padding_mask=src_key_padding_mask, attn_mask=src_mask)
        src = src + self.dropout1(attn_output)
        src_norm2 = self.norm2(src)
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(src_norm2))))
        src = src + self.dropout2(ff_output)
        return src

class TitansMACClassifier(PreTrainedModel):
    config_class = TitansConfig
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.num_labels = config.num_labels
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.persistent_memory = nn.Parameter(torch.randn(config.num_persistent_tokens, config.hidden_size))
        self.neural_memory = NeuralMemory(config.hidden_size, config.hidden_size)
        
        custom_layer = ExportFriendlyTransformerLayer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads,
            dim_feedforward=config.intermediate_size,
            dropout=0.1
        )
        self.core_attention = nn.TransformerEncoder(custom_layer, num_layers=config.num_hidden_layers)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        
    def forward(self, input_ids=None, **kwargs):
        x = self.embedding(input_ids)
        batch_size, seq_len, hidden_size = x.shape
        
        # Padding
        remainder = seq_len % self.config.segment_len
        if remainder > 0:
            pad_len = self.config.segment_len - remainder
            padding = torch.zeros(batch_size, pad_len, hidden_size, device=x.device)
            x = torch.cat([x, padding], dim=1)
        
        segments = x.split(self.config.segment_len, dim=1)
        memory_state = torch.zeros(batch_size, hidden_size, device=x.device)
        
        for segment in segments:
            context = memory_state.unsqueeze(1)
            persistent_batch = self.persistent_memory.unsqueeze(0).expand(batch_size, -1, -1)
            combined_input = torch.cat([persistent_batch, context, segment], dim=1)
            core_out = self.core_attention(combined_input)
            segment_out = core_out[:, -self.config.segment_len:, :]
            segment_summary = segment_out.mean(dim=1)
            memory_state = self.neural_memory(segment_summary, memory_state)
            
        logits = self.classifier(memory_state)
        return logits 

# ==========================================
# 3. EXPORT PIPELINE
# ==========================================

def export_pipeline():
    print(f"--- Starting Titans Export (Fixed 512 Seq Len) ---")
    
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"\n❌ ERROR: Folder not found: {CHECKPOINT_PATH}")
        return

    Path(OUTPUT_PACKAGE_DIR).mkdir(parents=True, exist_ok=True)

    print("\n[1/4] Loading Weights...")
    try:
        model = TitansMACClassifier.from_pretrained(CHECKPOINT_PATH, local_files_only=True)
        model.eval()
        model.cpu()
        print("      ✅ Success.")
    except Exception as e:
        print(f"      ❌ ERROR: {e}")
        return

    print("\n[2/4] Exporting to ONNX...")
    
    # 512 tokens -> Unrolls to 4 segments (128*4)
    dummy_input = torch.randint(0, 30522, (1, 512), dtype=torch.long)
    onnx_path = os.path.join(OUTPUT_PACKAGE_DIR, ONNX_FILENAME)
    
    try:
        with torch.no_grad():
            torch.onnx.export(
                model,
                (dummy_input,),
                onnx_path,
                export_params=True,
                opset_version=14, 
                do_constant_folding=True,
                input_names=['input_ids'],
                output_names=['logits'],
                # Fixed Graph (No dynamic axes for sequence length)
                dynamic_axes={
                    'input_ids': {0: 'batch_size'}, 
                    'logits':    {0: 'batch_size'}
                }
            )
        print(f"      ✅ Saved ONNX model (Fixed 512 len) to {onnx_path}")
    except Exception as e:
        print(f"      ❌ Export Failed: {e}")
        return

    print(f"\n[3/4] Copying Tokenizer Files...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_PATH, local_files_only=True)
        tokenizer.save_pretrained(OUTPUT_PACKAGE_DIR)
        print("      ✅ Tokenizer files copied.")
    except:
        print("      ⚠️  Using default tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("google/mobilebert-uncased")
        tokenizer.save_pretrained(OUTPUT_PACKAGE_DIR)

    print("\n[4/4] Creating Config...")
    with open(os.path.join(OUTPUT_PACKAGE_DIR, "inference_config.json"), "w") as f:
        json.dump({
            "model_type": "titans-mac",
            "onnx_filename": ONNX_FILENAME,
            "max_seq_len": 512,
            "labels": ["Normal", "Attack"],
            "segment_len": 128
        }, f, indent=4)
    
    print("\n🚀 DONE. Use the folder 'Titans_ONNX_Deploy_512' for evaluation.")

if __name__ == "__main__":
    export_pipeline()


import glob
import numpy as np
import os
import onnxruntime as ort
from tqdm import tqdm
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support
)
from transformers import AutoTokenizer

# Assumes utils.py is in the same directory
from utils import SegmentFromFile 

def evaluate_model_onnx(model_path, validation_data_directory, time_gap, use_gpu=True):
    """
    Evaluates the Titans ONNX model.
    """
    onnx_file_path = os.path.join(model_path, ONNX_FILE_NAME)
    print(f"\n--- Starting Evaluation of ONNX Model: {onnx_file_path} ---")

    if not os.path.exists(onnx_file_path):
        raise FileNotFoundError(f"ONNX model not found at: {onnx_file_path}")
        
    # 1. Load Tokenizer
    print("Loading Tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except Exception as e:
        print(f"Warning: Could not load tokenizer from {model_path}. Downloading default...")
        tokenizer = AutoTokenizer.from_pretrained("google/mobilebert-uncased")

    # 2. Setup ONNX Runtime
    available_providers = ort.get_available_providers()
    if use_gpu and 'CUDAExecutionProvider' in available_providers:
        print(f"✅ Using GPU (CUDAExecutionProvider)")
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    else:
        print(f"ℹ️ Using CPU (CPUExecutionProvider)")
        providers = ['CPUExecutionProvider']
        
    session = ort.InferenceSession(onnx_file_path, providers=providers)
    
    # Get expected input names (Critical for Titans which only expects input_ids)
    model_input_names = [i.name for i in session.get_inputs()]
    model_output_names = [o.name for o in session.get_outputs()]
    print(f"Model Inputs: {model_input_names}")

    # 3. Process Data
    print(f"Loading validation data from: {validation_data_directory}")
    csv_files = glob.glob(os.path.join(validation_data_directory, "*.csv"))
    if not csv_files:
        raise FileNotFoundError("No CSV files found.")

    all_chunks = []
    all_labels = []

    for file_path in tqdm(csv_files, desc="Processing files"):
        filename = os.path.basename(file_path)
        chunks, labels = SegmentFromFile(validation_data_directory, filename, time_gap=time_gap)
        all_chunks.extend(chunks)
        all_labels.extend(labels)

    # Format text (Must match training logic exactly)
    def format_chunk_to_string(chunk):
        tokens = []
        for pair in chunk:
            tokens.append(f"T{int(pair[0])}")
            tokens.append(f"G{int(pair[1])}")
        return " ".join(tokens)

    texts = [format_chunk_to_string(chunk) for chunk in all_chunks]

    # 4. Inference Loop
    print("Running inference...")
    batch_size = 16
    all_preds = []
    
    num_batches = (len(texts) + batch_size - 1) // batch_size
    for i in tqdm(range(0, len(texts), batch_size), total=num_batches):
        batch_texts = texts[i:i + batch_size]
        
        # Tokenize
        inputs = tokenizer(
            batch_texts,
            return_tensors="np", # Get numpy arrays directly
            padding="max_length", # Pad to max length for consistency
            truncation=True,
            max_length=512
        )

        # Prepare ONNX Inputs
        # STRICT FILTERING: Only pass what the ONNX graph asks for (usually just 'input_ids')
        # This prevents errors if tokenizer generates 'attention_mask' but model doesn't want it.
        ort_inputs = {}
        for name in model_input_names:
            if name in inputs:
                ort_inputs[name] = inputs[name].astype(np.int64) # Enforce int64
        
        # Run
        outputs = session.run(model_output_names, ort_inputs)
        
        # Logits are usually the first output
        logits = outputs[0]
        preds = np.argmax(logits, axis=-1)
        all_preds.extend(preds)

    # 5. Metrics
    print("\nCalculating metrics...")
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    print("\n=== Titans ONNX Results ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    
    tn, fp, fn, tp = cm.ravel()
    print("\n=== Confusion Matrix ===")
    print(f"Actual Neg | Pred Neg: {tn} | Pred Pos: {fp}")
    print(f"Actual Pos | Pred Neg: {fn} | Pred Pos: {tp}")
    print("=" * 30)

# --- CONFIGURATION ---
ONNX_FILE_NAME = "model.onnx"
# Path where you saved the ONNX package (must contain model.onnx and tokenizer files)
ONNX_MODEL_DIRECTORY = "/home/lisa/Arupreza/UIDS/UIDS-II/SFTSrc/titans/OnnxModels/TrainGenOnnx" 
VALIDATION_DATA_DIRECTORY = "/home/lisa/Arupreza/UIDS-II/Split_data/Test/Tesla/Lower Low"
TIME_GAP_TEST = 83.0

want_gpu = True 

evaluate_model_onnx(
    model_path=ONNX_MODEL_DIRECTORY,
    validation_data_directory=VALIDATION_DATA_DIRECTORY,
    time_gap=TIME_GAP_TEST,
    use_gpu=want_gpu
)