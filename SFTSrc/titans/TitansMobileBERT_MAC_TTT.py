import os, re, glob, argparse, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from dotenv import load_dotenv
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from transformers import (
    PretrainedConfig,
    PreTrainedModel,
    AutoTokenizer,
    MobileBertModel,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from transformers.modeling_outputs import SequenceClassifierOutput


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def discover_tg_tokens(texts, max_add: int = 50000):
    pat = re.compile(r"^[TG]\d+$")
    seen = set()
    for s in texts:
        for tok in s.split():
            if pat.match(tok):
                seen.add(tok)
                if len(seen) >= max_add:
                    break
        if len(seen) >= max_add:
            break
    return sorted(seen)


def get_segmenter():
    try:
        from utils import SegmentFromFile
        return SegmentFromFile
    except Exception as e:
        raise ImportError(
            "Could not import `SegmentFromFile` from `utils.py`.\n"
            "Ensure utils.py is in this folder (or PYTHONPATH).\n"
            f"Original error: {repr(e)}"
        )


def load_and_process_data(directory: str, time_gap: float, segment_from_file_fn):
    print(f"[Data] Loading from: {directory}")
    csv_files = sorted(glob.glob(os.path.join(directory, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {directory}")

    all_chunks, all_labels = [], []
    for fp in csv_files:
        fn = os.path.basename(fp)
        chunks, labels = segment_from_file_fn(directory, fn, time_gap=time_gap)
        all_chunks.extend(chunks)
        all_labels.extend(labels)

    def chunk_to_string(chunk):
        toks = []
        for pair in chunk:
            toks.append(f"T{int(pair[0])}")
            toks.append(f"G{int(pair[1])}")
        return " ".join(toks)

    texts = [chunk_to_string(c) for c in all_chunks]
    return pd.DataFrame({"text": texts, "label": all_labels})


class LinearAssociativeTTTMemory(nn.Module):
    def __init__(self, d_key, d_mem, num_steps, theta, eta, alpha, grad_clip):
        super().__init__()
        self.d_key = d_key
        self.d_mem = d_mem
        self.num_steps = num_steps
        self.theta = theta
        self.eta = eta
        self.alpha = alpha
        self.grad_clip = grad_clip

        self.W_init = nn.Parameter(torch.randn(d_mem, d_key) * 0.02)
        
        self.q_norm = nn.LayerNorm(d_key)
        self.k_norm = nn.LayerNorm(d_key)
        self.v_norm = nn.LayerNorm(d_mem)

    @staticmethod
    def _bmv(W, x):
        return torch.bmm(W, x.unsqueeze(-1)).squeeze(-1)

    def init_state(self, batch_size, device):
        W = self.W_init.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
        S = torch.zeros_like(W, device=device)
        return W, S

    def retrieve(self, q, W):
        q = self.q_norm(q)
        return self._bmv(W, q)

    def update(self, k, v, W, S, steps: int):
        k = self.k_norm(k)
        v = self.v_norm(v)

        for _ in range(steps):
            pred = self._bmv(W, k)
            err = pred - v
            grad = err.unsqueeze(-1) * k.unsqueeze(1)
            grad = torch.clamp(grad, -self.grad_clip, self.grad_clip)
            S = self.eta * S - self.theta * grad
            W = (1.0 - self.alpha) * W + S
            
        return W, S


class TitansMBConfig(PretrainedConfig):
    model_type = "titans_mobilebert_mac_ttt"

    def __init__(
        self,
        base_model_name="google/mobilebert-uncased",
        num_labels=2,
        segment_len=128,
        num_persistent_tokens=16,
        d_key=128,
        d_mem=128,
        num_ttt_steps=4,
        theta=0.05,
        eta=0.9,
        alpha=0.01,
        grad_clip=1.0,
        stop_grad_ttt=True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.base_model_name = base_model_name
        self.num_labels = num_labels
        self.segment_len = segment_len
        self.num_persistent_tokens = num_persistent_tokens
        self.d_key = d_key
        self.d_mem = d_mem
        self.num_ttt_steps = num_ttt_steps
        self.theta = theta
        self.eta = eta
        self.alpha = alpha
        self.grad_clip = grad_clip
        self.stop_grad_ttt = stop_grad_ttt


class TitansMobileBertMACTTT(PreTrainedModel):
    config_class = TitansMBConfig

    def __init__(self, config: TitansMBConfig, hf_token: str | None = None, class_weights=None):
        super().__init__(config)

        self.backbone = MobileBertModel.from_pretrained(config.base_model_name, token=hf_token)
        
        # MobileBERT Specifics
        # embedding_size is 128 (input to encoder)
        # hidden_size is 512 (output of encoder)
        self.emb_size = self.backbone.config.embedding_size
        self.hidden_size = self.backbone.config.hidden_size

        self.persistent_raw = nn.Parameter(torch.randn(config.num_persistent_tokens, self.hidden_size) * 0.02)

        # --- KEY FIX ---
        # 1. Projection for Query (comes from raw embeddings: size 128)
        self.emb_to_q = nn.Sequential(nn.Linear(self.emb_size, config.d_key), nn.SiLU())

        # 2. Projection for Key/Value (comes from encoder output: size 512)
        self.out_to_k = nn.Sequential(nn.Linear(self.hidden_size, config.d_key), nn.SiLU())
        self.out_to_v = nn.Sequential(nn.Linear(self.hidden_size, config.d_mem), nn.SiLU())

        self.ltm = LinearAssociativeTTTMemory(
            d_key=config.d_key,
            d_mem=config.d_mem,
            num_steps=config.num_ttt_steps,
            theta=config.theta,
            eta=config.eta,
            alpha=config.alpha,
            grad_clip=config.grad_clip,
        )

        self.mem_to_token = nn.Linear(config.d_mem, self.hidden_size)
        self.classifier = nn.Linear(self.hidden_size, config.num_labels)

        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

        self.post_init()

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.backbone.set_input_embeddings(value)

    @staticmethod
    def masked_mean(x, mask):
        mask = mask.unsqueeze(-1).type_as(x)
        denom = mask.sum(dim=1).clamp(min=1.0)
        return (x * mask).sum(dim=1) / denom

    def _run_segment(self, seg_ids, seg_mask, W, S, training: bool):
        B, Ls = seg_ids.shape
        device = seg_ids.device

        # Raw embeddings [B, Ls, 128]
        word_emb = self.get_input_embeddings()(seg_ids)

        # 1. RETRIEVE: uses raw embedding size (128)
        seg_summary = self.masked_mean(word_emb, seg_mask)  # [B, 128]
        q = self.emb_to_q(seg_summary)                      # [B, d_key]
        
        h = self.ltm.retrieve(q, W)                         # [B, d_mem]
        h_token = self.mem_to_token(h).unsqueeze(1)         # [B, 1, 512] -> expanded to hidden size

        P = self.persistent_raw.unsqueeze(0).expand(B, -1, -1) # [B, Np, 512]
        
        # We need to project word_emb (128) up to 512 to concatenate with P and h_token
        # MobileBERT normally handles this internal projection, but since we are
        # messing with inputs_embeds, we must be careful.
        # However, MobileBERT's forward() expects inputs_embeds of size `embedding_size` (128).
        # Wait, P and h_token are size 512. This concatenation will fail if word_emb is 128.
        
        # CRITICAL MOBILEBERT DETAIL: 
        # MobileBERT's embeddings are 128. The first layer projects them to 512.
        # But `inputs_embeds` argument typically replaces the lookup.
        # If we provide inputs_embeds, they must match the embedding layer output size (128).
        
        # RE-FIX:
        # We should keep P and h_token at size 128 (embedding size) so they can be concatenated
        # with word_emb (128) and passed into the backbone.
        
        # Let's override the P and h_token dimensions to match embedding size.
        
        pass 
        # (I will implement the dimension fix in the logic below)

class TitansMobileBertMACTTT_Fixed(PreTrainedModel):
    config_class = TitansMBConfig

    def __init__(self, config: TitansMBConfig, hf_token: str | None = None, class_weights=None):
        super().__init__(config)

        self.backbone = MobileBertModel.from_pretrained(config.base_model_name, token=hf_token)
        
        self.emb_size = self.backbone.config.embedding_size       # 128
        self.hidden_size = self.backbone.config.hidden_size       # 512

        # P and Memory should match EMBEDDING size (128) to enter the backbone
        self.persistent_raw = nn.Parameter(torch.randn(config.num_persistent_tokens, self.emb_size) * 0.02)
        self.mem_to_token = nn.Linear(config.d_mem, self.emb_size)

        # Query projection (from raw embeddings 128 -> d_key)
        self.emb_to_q = nn.Sequential(nn.Linear(self.emb_size, config.d_key), nn.SiLU())

        # Key/Value projection (from encoder output 512 -> d_key/d_mem)
        self.out_to_k = nn.Sequential(nn.Linear(self.hidden_size, config.d_key), nn.SiLU())
        self.out_to_v = nn.Sequential(nn.Linear(self.hidden_size, config.d_mem), nn.SiLU())

        self.ltm = LinearAssociativeTTTMemory(
            d_key=config.d_key, d_mem=config.d_mem, num_steps=config.num_ttt_steps,
            theta=config.theta, eta=config.eta, alpha=config.alpha, grad_clip=config.grad_clip
        )

        self.classifier = nn.Linear(self.hidden_size, config.num_labels)

        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)

        self.post_init()

    def get_input_embeddings(self): return self.backbone.get_input_embeddings()
    def set_input_embeddings(self, v): self.backbone.set_input_embeddings(v)
    
    @staticmethod
    def masked_mean(x, mask):
        mask = mask.unsqueeze(-1).type_as(x)
        denom = mask.sum(dim=1).clamp(min=1.0)
        return (x * mask).sum(dim=1) / denom

    def _run_segment(self, seg_ids, seg_mask, W, S, training):
        B, Ls = seg_ids.shape
        device = seg_ids.device

        # 1. Get raw word embeddings [B, Ls, 128]
        word_emb = self.get_input_embeddings()(seg_ids)

        # 2. Retrieve Memory (using raw embeddings)
        seg_summary = self.masked_mean(word_emb, seg_mask) # [B, 128]
        q = self.emb_to_q(seg_summary)                     # [B, d_key]
        h = self.ltm.retrieve(q, W)                        # [B, d_mem]
        
        # 3. Create context tokens (must be size 128 to match word_emb)
        h_token = self.mem_to_token(h).unsqueeze(1)        # [B, 1, 128]
        P = self.persistent_raw.unsqueeze(0).expand(B, -1, -1) # [B, Np, 128]

        # 4. Concatenate inputs
        inputs_embeds = torch.cat([P, h_token, word_emb], dim=1) # [B, Total, 128]
        
        prefix_len = self.config.num_persistent_tokens + 1
        prefix_mask = torch.ones(B, prefix_len, device=device, dtype=seg_mask.dtype)
        attn_mask = torch.cat([prefix_mask, seg_mask], dim=1)

        # 5. Run Backbone
        out = self.backbone(inputs_embeds=inputs_embeds, attention_mask=attn_mask).last_hidden_state
        # Output is [B, Total, 512] (MobileBERT projects up to 512 internally)

        # 6. Update Memory (using output 512)
        seg_out = out[:, -Ls:, :] # [B, Ls, 512]
        pooled = self.masked_mean(seg_out, seg_mask) # [B, 512]

        k = self.out_to_k(pooled) # [B, d_key]
        v = self.out_to_v(pooled) # [B, d_mem]

        steps = self.config.num_ttt_steps if (not training) else max(1, self.config.num_ttt_steps // 2)
        W, S = self.ltm.update(k, v, W, S, steps=steps)
        
        if self.config.stop_grad_ttt: W, S = W.detach(), S.detach()

        return pooled, W, S

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        if attention_mask is None: attention_mask = torch.ones_like(input_ids)
        B, T = input_ids.shape
        
        # Padding
        rem = T % self.config.segment_len
        if rem:
            pad = self.config.segment_len - rem
            input_ids = torch.cat([input_ids, input_ids.new_zeros(B, pad)], dim=1)
            attention_mask = torch.cat([attention_mask, attention_mask.new_zeros(B, pad)], dim=1)

        W, S = self.ltm.init_state(B, input_ids.device)
        pooled_last = None
        
        for i in range(0, input_ids.shape[1], self.config.segment_len):
            ids = input_ids[:, i:i+self.config.segment_len]
            mask = attention_mask[:, i:i+self.config.segment_len]
            pooled_last, W, S = self._run_segment(ids, mask, W, S, self.training)

        logits = self.classifier(pooled_last)
        loss = F.cross_entropy(logits, labels, weight=self.class_weights) if labels is not None else None
        return SequenceClassifierOutput(loss=loss, logits=logits)

def compute_metrics_fn(num_labels: int):
    def compute_metrics(pred):
        y_true = pred.label_ids
        y_pred = pred.predictions.argmax(-1)
        acc = accuracy_score(y_true, y_pred)
        avg = "binary" if num_labels == 2 else "macro"
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=avg, zero_division=0)
        return {"accuracy": acc, "f1": f1, "precision": p, "recall": r}
    return compute_metrics

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", type=str, required=True)
    ap.add_argument("--val_dir", type=str, required=True)
    ap.add_argument("--save_name", type=str, default="titans-mobilebert-mac-ttt")
    ap.add_argument("--output_dir", type=str, default="./TitansMobileBERT_MAC_TTT")
    ap.add_argument("--tokenizer_name", type=str, default="google/mobilebert-uncased")
    ap.add_argument("--base_model_name", type=str, default="google/mobilebert-uncased")
    ap.add_argument("--hf_token", type=str, default=None)
    ap.add_argument("--time_gap_train", type=float, default=100.0)
    ap.add_argument("--time_gap_val", type=float, default=93.0)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--train_bs", type=int, default=8)
    ap.add_argument("--eval_bs", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--segment_len", type=int, default=128)
    ap.add_argument("--persistent_tokens", type=int, default=16)
    ap.add_argument("--d_key", type=int, default=128)
    ap.add_argument("--d_mem", type=int, default=128)
    ap.add_argument("--ttt_steps", type=int, default=4)
    ap.add_argument("--theta", type=float, default=0.05)
    ap.add_argument("--eta", type=float, default=0.9)
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--stop_grad_ttt", action="store_true")
    ap.add_argument("--add_tg_tokens", action="store_true")
    ap.add_argument("--max_add_tokens", type=int, default=50000)
    args = ap.parse_args()

    set_seed(args.seed)
    load_dotenv()
    hf_token = args.hf_token or os.getenv("HF_TOKEN")

    SegmentFromFile = get_segmenter()
    train_df = load_and_process_data(args.train_dir, args.time_gap_train, SegmentFromFile)
    val_df = load_and_process_data(args.val_dir, args.time_gap_val, SegmentFromFile)
    num_labels = int(pd.concat([train_df["label"], val_df["label"]]).nunique())

    raw = DatasetDict({"train": Dataset.from_pandas(train_df), "eval": Dataset.from_pandas(val_df)})
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, token=hf_token)

    if args.add_tg_tokens:
        toks = discover_tg_tokens(train_df["text"].tolist(), max_add=args.max_add_tokens)
        tokenizer.add_tokens(toks, special_tokens=False)

    def tok_fn(batch): return tokenizer(batch["text"], truncation=True, padding=False, max_length=args.max_len)
    tokenized = raw.map(tok_fn, batched=True).remove_columns(["text"])
    if "__index_level_0__" in tokenized["train"].column_names: tokenized = tokenized.remove_columns(["__index_level_0__"])
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch")

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    y = torch.tensor(train_df["label"].values, dtype=torch.long)
    counts = torch.bincount(y, minlength=num_labels).float()
    weights = counts.sum() / (counts + 1e-6)
    weights = weights / weights.mean()

    config = TitansMBConfig(
        base_model_name=args.base_model_name, num_labels=num_labels, segment_len=args.segment_len,
        num_persistent_tokens=args.persistent_tokens, d_key=args.d_key, d_mem=args.d_mem,
        num_ttt_steps=args.ttt_steps, theta=args.theta, eta=args.eta, alpha=args.alpha,
        grad_clip=args.grad_clip, stop_grad_ttt=args.stop_grad_ttt
    )

    # Use the FIXED model class
    model = TitansMobileBertMACTTT_Fixed(config=config, hf_token=hf_token, class_weights=weights)
    model.resize_token_embeddings(len(tokenizer))

    train_args = TrainingArguments(
        output_dir=args.output_dir, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_bs, per_device_eval_batch_size=args.eval_bs,
        gradient_accumulation_steps=args.grad_accum, learning_rate=args.lr,
        optim="adamw_torch", logging_steps=25, eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="f1", greater_is_better=True,
        max_grad_norm=1.0, fp16=torch.cuda.is_available()
    )

    trainer = Trainer(
        model=model, args=train_args, train_dataset=tokenized["train"], eval_dataset=tokenized["eval"],
        processing_class=tokenizer, data_collator=collator, compute_metrics=compute_metrics_fn(num_labels)
    )

    trainer.train()
    trainer.save_model(args.save_name)
    tokenizer.save_pretrained(args.save_name)

if __name__ == "__main__":
    main()


# python TitansMobileBERT_MAC_TTT.py \
#     --train_dir /home/lisa/Arupreza/UIDS-II/Split_data/Train/Kia \
#     --val_dir /home/lisa/Arupreza/UIDS-II/Split_data/Val \
#     --save_name titans-mobilebert-mac-ttt \
#     --output_dir ./Titans_MB_TTT_Output \
#     --time_gap_train 100.0 \
#     --time_gap_val 93.0 \
#     --epochs 15 \
#     --train_bs 8 \
#     --eval_bs 8 \
#     --grad_accum 2 \
#     --add_tg_tokens