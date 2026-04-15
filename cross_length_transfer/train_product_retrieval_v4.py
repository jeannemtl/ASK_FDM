"""
Product Retrieval FDM — Hermes3-3B Fine-tune

Trains the FDM model to answer single-value retrieval questions
from the FDM signal using product channel labels.

Based on eidetic_hermes3_fdm_e2e.py with:
  - Product channel schemas (USER.*, PREF.*, TASK.*, PROJ.*)
  - Simple Q&A format: "What is USER.gpu?" → "RTX5090"
  - Full 5-stage curriculum (a_low 0.0 → 0.25)
  - Loads from prompterminal/fdm-40ch-hermes3-3b (already FDM-trained)
  - No bitsandbytes — uses standard AdamW (H100 has enough memory)

Usage:
  python train_product_retrieval_v3.py generate
  python train_product_retrieval_v3.py train
  python train_product_retrieval_v3.py eval [stage]
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim import AdamW
import json
import random
from tqdm import tqdm
import os
import sys


# ============================================================
# Model configuration
# ============================================================

MODEL_ID = "prompterminal/fdm-40ch-hermes3-3b"  # start from FDM-trained model
MODEL_SHORT = "hermes3_product"
VOCAB_SIZE = 128256
DTYPE = torch.bfloat16
BATCH_SIZE = 4        # H100 80GB can handle this
MAX_LENGTH = 1024
OUTPUT_PREFIX = "fdm_product_v3"
CHECKPOINT_DIR = "checkpoints_fdm_product_v3"
FINAL_MODEL_DIR = "fdm_product_v3_final"


# ============================================================
# Product channel schemas
# These map to FDM channels 0-19 (using only 20 channels)
# ============================================================

PRODUCT_SCHEMAS = {
    0:  ("USER.name",          ["Alice", "Bob", "Jeanne", "Carlos", "Diana"]),
    1:  ("USER.project",       ["FDM_NeurIPS", "spectral_memory", "hermes_plugin", "paper2", "benchmark"]),
    2:  ("USER.affiliation",   ["independent", "university", "startup", "corporate", "research_lab"]),
    3:  ("USER.gpu",           ["RTX5090", "H100", "A100", "RTX4090", "A40"]),
    4:  ("USER.working_dir",   ["/workspace", "/home/user", "/data", "/projects", "/research"]),
    5:  ("USER.os",            ["Ubuntu24", "Ubuntu22", "MacOS", "Windows11", "Debian"]),
    6:  ("PREF.response_style", ["concise", "detailed", "verbose", "minimal", "structured"]),
    7:  ("PREF.code_style",    ["functional", "oop", "scripting", "typed", "minimal"]),
    8:  ("PREF.verbosity",     ["low", "medium", "high", "concise", "verbose"]),
    9:  ("PREF.format",        ["markdown", "plain", "json", "structured", "bullet"]),
    10: ("TASK.current",       ["benchmark_eval", "writing_paper", "debugging", "training_model", "data_analysis"]),
    11: ("TASK.next",          ["write_latex", "run_eval", "push_to_hf", "submit_pr", "review_results"]),
    12: ("TASK.blocked_on",    ["CLUTRR_rerun", "HPC_access", "data_collection", "review", "none"]),
    13: ("TASK.deadline",      ["NeurIPS2025", "ICLR2026", "ICML2026", "next_week", "none"]),
    14: ("TASK.priority",      ["high", "medium", "low", "critical", "deferred"]),
    15: ("PROJ.status",        ["near_submission", "in_progress", "blocked", "complete", "planning"]),
    16: ("PROJ.last_result",   ["100pct_rule_accuracy", "98.7_retrieval", "loss_0.05", "eval_passed", "training_failed"]),
    17: ("PROJ.model_target",  ["Hermes3_3B", "Qwen3_0.6B", "LFM2_1.2B", "GPT2_345M", "all_four"]),
    18: ("PROJ.token_budget",  ["512", "1024", "2048", "256", "4096"]),
    19: ("PROJ.accuracy",      ["100.0", "99.9", "98.7", "95.0", "85.0"]),
}

NUM_CHANNELS = 20  # only 20 product channels


# ============================================================
# FDM Encoder — uses same signal math but product schemas
# ============================================================

_global_tokenizer = None

def get_tokenizer():
    global _global_tokenizer
    if _global_tokenizer is None:
        _global_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    return _global_tokenizer


class ProductFDMEncoder:
    """FDM encoder using product channel schemas."""

    def __init__(self, num_tokens_per_encoder=256, sample_rate=100.0,
                 a_high=1.0, a_low=0.0, num_levels=64, seed=42):
        self.num_tokens_per_encoder = num_tokens_per_encoder
        self.total_tokens = num_tokens_per_encoder * 2
        self.sample_rate = sample_rate
        self.a_high = a_high
        self.a_low = a_low
        self.num_levels = num_levels
        self.num_channels = NUM_CHANNELS
        self.carrier_freqs = [1.0 + i * 1.0 for i in range(NUM_CHANNELS)]

        S = int(np.sqrt(num_tokens_per_encoder / 2))
        self.interleaver = self._generate_s_random_interleaver(num_tokens_per_encoder, S)

        rng = np.random.RandomState(seed)
        self.token_map = rng.choice(VOCAB_SIZE, size=num_levels, replace=False)

        self.tokenizer = get_tokenizer()

    def _generate_s_random_interleaver(self, length, S):
        import random as rnd
        rnd.seed(42)
        interleaver = list(range(length))
        for i in range(length):
            for _ in range(100):
                j = rnd.randint(i, length - 1)
                valid = True
                for k in range(max(0, i - S + 1), i):
                    if abs(interleaver[j] - interleaver[k]) < S:
                        valid = False
                        break
                if valid:
                    interleaver[i], interleaver[j] = interleaver[j], interleaver[i]
                    break
        return interleaver

    def value_to_bits(self, channel_id, value):
        _, values = PRODUCT_SCHEMAS[channel_id]
        idx = values.index(value)
        num_bits = max(1, int(np.ceil(np.log2(max(len(values), 2)))))
        return format(idx, f'0{num_bits}b'), num_bits

    def encode_memory(self, memory):
        all_bits = {}
        max_bits = 0
        for ch in range(self.num_channels):
            bits, nb = self.value_to_bits(ch, memory[ch])
            all_bits[ch] = bits
            max_bits = max(max_bits, nb)
        for ch in range(self.num_channels):
            all_bits[ch] = all_bits[ch].ljust(max_bits, '0')

        t = np.arange(self.num_tokens_per_encoder) / self.sample_rate
        samples_per_bit = self.num_tokens_per_encoder // max_bits

        composite = np.zeros(self.num_tokens_per_encoder)
        for ch in range(self.num_channels):
            bits = all_bits[ch]
            for bi, bit in enumerate(bits):
                start = bi * samples_per_bit
                end = min((bi + 1) * samples_per_bit, self.num_tokens_per_encoder)
                amp = self.a_high if bit == '1' else self.a_low
                composite[start:end] += amp * np.sin(
                    2 * np.pi * self.carrier_freqs[ch] * t[start:end]
                )

        sig_min, sig_max = composite.min(), composite.max()
        sig_range = sig_max - sig_min + 1e-10
        norm1 = (composite - sig_min) / sig_range
        q1 = np.floor(norm1 * (self.num_levels - 1) + 0.5).astype(int)
        q1 = np.clip(q1, 0, self.num_levels - 1)
        tokens1 = [int(self.token_map[q]) for q in q1]

        interleaved = composite[self.interleaver]
        norm2 = (interleaved - sig_min) / sig_range
        q2 = np.floor(norm2 * (self.num_levels - 1) + 0.5).astype(int)
        q2 = np.clip(q2, 0, self.num_levels - 1)
        tokens2 = [int(self.token_map[q]) for q in q2]

        all_tokens = tokens1 + tokens2
        fdm_text = self.tokenizer.decode(all_tokens)
        return fdm_text, all_tokens


# ============================================================
# Dataset
# ============================================================

class ProductRetrievalDataset(Dataset):
    def __init__(self, path, tokenizer, max_length=MAX_LENGTH):
        self.samples = [json.loads(l) for l in open(path)]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        input_text = f"[MEMORY]{s['fdm_text']}[/MEMORY]\nQuestion: {s['question']}\nAnswer:"
        full_text = input_text + " " + s['answer']

        encoding = self.tokenizer(
            full_text, truncation=True, max_length=self.max_length,
            padding='max_length', return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        labels = input_ids.clone()

        input_len = len(self.tokenizer.encode(input_text))
        labels[:input_len] = -100
        labels[attention_mask == 0] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }


# ============================================================
# Data generation
# ============================================================

def generate_stage_data(stage_config, num_samples, output_prefix):
    encoder = ProductFDMEncoder(
        num_tokens_per_encoder=stage_config.get('num_tokens', 256),
        sample_rate=stage_config.get('sample_rate', 100.0),
        a_high=stage_config.get('a_high', 1.0),
        a_low=stage_config.get('a_low', 0.0),
        num_levels=stage_config.get('num_levels', 64),
    )

    samples = []
    print(f"Generating {num_samples} samples (a_low={stage_config.get('a_low', 0.0)})...")

    for _ in tqdm(range(num_samples)):
        # Encode all 20 product channels
        memory = {}
        for ch in range(NUM_CHANNELS):
            _, values = PRODUCT_SCHEMAS[ch]
            memory[ch] = random.choice(values)

        fdm_text, _ = encoder.encode_memory(memory)

        # Pick a random channel to query
        query_ch = random.randint(0, NUM_CHANNELS - 1)
        label = PRODUCT_SCHEMAS[query_ch][0]
        answer = memory[query_ch]
        question = f"What is {label}?"

        samples.append({
            "fdm_text": fdm_text,
            "question": question,
            "answer": answer,
            "label": label,
            "channel": query_ch,
            "memory": {str(k): v for k, v in memory.items()},
            "stage_config": {k: v for k, v in stage_config.items() if k != 'name'},
        })

    random.shuffle(samples)
    n = len(samples)
    splits = {
        'train': samples[:int(0.85 * n)],
        'val':   samples[int(0.85 * n):int(0.95 * n)],
        'test':  samples[int(0.95 * n):],
    }

    for split_name, split_data in splits.items():
        path = f"{output_prefix}_{split_name}.jsonl"
        with open(path, 'w') as f:
            for s in split_data:
                f.write(json.dumps(s) + "\n")
        print(f"  {split_name}: {len(split_data)} samples -> {path}")

    return samples


STAGES = [
    {"name": "Stage 0: Easy",     "num_tokens": 256, "a_high": 1.0, "a_low": 0.0,  "num_levels": 64, "samples": 15000, "epochs": 5,  "lr": 5e-5},
    {"name": "Stage 1: Standard", "num_tokens": 256, "a_high": 1.0, "a_low": 0.1,  "num_levels": 64, "samples": 20000, "epochs": 5,  "lr": 3e-5},
    {"name": "Stage 2: Moderate", "num_tokens": 256, "a_high": 1.0, "a_low": 0.15, "num_levels": 64, "samples": 25000, "epochs": 7,  "lr": 2e-5},
    {"name": "Stage 3: Harder",   "num_tokens": 256, "a_high": 1.0, "a_low": 0.2,  "num_levels": 64, "samples": 25000, "epochs": 7,  "lr": 1e-5},
    {"name": "Stage 4: Hardest",  "num_tokens": 256, "a_high": 1.0, "a_low": 0.25, "num_levels": 64, "samples": 30000, "epochs": 10, "lr": 1e-5},
]


def generate_all_stages():
    for i, stage in enumerate(STAGES):
        print(f"\n{'='*60}")
        print(f"GENERATING STAGE {i}: {stage['name']}")
        print(f"{'='*60}")
        generate_stage_data(stage, stage['samples'], f"{OUTPUT_PREFIX}_stage{i}")


# ============================================================
# Training
# ============================================================

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Base model: {MODEL_ID}")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    special_tokens = tokenizer.add_special_tokens({
        'additional_special_tokens': ['[MEMORY]', '[/MEMORY]']
    })
    print(f"Added {special_tokens} special tokens")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        trust_remote_code=True,
    )
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    model.gradient_checkpointing_enable()
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total_params:,}")

    # Resume logic
    resume_stage = 0
    resume_epoch = 0
    ckpt_files = sorted([f for f in os.listdir(CHECKPOINT_DIR) if f.endswith('.pt')])
    if ckpt_files:
        latest = ckpt_files[-1]
        print(f"Found checkpoint: {latest}")
        ckpt = torch.load(os.path.join(CHECKPOINT_DIR, latest), map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        resume_stage = ckpt.get('stage', 0)
        resume_epoch = ckpt.get('epoch', 0) + 1
        print(f"Resuming from stage {resume_stage}, epoch {resume_epoch}")

    for stage_idx, stage in enumerate(STAGES):
        if stage_idx < resume_stage:
            print(f"Skipping stage {stage_idx} (already completed)")
            continue

        prefix = f"{OUTPUT_PREFIX}_stage{stage_idx}"
        train_path = f"{prefix}_train.jsonl"
        val_path = f"{prefix}_val.jsonl"

        if not os.path.exists(train_path):
            print(f"ERROR: {train_path} not found. Run 'generate' first.")
            return

        print(f"\n{'='*60}")
        print(f"STAGE {stage_idx}: {stage['name']}")
        print(f"  lr={stage['lr']}, epochs={stage['epochs']}")
        print(f"{'='*60}")

        train_dataset = ProductRetrievalDataset(train_path, tokenizer)
        val_dataset = ProductRetrievalDataset(val_path, tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

        optimizer = AdamW(model.parameters(), lr=stage['lr'])
        best_val_loss = float('inf')

        start_epoch = resume_epoch if stage_idx == resume_stage else 0
        resume_epoch = 0

        for epoch in range(start_epoch, stage['epochs']):
            model.train()
            train_loss = 0
            for batch in tqdm(train_loader, desc=f"S{stage_idx} E{epoch+1}/{stage['epochs']}"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                train_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            train_loss /= len(train_loader)

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    val_loss += outputs.loss.item()
            val_loss /= len(val_loader)

            print(f"  Stage {stage_idx} Epoch {epoch+1}: Train={train_loss:.4f}, Val={val_loss:.4f}")

            if (epoch + 1) % 2 == 0 or epoch == stage['epochs'] - 1:
                quick_eval(model, tokenizer, val_path, device, num_samples=10, stage_idx=stage_idx)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"  New best val loss: {val_loss:.4f}")

            torch.save({
                'model_state_dict': model.state_dict(),
                'stage': stage_idx,
                'epoch': epoch,
                'val_loss': val_loss,
            }, os.path.join(CHECKPOINT_DIR, f'stage{stage_idx}_epoch{epoch+1}.pt'))

        model.save_pretrained(f'{OUTPUT_PREFIX}_model_stage{stage_idx}')
        tokenizer.save_pretrained(f'{OUTPUT_PREFIX}_model_stage{stage_idx}')

    model.save_pretrained(FINAL_MODEL_DIR)
    tokenizer.save_pretrained(FINAL_MODEL_DIR)
    print(f"\nSaved final model: {FINAL_MODEL_DIR}/")


def quick_eval(model, tokenizer, val_path, device, num_samples=10, stage_idx=0):
    samples = [json.loads(l) for l in open(val_path)][:num_samples]
    model.eval()
    correct = 0
    total = 0

    for s in samples:
        input_text = f"[MEMORY]{s['fdm_text']}[/MEMORY]\nQuestion: {s['question']}\nAnswer:"
        input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)

        with torch.no_grad():
            output = model.generate(
                input_ids, max_new_tokens=20, do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
            )

        response = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
        if s['answer'].lower() in response.lower():
            correct += 1
        total += 1

    print(f"    Quick eval (S{stage_idx}): {correct}/{total} = {100*correct/total:.0f}%")


# ============================================================
# Full evaluation
# ============================================================

def evaluate(model_path=None, stage_idx=None):
    if model_path is None:
        model_path = FINAL_MODEL_DIR

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading model from {model_path}/...")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=DTYPE, trust_remote_code=True
    )
    model.to(device)
    model.eval()

    si = stage_idx if stage_idx is not None else 4
    test_path = f"{OUTPUT_PREFIX}_stage{si}_test.jsonl"

    if not os.path.exists(test_path):
        print(f"ERROR: {test_path} not found.")
        return

    test_samples = [json.loads(l) for l in open(test_path)]
    print(f"Evaluating on {len(test_samples)} samples...\n")

    correct = 0
    per_channel = {}
    total = 0

    for i, s in enumerate(tqdm(test_samples, desc="Evaluating")):
        input_text = f"[MEMORY]{s['fdm_text']}[/MEMORY]\nQuestion: {s['question']}\nAnswer:"
        input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)

        with torch.no_grad():
            output = model.generate(
                input_ids, max_new_tokens=20, do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
            )

        response = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
        ch = s['channel']
        label = s['label']
        answer = s['answer']

        ok = answer.lower() in response.lower()
        if ok:
            correct += 1
        per_channel[ch] = per_channel.get(ch, [0, 0])
        per_channel[ch][0] += int(ok)
        per_channel[ch][1] += 1
        total += 1

        if i < 5:
            print(f"  [{label}] Expected: {answer} | Got: {response[:50]}")

    print(f"\n{'='*60}")
    print(f"PRODUCT RETRIEVAL EVALUATION")
    print(f"{'='*60}")
    print(f"  Overall: {correct}/{total} = {100*correct/total:.1f}%")
    print(f"\nPer-channel accuracy:")
    for ch in sorted(per_channel.keys()):
        c, t = per_channel[ch]
        label = PRODUCT_SCHEMAS[ch][0]
        print(f"  ch{ch:2d} ({label:<25}): {100*c/t:.1f}%")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Product Retrieval FDM Training")
        print("Usage:")
        print("  python train_product_retrieval_v3.py generate")
        print("  python train_product_retrieval_v3.py train")
        print("  python train_product_retrieval_v3.py eval [stage]")
        sys.exit(0)

    cmd = sys.argv[1]
    if cmd == "generate":
        generate_all_stages()
    elif cmd == "train":
        train()
    elif cmd == "eval":
        stage = int(sys.argv[2]) if len(sys.argv) > 2 else None
        evaluate(stage_idx=stage)
