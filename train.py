"""
train.py
LoRA finetuning of Qwen2-VL on the RenAIssance handwritten dataset.

Runs the offline finetuning step once.
Saves adapter weights to CHECKPOINTS_DIR for use by pipeline.py.

Usage (on Kaggle):
    python train.py

Or from the notebook:
    from train import train
    train()
"""

import os
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import TrainingArguments, Trainer
from qwen_vl_utils import process_vision_info           

from config import (
    CHECKPOINTS_DIR, LOGS_DIR,
    EPOCHS, BATCH_SIZE, GRAD_ACCUM,
    LEARNING_RATE, WARMUP_RATIO,
    MAX_SEQ_LEN, SAVE_STEPS, EVAL_STEPS,
    LOGGING_STEPS, FP16, SEED, LORA_ADAPTER,
)
from dataset import HandwrittenLineDataset, split_dataset
from model import load_base_model, apply_lora


# ─────────────────────────────────────────────────────────────
# COLLATOR
# ─────────────────────────────────────────────────────────────

class VLMCollator:
    """
    Converts a batch of dataset items into tokenized tensors
    that the Qwen2-VL Trainer expects.

    The processor handles both image encoding and text tokenization
    in one call — we just need to format the messages correctly.
    """

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch: list[dict]) -> dict:
        all_messages = [item["messages"] for item in batch]

        # process_vision_info extracts image tensors from messages
        image_inputs, video_inputs = process_vision_info(all_messages)

        # build text prompts 
        texts = [
            self.processor.apply_chat_template(
                msgs[:-1],               #
                tokenize=False,
                add_generation_prompt=True,
            )
            for msgs in all_messages
        ]

        # labels = ground truth transcriptions
        labels_text = [item["text"] for item in batch]

        # tokenize inputs
        inputs = self.processor(
            text            = texts,
            images          = image_inputs,
            videos          = video_inputs,
            return_tensors  = "pt",
            padding         = True,
            truncation      = True,
            max_length      = MAX_SEQ_LEN,
        )

        # tokenize labels separately
        label_ids = self.processor.tokenizer(
            labels_text,
            return_tensors  = "pt",
            padding         = True,
            truncation      = True,
            max_length      = MAX_SEQ_LEN,
        ).input_ids

        # replace padding token id with -100 so loss ignores it
        label_ids[label_ids == self.processor.tokenizer.pad_token_id] = -100
        inputs["labels"] = label_ids

        return inputs


# ─────────────────────────────────────────────────────────────
# TRAINING ARGUMENTS
# ─────────────────────────────────────────────────────────────

def get_training_args() -> TrainingArguments:
    return TrainingArguments(
        output_dir                  = str(CHECKPOINTS_DIR),
        num_train_epochs            = EPOCHS,
        per_device_train_batch_size = BATCH_SIZE,
        per_device_eval_batch_size  = BATCH_SIZE,
        gradient_accumulation_steps = GRAD_ACCUM,
        learning_rate               = LEARNING_RATE,
        warmup_ratio                = WARMUP_RATIO,
        fp16                        = FP16,         
        logging_dir                 = str(LOGS_DIR),
        logging_steps               = LOGGING_STEPS,
        save_steps                  = SAVE_STEPS,
        eval_steps                  = EVAL_STEPS,
        evaluation_strategy         = "steps",
        save_total_limit            = 3,            # keep only last 3 checkpoints
        load_best_model_at_end      = True,
        metric_for_best_model       = "eval_loss",
        greater_is_better           = False,
        dataloader_num_workers      = 2,
        remove_unused_columns       = False,        
        report_to                   = "none",      
        seed                        = SEED,
        gradient_checkpointing      = True,
    )


# ─────────────────────────────────────────────────────────────
# TRAINING HISTORY  (saved to JSON for notebook plotting)
# ─────────────────────────────────────────────────────────────

class HistoryCallback:
    """Saves train/eval loss at each logging step."""

    def __init__(self):
        self.history = {"train_loss": [], "eval_loss": [], "steps": []}

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        step = state.global_step
        if "loss" in logs:
            self.history["train_loss"].append(logs["loss"])
            self.history["steps"].append(step)
        if "eval_loss" in logs:
            self.history["eval_loss"].append(logs["eval_loss"])

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"[train] History saved to {path}")


# ─────────────────────────────────────────────────────────────
# MAIN TRAINING FUNCTION
# ─────────────────────────────────────────────────────────────

def train():
    print("=" * 60)
    print(" RenAIssance HTR — LoRA Finetuning")
    print("=" * 60)

    # 1. Load model
    model, processor = load_base_model(quantize=True)
    model = apply_lora(model)

    # 2. Build datasets
    full_dataset = HandwrittenLineDataset(augment=True)
    if len(full_dataset) == 0:
        raise RuntimeError(
            "No training samples found. "
            "Make sure you have:\n"
            "  1. PDF files in data/raw/\n"
            "  2. Ground truth .txt files in data/ground_truth/\n"
            "Then run: python dataset.py  to convert PDFs first."
        )

    train_ds, val_ds = split_dataset(full_dataset, val_ratio=0.15)
    print(f"[train] Train: {len(train_ds)} samples | Val: {len(val_ds)} samples")

    # 3. Collator
    collator = VLMCollator(processor)

    # 4. Trainer
    history_cb = HistoryCallback()
    trainer = Trainer(
        model           = model,
        args            = get_training_args(),
        train_dataset   = train_ds,
        eval_dataset    = val_ds,
        data_collator   = collator,
        callbacks       = [history_cb],
    )

    # 5. Train
    print("[train] Starting training …")
    trainer.train()

    # 6. Save LoRA adapter
    print(f"[train] Saving LoRA adapter to {LORA_ADAPTER} …")
    model.save_pretrained(LORA_ADAPTER)
    processor.save_pretrained(LORA_ADAPTER)

    # 7. Save training history
    history_cb.save(str(CHECKPOINTS_DIR / "training_history.json"))

    print("[train] Done.")
    return trainer


if __name__ == "__main__":
    train()
