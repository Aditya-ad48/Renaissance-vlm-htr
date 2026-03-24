"""
config.py — RenAIssance HTR
Auto-detects Kaggle (CUDA) vs Mac M2 (MPS).

Paleographer notes from the actual GT files are embedded in prompts:
  - u/v interchangeable
  - f/s interchangeable
  - ç always means z
  - accents inconsistent (ignore except ñ)
  - some line-end hyphens missing (words split across lines)
"""

import os
import torch
from pathlib import Path

# ── Environment ───────────────────────────────────────────────
IS_KAGGLE = os.path.exists("/kaggle/working")
IS_MAC_M2 = (not IS_KAGGLE) and torch.backends.mps.is_available()
HAS_CUDA  = torch.cuda.is_available()

if IS_KAGGLE:
    ENV = "kaggle"
elif IS_MAC_M2:
    ENV = "mac_m2"
elif HAS_CUDA:
    ENV = "gpu"
else:
    ENV = "cpu"

if IS_KAGGLE or HAS_CUDA:
    DEVICE = "cuda"
elif IS_MAC_M2:
    DEVICE = "mps"
else:
    DEVICE = "cpu"

print(f"[config] env={ENV}  device={DEVICE}")

# ── Paths ─────────────────────────────────────────────────────
ROOT_DIR        = Path("/kaggle/working/renaissance_ocr") if IS_KAGGLE \
                  else Path(__file__).parent
DATA_DIR        = ROOT_DIR / "data"
RAW_DIR         = DATA_DIR / "raw"            
IMAGES_DIR      = DATA_DIR / "images"
CROPS_DIR       = DATA_DIR / "crops"
GT_DIR          = DATA_DIR / "ground_truth"   
RODRIGO_DIR     = DATA_DIR / "rodrigo"        
CHECKPOINTS_DIR = ROOT_DIR / "checkpoints"
OUTPUTS_DIR     = ROOT_DIR / "outputs"
LOGS_DIR        = ROOT_DIR / "logs"
LORA_ADAPTER = str(CHECKPOINTS_DIR / "qwen2vl_lora_renaissance")

for d in [RAW_DIR, IMAGES_DIR, CROPS_DIR, GT_DIR, RODRIGO_DIR,
          CHECKPOINTS_DIR, OUTPUTS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Model ─────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"

# ── Quantization ──────────────────────────────────────────────
USE_4BIT          = IS_KAGGLE or HAS_CUDA
BNB_COMPUTE_DTYPE = "float16"

# ── LoRA ──────────────────────────────────────────────────────
LORA_R              = 16
LORA_ALPHA          = 32
LORA_DROPOUT        = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# ── Training ──────────────────────────────────────────────────
EPOCHS        = 5
BATCH_SIZE    = 1
GRAD_ACCUM    = 8
LEARNING_RATE = 2e-4
WARMUP_RATIO  = 0.05
MAX_SEQ_LEN   = 512
SAVE_STEPS    = 50
EVAL_STEPS    = 50
LOGGING_STEPS = 10
FP16          = IS_KAGGLE
SEED          = 42

# ── Paleographer notes (from actual GT files) ─────────────────
PALEO_NOTES = (
    "Manuscript-specific spelling rules: "
    "(1) u and v are interchangeable — both forms are correct. "
    "(2) f and s are interchangeable — both forms are correct. "
    "(3) ç always represents modern z — transcribe accordingly. "
    "(4) Accents are inconsistent throughout — ignore all accents except ñ. "
    "(5) Some line-end hyphens are missing — words may be split across lines. "
    "(6) Preserve all archaic Spanish forms exactly as written — "
    "do NOT modernize spelling (e.g. keep 'Vezino', 'Leg.mo', 'dha', 'Vm')."
)

# ── Inference prompts ─────────────────────────────────────────
LAYOUT_PROMPT = (
    "You are analyzing a scanned page of an 18th-century Spanish "
    "handwritten legal manuscript (1744). "
    "Identify the bounding box of the main text body only. "
    "Ignore marginal annotations, page numbers, signatures, and decorative strokes. "
    "Return ONLY valid JSON: "
    "{\"main_text_bbox\": [x1, y1, x2, y2], \"has_marginalia\": bool}"
)

CROP_VALID_PROMPT = (
    "This is a cropped region from an 18th-century Spanish handwritten manuscript. "
    "Is this a clean, readable line of main body text? "
    "Answer false if it is blank, a signature, a decorative stroke, or marginalia. "
    "Reply ONLY: {\"valid\": true} or {\"valid\": false}"
)

FEW_SHOT_EXAMPLES = (
    "Example 1: [Image shows: 'Andres de Muguruza n.al y Vezino de esta Villa'] -> Transcription: 'Andres de Muguruza n.al y Vezino de esta Villa'\n"
    "Example 2: [Image shows: 'soy hijo Leg.mo de Domingo de Muguruza y Antonia'] -> Transcription: 'soy hijo Leg.mo de Domingo de Muguruza y Antonia'\n"
)

# config.py
TRANSCRIBE_PROMPT = (
    "You are an expert paleographer specializing in 18th-century Spanish manuscripts. "
    "Transcribe the text in this image exactly as it is written in the ink. "
    "STRICT RULES: "
    "1. Do not modernize spelling, grammar, or punctuation. "
    "2. Preserve all archaic abbreviations exactly (e.g., 'Leg.mo', 'Vm', 'dha', 'Vezino'). "
    "3. Maintain original capitalization, even if it is grammatically incorrect by modern standards. "
    "4. Differentiate carefully between the archaic long-s and 'f', and interchangeable letters like 'u'/'v' and 'c'/'ç'/'z'. "
    "5. Do not hallucinate or guess modern words based on context; read only the visible ink. "
    "Output ONLY the transcribed text. Do not include any explanations or markdown formatting."
)

# Note: {draft} and {notes} are filled in at runtime by pipeline.py
CORRECT_PROMPT = (
    "You are an expert paleographer specializing in 18th-century Spanish manuscripts. "
    "A draft transcription of the handwritten line shown is provided below. "
    "Look at the image carefully and correct any misread characters or words. "
    f"{PALEO_NOTES} "
    "Draft: {{draft}}\n"
    "Output ONLY the corrected transcription, nothing else."
)

CONFIDENCE_THRESHOLD = 0.75

# ── Image settings ────────────────────────────────────────────
PAGE_DPI       = 150
CROP_HEIGHT    = 64
MIN_CROP_WIDTH = 100
MAX_CROP_WIDTH = 2000

# ── Evaluation ────────────────────────────────────────────────
BERTSCORE_LANG  = "es"
BERTSCORE_MODEL = "bert-base-multilingual-cased"