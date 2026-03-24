"""
model.py
Loads Qwen2-VL-2B with 4-bit BitsAndBytes quantization
and wraps it with LoRA adapters for efficient finetuning and inference.
"""

import torch
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
)

from config import (
    MODEL_NAME,
    LORA_ADAPTER,
    USE_4BIT,
    BNB_COMPUTE_DTYPE,
    LORA_R,
    DEVICE,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_TARGET_MODULES,
)


# ─────────────────────────────────────────────────────────────
# QUANTIZATION CONFIG
# ─────────────────────────────────────────────────────────────

def get_bnb_config() -> BitsAndBytesConfig:
    """
    4-bit NormalFloat quantization.
    Double quantization saves extra VRAM at minimal quality cost.
    """
    compute_dtype = (
        torch.float16 if BNB_COMPUTE_DTYPE == "float16" else torch.bfloat16
    )
    return BitsAndBytesConfig(
        load_in_4bit               = True,
        bnb_4bit_compute_dtype     = compute_dtype,
        bnb_4bit_use_double_quant  = True,        
        bnb_4bit_quant_type        = "nf4",       
    )


# ─────────────────────────────────────────────────────────────
# MODEL + PROCESSOR LOADING (FOR TRAINING)
# ─────────────────────────────────────────────────────────────

def load_base_model(quantize: bool = USE_4BIT):
    """
    Load Qwen2-VL with optional 4-bit quantization for training.
    """
    print(f"[model] Loading {MODEL_NAME} …")

    processor = AutoProcessor.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )

    kwargs = dict(
        trust_remote_code = True,
        device_map        = "auto",   
        low_cpu_mem_usage = True, 
    )

    if quantize:
        kwargs["quantization_config"] = get_bnb_config()
    else:
        kwargs["torch_dtype"] = torch.float16

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_NAME, **kwargs
    )

    model.config.use_cache = False    

    print(f"[model] Base model loaded. "
          f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    return model, processor


# ─────────────────────────────────────────────────────────────
# LORA WRAPPING (FOR TRAINING)
# ─────────────────────────────────────────────────────────────

def apply_lora(model):
    """
    Wrap the quantized base model with LoRA adapters.
    """
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )

    lora_config = LoraConfig(
        r               = LORA_R,
        lora_alpha      = LORA_ALPHA,
        lora_dropout    = LORA_DROPOUT,
        target_modules  = LORA_TARGET_MODULES,
        bias            = "none",
        task_type       = "CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# ─────────────────────────────────────────────────────────────
# INFERENCE MODEL (Cloud GPU Optimized)
# ─────────────────────────────────────────────────────────────

def load_finetuned_model(adapter_path: str = LORA_ADAPTER):
    """
    Loads the base model in 4-bit and dynamically applies the LoRA weights.
    This prevents the massive RAM spike that causes Bus Errors.
    """
    print(f"[model] Loading base model for inference...")

    processor = AutoProcessor.from_pretrained(
        MODEL_NAME, trust_remote_code=True
    )

    # 1. Load Base Model in 4-bit directly to the GPU
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        quantization_config=get_bnb_config(),
        device_map="auto",
        low_cpu_mem_usage=True, # Stops the RAM crash!
        trust_remote_code=True,
    )

    print(f"[model] Attaching LoRA adapter from {adapter_path}...")
    
    # 2. Attach the adapter on the fly 
    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
    )

    model.eval()
    print(f"[model] Finetuned model ready on CUDA.")
    
    return model, processor


# ─────────────────────────────────────────────────────────────
# VRAM DIAGNOSTICS
# ─────────────────────────────────────────────────────────────

def print_vram_usage(label: str = ""):
    """Print current VRAM usage across all GPUs."""
    if not torch.cuda.is_available():
        print("[model] No CUDA devices found.")
        return
    for i in range(torch.cuda.device_count()):
        alloc   = torch.cuda.memory_allocated(i) / 1e9
        reserved = torch.cuda.memory_reserved(i) / 1e9
        total   = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"[model] {label} GPU {i}: {alloc:.1f} GB alloc / "
              f"{reserved:.1f} GB reserved / {total:.1f} GB total")


if __name__ == "__main__":
    print_vram_usage("before load")
    model, processor = load_finetuned_model()
    print_vram_usage("after load")
    print("model.py OK")