"""
transfer_adapter.py
Utility for moving the trained LoRA adapter from Kaggle to Mac M2.

On Kaggle  → run zip_adapter()    saves adapter.zip to /kaggle/working/
On Mac M2  → run verify_adapter() checks the adapter loaded correctly
"""

import shutil
import zipfile
from pathlib import Path

from config import LORA_ADAPTER, CHECKPOINTS_DIR, ROOT_DIR


# ── STEP 1: Run this on Kaggle after training ──────────────────────────────

def zip_adapter(output_path: str = "/kaggle/working/adapter.zip") -> str:
    """
    Zip the LoRA adapter folder so you can download it from Kaggle outputs.
    After running, click the output file in Kaggle → Download.
    """
    adapter_dir = Path(LORA_ADAPTER)
    if not adapter_dir.exists():
        raise FileNotFoundError(
            f"Adapter not found at {adapter_dir}. "
            "Run train.py first."
        )

    print(f"[transfer] Zipping {adapter_dir} → {output_path} …")

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in adapter_dir.rglob("*"):
            if file.is_file():
                arcname = file.relative_to(adapter_dir.parent)
                zf.write(file, arcname)
                print(f"  + {arcname}")

    size_mb = Path(output_path).stat().st_size / 1e6
    print(f"[transfer] Done. {output_path}  ({size_mb:.1f} MB)")
    print("\nNext steps:")
    print("  1. In Kaggle notebook → Output panel → Download adapter.zip")
    print("  2. On Mac: unzip adapter.zip -d /path/to/renaissance_ocr/checkpoints/")
    print("  3. Run pipeline.py — it auto-detects MPS and loads the adapter")
    return output_path


# ── STEP 2: Run this on Mac after unzipping ───────────────────────────────

def verify_adapter() -> bool:
    """
    Verify the adapter files are present and the model loads on MPS.
    Run this on Mac after downloading and unzipping the adapter.
    """
    from config import IS_MAC_M2, DEVICE
    adapter_dir = Path(LORA_ADAPTER)

    print(f"[transfer] Checking adapter at {adapter_dir} …")

    required_files = [
        "adapter_config.json",
        "adapter_model.safetensors",   
    ]
    missing = []
    for f in required_files:
        path = adapter_dir / f
        # accept either safetensors or bin
        alt  = adapter_dir / f.replace(".safetensors", ".bin")
        if not path.exists() and not alt.exists():
            missing.append(f)

    if missing:
        print(f"[transfer] Missing files: {missing}")
        print(f"[transfer] Make sure you unzipped to: {CHECKPOINTS_DIR}/")
        return False

    print(f"[transfer] Adapter files present. Device: {DEVICE}")

    # Quick load test
    print("[transfer] Loading model on MPS for verification …")
    from model import load_finetuned_model
    model, processor = load_finetuned_model()

    # Quick inference test
    from PIL import Image
    import numpy as np
    dummy_img = Image.fromarray(
        np.ones((64, 400, 3), dtype=np.uint8) * 255
    )
    from pipeline import vlm_generate
    from config import TRANSCRIBE_PROMPT
    out = vlm_generate(model, processor, dummy_img, TRANSCRIBE_PROMPT, max_new_tokens=10)
    print(f"[transfer] Test inference output: '{out}'")
    print("[transfer] Adapter verified successfully on Mac M2.")
    return True


if __name__ == "__main__":
    import sys
    from config import IS_KAGGLE

    if IS_KAGGLE:
        print("Running on Kaggle — zipping adapter …")
        zip_adapter()
    else:
        print("Running on Mac — verifying adapter …")
        verify_adapter()
