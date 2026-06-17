import os
import json
import inspect
from pathlib import Path
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import PeftModel, LoraConfig

# --- CONFIGURATION ---
base_model_name = 'Qwen/Qwen2.5-VL-7B-Instruct'

current_dir = Path.cwd()
lora_weights_path = str((current_dir / "models" / "ruralmed_adapter").as_posix())
output_dir = str((current_dir / "models" / "ruralmed_merged_fp16").as_posix())
# ---------------------

print("Starting weight merge pipeline with strict whitelist sanitation...")
print(f"Base VLM: {base_model_name}")
print(f"LoRA Target: {lora_weights_path}")
print(f"Output Target: {output_dir}")

os.makedirs(output_dir, exist_ok=True)

# --- WHITELIST SANITIZATION ---
config_file_path = Path(lora_weights_path) / "adapter_config.json"
if config_file_path.exists():
    print(f"\nFiltering config parameters using strict LoraConfig whitelist...")
    with open(config_file_path, 'r', encoding='utf-8') as f:
        config_data = json.load(f)
    
    # Dynamically look up exactly what parameters the active PEFT library accepts
    valid_keys = set(inspect.signature(LoraConfig.__init__).parameters.keys())
    # Always include peft_type as it's required for mapping
    valid_keys.add('peft_type') 
    
    # Build a clean dictionary containing ONLY valid keys
    clean_config = {}
    dropped_keys = []
    for key, value in config_data.items():
        if key in valid_keys:
            clean_config[key] = value
        else:
            dropped_keys.append(key)
            
    if dropped_keys:
        print(f"  -> Successfully stripped all non-standard keys: {dropped_keys}")
        with open(config_file_path, 'w', encoding='utf-8') as f:
            json.dump(clean_config, f, indent=4)
        print("  -> Configuration file locked to native PEFT standard.")
    else:
        print("  -> Configuration file is already clean.")
else:
    print(f"\n⚠️ Warning: Could not find config at {config_file_path}")


print("\n1/4: Loading base model parameters into CPU RAM...")
base_model = AutoModelForVision2Seq.from_pretrained(
    base_model_name, 
    torch_dtype=torch.float16, 
    device_map='cpu'
)

print("\n2/4: Loading processing pipeline configurations...")
processor = AutoProcessor.from_pretrained(base_model_name)

print("\n3/4: Overlaying fine-tuned LoRA matrices onto base layers...")
model = PeftModel.from_pretrained(base_model, lora_weights_path)
merged_model = model.merge_and_unload()

print("\n4/4: Writing uncompressed, unified weights to local disk storage...")
merged_model.save_pretrained(output_dir)
processor.save_pretrained(output_dir)

print("\n✅ Execution Successful! Combined parameters are isolated at: ./models/ruralmed_merged_fp16")