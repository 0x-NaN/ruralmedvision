"""
data_prep.py -- RuralMed Vision Dataset Preparation Pipeline

Converts HAM10000 skin lesion images + HealthCareMagic symptom dialogues
into Qwen2.5-VL chat-template JSONL format for QLoRA fine-tuning.

Usage:
    python src/data_prep.py --config configs/config.yaml
"""

import os
import json
import random
import argparse
import yaml
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split

random.seed(42)

LABEL_MAP = {
    "mel":   ("Melanoma",                 "HIGH",   "Urgent referral to dermatologist -- do not delay"),
    "bcc":   ("Basal Cell Carcinoma",     "HIGH",   "Dermatologist referral within 2 weeks"),
    "akiec": ("Actinic Keratosis",        "MEDIUM", "Dermatologist review within 1 month"),
    "bkl":   ("Benign Keratosis",         "LOW",    "Monitor; no immediate action required"),
    "df":    ("Dermatofibroma",           "LOW",    "Reassure patient; follow up if changes"),
    "nv":    ("Melanocytic Nevus (Mole)", "LOW",    "Routine monitoring; ABCDE self-check"),
    "vasc":  ("Vascular Lesion",          "MEDIUM", "Physician review recommended"),
}

RED_FLAGS = {
    "mel":   "Rapid growth, bleeding, satellite lesions, lymph node swelling",
    "bcc":   "Non-healing ulcer, perineural invasion signs, facial nerve involvement",
    "akiec": "Sudden thickening, ulceration, rapid size increase",
    "bkl":   "Sudden change in appearance, bleeding, rapid growth",
    "df":    "Rapid growth, pain, ulceration -- reconsider diagnosis",
    "nv":    "ABCDE changes: Asymmetry, Border, Colour variation, Diameter >6mm, Evolution",
    "vasc":  "Significant enlargement, ulceration, systemic symptoms",
}

SYMPTOM_TEMPLATES = {
    "mel":   [
        "Dark irregular spot growing over 3 months, uneven edges, slight itching.",
        "Black lesion with multiple colours, patient noticed recent size increase.",
        "Asymmetric mole with occasional bleeding, noticed 6 weeks ago.",
        "Rapidly changing pigmented lesion, patient reports colour variation.",
    ],
    "bcc":   [
        "Pearly bump on face, slow growing, bleeds when scratched.",
        "Shiny translucent nodule on nose, 4 months, no pain.",
        "Non-healing lesion with visible blood vessels on cheek.",
        "Waxy pinkish growth near ear, present for 6 months.",
    ],
    "akiec": [
        "Rough scaly patch on forearm, mildly red, present 2 months.",
        "Crusty lesion on sun-exposed area, mild itching.",
        "Flat scaly area on scalp, recurrent, no bleeding.",
        "Sandpaper-textured spot on back of hand, worsens in summer.",
    ],
    "bkl":   [
        "Waxy brown warty lesion, longstanding, asymptomatic.",
        "Multiple stuck-on spots on back, stable for years.",
        "Rough pigmented patch since childhood, no symptoms.",
        "Greasy brownish plaque, well-defined borders, no change.",
    ],
    "df":    [
        "Small firm nodule on leg, slightly itchy, present 1 year.",
        "Hard bump on thigh, dimples inward when pinched.",
        "Firm dermal nodule, hyperpigmented, non-tender.",
        "Raised hard spot on shin, patient recalls insect bite at site.",
    ],
    "nv":    [
        "Symmetric brown mole, stable for years, no symptoms.",
        "Flat pigmented spot, regular borders, no recent change.",
        "Common mole on back, uniform colour, no itching or bleeding.",
        "Small round tan lesion, present since teenage years, unchanged.",
    ],
    "vasc":  [
        "Bright red spot that blanches on pressure, no pain.",
        "Spider angioma on cheek, new onset, no bleeding.",
        "Port-wine coloured lesion present from birth, slowly enlarging.",
        "Multiple small red dots on trunk, patient on blood thinners.",
    ],
}

SYSTEM_PROMPT = """You are RuralMed Vision, a multimodal medical triage assistant for community health workers in rural and low-resource settings.

Analyse skin lesion images and patient symptom descriptions to produce structured triage reports.
Output valid JSON with keys: condition, severity (LOW/MEDIUM/HIGH), confidence (0.0-1.0), action, red_flags.
Be clinically conservative. When in doubt, escalate severity. Always recommend urgent referral for HIGH severity."""


def find_image(image_id: str, image_dirs: list):
    for folder in image_dirs:
        for ext in [".jpg", ".jpeg"]:
            p = Path(folder) / f"{image_id}{ext}"
            if p.exists():
                return str(p)
    return None


def build_image_sample(row, image_dirs):
    img_path = find_image(row["image_id"], image_dirs)
    if img_path is None:
        return None

    dx = row["dx"]
    condition, severity, action = LABEL_MAP[dx]
    symptom  = random.choice(SYMPTOM_TEMPLATES[dx])
    location = row.get("localization", "unspecified")

    user_text = (
        f"Patient symptoms: {symptom}\n"
        f"Body location: {location}\n"
        f"Patient age: {row.get('age', 'unknown')} | Sex: {row.get('sex', 'unknown')}\n"
        "Please assess the skin lesion shown in the image."
    )
    assistant = json.dumps({
        "condition":  condition,
        "severity":   severity,
        "confidence": 0.85,
        "action":     action,
        "red_flags":  RED_FLAGS[dx],
    }, indent=2)

    return {
        "image_path": img_path,
        "system":     SYSTEM_PROMPT,
        "user_text":  user_text,
        "assistant":  assistant,
        "dx":         dx,
        "severity":   severity,
        "modality":   "multimodal",
    }


def process_ham10000(cfg):
    meta_path  = cfg["data"]["ham10000_meta"]
    image_dirs = cfg["data"]["ham10000_images"]

    if not Path(meta_path).exists():
        print(f"HAM10000 metadata not found at {meta_path}. Skipping.")
        return []

    df = pd.read_csv(meta_path)
    print(f"HAM10000: {len(df)} rows")

    samples = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing HAM10000"):
        s = build_image_sample(row, image_dirs)
        if s:
            samples.append(s)

    print(f"HAM10000: {len(samples)} samples built ({len(df) - len(samples)} images not found)")
    return samples


def process_healthcaremagic(cfg):
    n = cfg["data"]["healthcaremagic_samples"]
    try:
        from datasets import load_dataset
        print(f"Loading HealthCareMagic-100k ({n} samples)...")
        hcm = load_dataset("lavita/ChatDoctor-HealthCareMagic-100k", split="train")
        hcm = hcm.select(range(min(n, len(hcm))))
    except Exception as e:
        print(f"HealthCareMagic load failed: {e}. Skipping.")
        return []

    samples = []
    for item in tqdm(hcm, desc="Processing HealthCareMagic"):
        samples.append({
            "image_path": None,
            "system":     SYSTEM_PROMPT,
            "user_text":  item["input"],
            "assistant":  item["output"],
            "dx":         "text_only",
            "severity":   "UNKNOWN",
            "modality":   "text",
        })

    print(f"HealthCareMagic: {len(samples)} samples")
    return samples


def split_and_save(samples, cfg):
    processed_dir = Path(cfg["data"]["train_jsonl"]).parent
    processed_dir.mkdir(parents=True, exist_ok=True)

    img_samples  = [s for s in samples if s["modality"] == "multimodal"]
    text_samples = [s for s in samples if s["modality"] == "text"]

    if len(img_samples) > 1:
        img_train, img_val = train_test_split(
            img_samples, test_size=0.1, random_state=42,
            stratify=[s["dx"] for s in img_samples]
        )
    else:
        img_train, img_val = img_samples, []

    split = int(len(text_samples) * 0.9)
    train = img_train + text_samples[:split]
    val   = img_val   + text_samples[split:]
    random.shuffle(train)

    max_train = cfg["training"]["max_train_samples"]
    max_val   = cfg["training"]["max_val_samples"]
    train = train[:max_train]
    val   = val[:max_val]

    for name, data, key in [("train", train, "train_jsonl"), ("val", val, "val_jsonl")]:
        with open(cfg["data"][key], "w") as f:
            for s in data:
                f.write(json.dumps(s) + "\n")
        print(f"Saved {len(data)} {name} samples -> {cfg['data'][key]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    print("=== RuralMed Vision -- Data Preparation ===\n")
    image_samples = process_ham10000(cfg)
    text_samples  = process_healthcaremagic(cfg)
    all_samples   = image_samples + text_samples

    if not all_samples:
        print("No samples built. Check dataset paths in config.yaml.")
        return

    split_and_save(all_samples, cfg)
    print("\nData preparation complete.")


if __name__ == "__main__":
    main()
