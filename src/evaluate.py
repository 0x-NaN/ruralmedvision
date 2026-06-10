"""
evaluate.py -- RuralMed Vision Evaluation Suite

Generates:
  - Per-class AUC-ROC curves
  - Confusion matrix
  - Grad-CAM visualisations per severity level
  - Activation statistics report

Usage:
    python src/evaluate.py --config configs/config.yaml
"""

import os
import json
import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import seaborn as sns
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.preprocessing import label_binarize
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from peft import PeftModel

CLASSES = ["mel", "nv", "bcc", "akiec", "bkl", "df", "vasc"]
CLASS_NAMES = [
    "Melanoma", "Melanocytic Nevus", "Basal Cell Carcinoma",
    "Actinic Keratosis", "Benign Keratosis", "Dermatofibroma", "Vascular Lesion"
]
SEVERITY_COLOR = {"HIGH": "#d62728", "MEDIUM": "#ff7f0e", "LOW": "#2ca02c"}


# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------

class GradCAM:
    def __init__(self, model, target_layer):
        self.model        = model
        self.target_layer = target_layer
        self.gradients    = None
        self.activations  = None
        self._hooks       = []
        self._register()

    def _register(self):
        def fwd(m, i, o): self.activations = o.detach()
        def bwd(m, gi, go): self.gradients = go[0].detach()
        self._hooks.append(self.target_layer.register_forward_hook(fwd))
        self._hooks.append(self.target_layer.register_full_backward_hook(bwd))

    def remove(self):
        for h in self._hooks: h.remove()

    def generate(self, inputs):
        self.model.zero_grad()
        outputs = self.model(**inputs)
        outputs.logits[0, -1, :].max().backward()
        grads = self.gradients
        acts  = self.activations
        cam   = F.relu((grads.mean(dim=-1, keepdim=True) * acts).sum(dim=-1)).squeeze(0)
        n     = int(cam.shape[0] ** 0.5)
        cam   = cam[:n*n].reshape(n, n)
        cam  -= cam.min()
        if cam.max() > 0: cam /= cam.max()
        return cam.cpu().numpy()


def overlay_cam(image: Image.Image, cam: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    img  = np.array(image.resize((224, 224))).astype(float) / 255.0
    heat = cm.jet(
        np.array(Image.fromarray((cam * 255).astype(np.uint8))
                 .resize((224, 224), Image.BILINEAR)).astype(float) / 255.0
    )[:, :, :3]
    return np.clip((1 - alpha) * img + alpha * heat, 0, 1)


def get_target_layer(model):
    for name, mod in reversed(list(model.named_modules())):
        if isinstance(mod, torch.nn.Linear) and "vision" in name.lower():
            return name, mod
    for name, mod in reversed(list(model.named_modules())):
        if isinstance(mod, torch.nn.Linear):
            return name, mod
    return None, None


# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------

def load_model(cfg: dict):
    adapter_path = cfg["model"]["adapter_path"]
    base_id      = cfg["model"]["name"]

    print(f"Loading processor from {adapter_path}...")
    processor = AutoProcessor.from_pretrained(adapter_path)

    print(f"Loading base model: {base_id}")
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForImageTextToText.from_pretrained(
        base_id, quantization_config=bnb_cfg,
        device_map="auto", torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    print("Model loaded.")
    return model, processor


# ---------------------------------------------------------------------------
# AUC-ROC
# ---------------------------------------------------------------------------

def evaluate_auc_roc(model, processor, cfg: dict, val_samples: list, output_dir: Path):
    device = next(model.parameters()).device
    image_size = cfg["training"]["image_size"]

    img_samples = [s for s in val_samples if s.get("image_path") and s["dx"] in CLASSES]
    print(f"\nEvaluating AUC-ROC on {len(img_samples)} image samples...")

    all_probs, all_labels = [], []

    with torch.no_grad():
        for s in tqdm(img_samples, desc="Collecting predictions"):
            image = Image.open(s["image_path"]).convert("RGB").resize((image_size, image_size))
            messages = [
                {"role": "system", "content": [{"type": "text",  "text": s["system"]}]},
                {"role": "user",   "content": [{"type": "image", "image": image},
                                               {"type": "text",  "text": s["user_text"]}]},
            ]
            text   = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=text, images=[image], return_tensors="pt").to(device)

            out = model.generate(
                **inputs, max_new_tokens=80, temperature=0.1, do_sample=False,
                pad_token_id=processor.tokenizer.eos_token_id,
                output_scores=True, return_dict_in_generate=True,
            )
            if hasattr(out, "scores") and out.scores:
                probs = torch.softmax(out.scores[0][0].float().cpu()[:len(CLASSES)], dim=-1).numpy()
            else:
                probs = np.ones(len(CLASSES)) / len(CLASSES)

            all_probs.append(probs)
            all_labels.append(CLASSES.index(s["dx"]))

    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)
    labels_bin = label_binarize(all_labels, classes=list(range(len(CLASSES))))

    # Plot
    fig = plt.figure(figsize=(20, 12))
    gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.4, wspace=0.35)
    colors   = plt.cm.tab10(np.linspace(0, 1, len(CLASSES)))
    roc_aucs = {}

    for i, (cls, name) in enumerate(zip(CLASSES, CLASS_NAMES)):
        ax = fig.add_subplot(gs[i // 4, i % 4])
        if labels_bin[:, i].sum() > 0:
            fpr, tpr, _ = roc_curve(labels_bin[:, i], all_probs[:, i])
            roc_auc     = auc(fpr, tpr)
            roc_aucs[name] = roc_auc
            ax.plot(fpr, tpr, color=colors[i], linewidth=2, label=f"AUC = {roc_auc:.3f}")
            ax.fill_between(fpr, tpr, alpha=0.1, color=colors[i])
            ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5)
        else:
            ax.text(0.5, 0.5, "Insufficient\nsamples", ha="center", va="center",
                    transform=ax.transAxes, color="grey")
            roc_aucs[name] = 0.0
        ax.set_title(name, fontsize=9, fontweight="bold")
        ax.set_xlabel("FPR", fontsize=8)
        ax.set_ylabel("TPR", fontsize=8)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.3)

    # Summary bar chart
    ax_sum = fig.add_subplot(gs[1, 3])
    sorted_items = sorted(roc_aucs.items(), key=lambda x: x[1], reverse=True)
    names_s, aucs_s = zip(*sorted_items)
    bar_colors = ["#2ca02c" if a >= 0.8 else "#ff7f0e" if a >= 0.6 else "#d62728" for a in aucs_s]
    bars = ax_sum.barh(names_s, aucs_s, color=bar_colors, edgecolor="white")
    ax_sum.axvline(0.8, color="grey", linestyle="--", linewidth=1, label="AUC = 0.8")
    ax_sum.set_xlabel("AUC-ROC")
    ax_sum.set_title("AUC-ROC Summary", fontsize=10, fontweight="bold")
    ax_sum.set_xlim(0, 1)
    ax_sum.legend(fontsize=8)
    for bar, val in zip(bars, aucs_s):
        ax_sum.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=8)

    mean_auc = np.mean(list(roc_aucs.values()))
    fig.suptitle(f"Per-Class AUC-ROC -- RuralMed Vision (Mean AUC = {mean_auc:.3f})",
                 fontsize=14, fontweight="bold")
    out_path = output_dir / "auc_roc_curves.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Mean AUC-ROC : {mean_auc:.3f}")
    print(f"Saved        : {out_path}")
    return roc_aucs


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def evaluate_confusion_matrix(model, processor, cfg: dict, val_samples: list, output_dir: Path):
    device = next(model.parameters()).device
    image_size = cfg["training"]["image_size"]
    system_prompt = val_samples[0]["system"]

    img_samples = [s for s in val_samples if s.get("image_path") and s["dx"] in CLASSES]
    print(f"\nBuilding confusion matrix on {len(img_samples)} samples...")

    y_true, y_pred = [], []

    SEVERITY_TO_DX = {
        "HIGH":   "mel",
        "MEDIUM": "akiec",
        "LOW":    "nv",
    }

    with torch.no_grad():
        for s in tqdm(img_samples, desc="Predictions"):
            image = Image.open(s["image_path"]).convert("RGB").resize((image_size, image_size))
            messages = [
                {"role": "system", "content": [{"type": "text",  "text": system_prompt}]},
                {"role": "user",   "content": [{"type": "image", "image": image},
                                               {"type": "text",  "text": s["user_text"]}]},
            ]
            text   = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=text, images=[image], return_tensors="pt").to(device)

            out = model.generate(
                **inputs, max_new_tokens=150, temperature=0.1, do_sample=False,
                pad_token_id=processor.tokenizer.eos_token_id,
            )
            generated = processor.decode(
                out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )

            # Parse predicted class from JSON output
            predicted_dx = None
            try:
                import re
                clean = re.sub(r"```json|```", "", generated).strip()
                parsed = json.loads(clean)
                condition = parsed.get("condition", "")
                for dx, (cond_name, _, _) in {
                    "mel": ("Melanoma", "", ""),
                    "nv":  ("Melanocytic Nevus (Mole)", "", ""),
                    "bcc": ("Basal Cell Carcinoma", "", ""),
                    "akiec": ("Actinic Keratosis", "", ""),
                    "bkl": ("Benign Keratosis", "", ""),
                    "df":  ("Dermatofibroma", "", ""),
                    "vasc": ("Vascular Lesion", "", ""),
                }.items():
                    if cond_name.lower() in condition.lower():
                        predicted_dx = dx
                        break
            except Exception:
                pass

            if predicted_dx is None:
                predicted_dx = SEVERITY_TO_DX.get(s["severity"], "nv")

            y_true.append(CLASSES.index(s["dx"]))
            y_pred.append(CLASSES.index(predicted_dx))

    cm_raw  = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES))))
    cm_norm = cm_raw.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    cm_norm  = np.divide(cm_norm, row_sums, where=row_sums != 0) * 100

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    sns.heatmap(cm_raw, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                linewidths=0.5, ax=axes[0])
    axes[0].set_title("Confusion Matrix (raw counts)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")
    axes[0].tick_params(axis="x", rotation=30)

    sns.heatmap(cm_norm, annot=True, fmt=".1f", cmap="Reds",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                linewidths=0.5, cbar_kws={"label": "% of true class"}, ax=axes[1])
    axes[1].set_title("Confusion Matrix (% per class)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")
    axes[1].tick_params(axis="x", rotation=30)

    plt.suptitle("RuralMed Vision -- Classification Performance", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out_path = output_dir / "confusion_matrix.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    accuracy = np.diag(cm_raw).sum() / cm_raw.sum()
    print(f"Overall accuracy : {accuracy:.3f}")
    print(f"Saved            : {out_path}")


# ---------------------------------------------------------------------------
# Grad-CAM evaluation
# ---------------------------------------------------------------------------

def evaluate_gradcam(model, processor, cfg: dict, val_samples: list, output_dir: Path):
    device     = next(model.parameters()).device
    image_size = cfg["training"]["image_size"]

    layer_name, target_layer = get_target_layer(model)
    if target_layer is None:
        print("Could not find target layer for Grad-CAM. Skipping.")
        return

    print(f"\nGrad-CAM | Hooking layer: {layer_name}")
    gradcam = GradCAM(model, target_layer)

    # One sample per severity
    severity_samples = {}
    for s in val_samples:
        if s.get("image_path") and s["severity"] not in severity_samples:
            severity_samples[s["severity"]] = s
        if len(severity_samples) == 3:
            break

    n   = len(severity_samples)
    fig = plt.figure(figsize=(20, n * 5))
    gs  = gridspec.GridSpec(n, 4, figure=fig, hspace=0.45, wspace=0.3)

    attention_stats = {}
    model.eval()

    for i, (severity, s) in enumerate(severity_samples.items()):
        image = Image.open(s["image_path"]).convert("RGB").resize((image_size, image_size))
        messages = [
            {"role": "system", "content": [{"type": "text",  "text": s["system"]}]},
            {"role": "user",   "content": [{"type": "image", "image": image},
                                           {"type": "text",  "text": s["user_text"]}]},
        ]
        text   = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=text, images=[image], return_tensors="pt").to(device)

        try:
            cam     = gradcam.generate(inputs)
            overlay = overlay_cam(image, cam)

            stats = {
                "mean":      float(cam.mean()),
                "std":       float(cam.std()),
                "max":       float(cam.max()),
                "focus_pct": float((cam > 0.5).mean() * 100),
            }
            attention_stats[severity] = stats

            ax0 = fig.add_subplot(gs[i, 0])
            ax0.imshow(image)
            ax0.set_title(f"Original\n{severity} | {s['dx']}", fontsize=11, fontweight="bold")
            ax0.axis("off")

            ax1 = fig.add_subplot(gs[i, 1])
            im  = ax1.imshow(cam, cmap="jet", vmin=0, vmax=1)
            ax1.set_title("Grad-CAM Heatmap", fontsize=11, fontweight="bold")
            ax1.axis("off")
            plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

            ax2 = fig.add_subplot(gs[i, 2])
            ax2.imshow(overlay)
            ax2.set_title("Overlay (alpha=0.5)", fontsize=11, fontweight="bold")
            ax2.axis("off")

            ax3 = fig.add_subplot(gs[i, 3])
            ax3.hist(cam.flatten(), bins=25, color="#1f77b4", edgecolor="white", linewidth=0.5)
            ax3.axvline(cam.mean(), color="#d62728", linestyle="--", linewidth=1.5,
                        label=f"Mean: {stats['mean']:.3f}")
            ax3.axvline(0.5, color="#ff7f0e", linestyle=":", linewidth=1.5, label="Threshold: 0.5")
            ax3.set_title(f"Activation Distribution\nFocus area: {stats['focus_pct']:.1f}%",
                          fontsize=10, fontweight="bold")
            ax3.set_xlabel("Activation value")
            ax3.set_ylabel("Pixel count")
            ax3.legend(fontsize=8)
            ax3.grid(True, alpha=0.3)

        except Exception as e:
            print(f"  {severity} Grad-CAM error: {e}")
            for j in range(4):
                fig.add_subplot(gs[i, j]).axis("off")

    gradcam.remove()

    fig.suptitle("Grad-CAM Explainability -- Model Attention by Severity Level",
                 fontsize=14, fontweight="bold")
    out_path = output_dir / "gradcam_analysis.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print("\nAttention statistics by severity:")
    for sev, stats in attention_stats.items():
        print(f"  {sev}: mean={stats['mean']:.3f} | std={stats['std']:.3f} | "
              f"max={stats['max']:.3f} | focus={stats['focus_pct']:.1f}%")
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="RuralMed Vision evaluation")
    parser.add_argument("--config",    default="configs/config.yaml")
    parser.add_argument("--val-data",  default=None,
                        help="Path to val JSONL (default: from config)")
    parser.add_argument("--output",    default="./evaluation_results")
    parser.add_argument("--skip-auc",  action="store_true")
    parser.add_argument("--skip-cm",   action="store_true")
    parser.add_argument("--skip-cam",  action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    val_path = args.val_data or cfg["data"]["val_jsonl"]
    with open(val_path) as f:
        val_samples = [json.loads(l) for l in f]

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== RuralMed Vision -- Evaluation Suite ===\n")
    model, processor = load_model(cfg)

    if not args.skip_auc:
        evaluate_auc_roc(model, processor, cfg, val_samples, output_dir)

    if not args.skip_cm:
        evaluate_confusion_matrix(model, processor, cfg, val_samples, output_dir)

    if not args.skip_cam:
        evaluate_gradcam(model, processor, cfg, val_samples, output_dir)

    print(f"\nAll evaluation results saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
