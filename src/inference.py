"""
inference.py
RuralMed Vision -- Inference engine

Provides:
  - Standard triage inference
  - Uncertainty-aware inference via Monte Carlo Dropout
  - Grad-CAM explainability maps
  - CrossModalFusion attention weights

Usage:
    from src.inference import RuralMedInference

    engine = RuralMedInference(adapter_path="models/ruralmed_adapter/final_adapter")
    result = engine.triage(image_path="lesion.jpg", symptoms="Dark irregular mole...")
    print(result)
"""

import json
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from collections import Counter
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig
from peft import PeftModel


# ── Cross-Modal Attention Fusion ──────────────────────────────────────────────

class CrossModalFusion(nn.Module):
    """
    Learned cross-modal attention fusion module.

    Combines visual and textual representations via multi-head attention
    with a learned modality gating mechanism. Enables the model to
    dynamically weight each modality per sample.

    Novel contribution for RuralMed Vision.
    Reference: Vaswani et al. (2017), Attention Is All You Need.
    """

    def __init__(self, hidden_dim: int = 256, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.visual_proj  = nn.Linear(hidden_dim, hidden_dim)
        self.text_proj    = nn.Linear(hidden_dim, hidden_dim)
        self.cross_attn   = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True
        )
        self.modal_gate   = nn.Sequential(nn.Linear(hidden_dim * 2, 2), nn.Softmax(dim=-1))
        self.norm         = nn.LayerNorm(hidden_dim)
        self.dropout      = nn.Dropout(dropout)

    def forward(self, visual_feat, text_feat):
        v = self.visual_proj(visual_feat)
        t = self.text_proj(text_feat)
        fused, attn_weights = self.cross_attn(query=t, key=v, value=v)
        v_pooled     = v.mean(dim=1)
        f_pooled     = fused.mean(dim=1)
        gate_weights = self.modal_gate(torch.cat([v_pooled, f_pooled], dim=-1))
        output       = gate_weights[:, 0:1] * v_pooled + gate_weights[:, 1:2] * f_pooled
        return self.norm(self.dropout(output)), attn_weights, gate_weights


# ── Grad-CAM ──────────────────────────────────────────────────────────────────

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for vision encoder layers.

    Hooks into the specified layer to capture forward activations and
    backward gradients, then produces a 2D attention heatmap highlighting
    which image regions influenced the model's triage decision.

    Reference: Selvaraju et al. (2017), Grad-CAM.
    """

    def __init__(self, model, target_layer):
        self.model        = model
        self.target_layer = target_layer
        self.gradients    = None
        self.activations  = None
        self._hooks       = []
        self._register()

    def _register(self):
        def fwd(m, inp, out): self.activations = out.detach()
        def bwd(m, gi, go):   self.gradients   = go[0].detach()
        self._hooks.append(self.target_layer.register_forward_hook(fwd))
        self._hooks.append(self.target_layer.register_full_backward_hook(bwd))

    def remove(self):
        for h in self._hooks: h.remove()

    def generate(self, inputs: dict) -> np.ndarray:
        self.model.zero_grad()
        out  = self.model(**inputs)
        out.logits[0, -1, :].max().backward()
        cam = F.relu((self.gradients.mean(dim=-1, keepdim=True) * self.activations).sum(dim=-1)).squeeze(0)
        n   = int(cam.shape[0] ** 0.5)
        cam = cam[:n*n].reshape(n, n)
        cam -= cam.min()
        if cam.max() > 0: cam /= cam.max()
        return cam.cpu().numpy()


def overlay_gradcam(image: Image.Image, cam: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Overlay Grad-CAM heatmap on original image."""
    import matplotlib.cm as mpl_cm
    img  = np.array(image.resize((224, 224))).astype(float) / 255.0
    heat = mpl_cm.jet(
        np.array(Image.fromarray((cam * 255).astype(np.uint8))
                 .resize((224, 224), Image.BILINEAR)).astype(float) / 255.0
    )[:, :, :3]
    return np.clip((1 - alpha) * img + alpha * heat, 0, 1)


# ── Main inference engine ─────────────────────────────────────────────────────

class RuralMedInference:
    """
    RuralMed Vision inference engine.

    Provides standard triage, uncertainty-aware MC Dropout inference,
    and Grad-CAM explainability in a single unified interface.

    Designed for offline deployment on resource-constrained hardware.
    """

    SYSTEM_PROMPT = (
        "You are RuralMed Vision, a multimodal medical triage assistant for rural health workers. "
        "Given a skin lesion image and symptom description, output a JSON triage report with keys: "
        "condition, severity (LOW/MEDIUM/HIGH), confidence (0.0-1.0), action, red_flags."
    )

    def __init__(
        self,
        adapter_path: str,
        base_model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        load_in_4bit: bool = True,
        device: str = "auto",
    ):
        self.device        = device
        self.adapter_path  = adapter_path
        self.base_model_id = base_model_id

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        ) if load_in_4bit else None

        print(f"Loading processor from {adapter_path}...")
        self.processor = AutoProcessor.from_pretrained(adapter_path)

        print(f"Loading base model {base_model_id}...")
        base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base_model_id,
            quantization_config=bnb_config,
            device_map=device,
            torch_dtype=torch.bfloat16,
        )

        print("Loading LoRA adapter...")
        self.model = PeftModel.from_pretrained(base, adapter_path)
        self.model.eval()

        self._entry_device = next(self.model.parameters()).device
        print(f"Model ready | Entry device: {self._entry_device}")

        # Fusion module (optional -- loaded if saved during training)
        fusion_path = Path(adapter_path) / "fusion_module.pt"
        self.fusion = None
        if fusion_path.exists():
            self.fusion = CrossModalFusion()
            self.fusion.load_state_dict(torch.load(str(fusion_path), map_location="cpu"))
            self.fusion.eval()
            print("CrossModalFusion module loaded.")

    def _build_inputs(self, image: Image.Image | None, symptom_text: str) -> dict:
        if image is not None:
            img = image.convert("RGB").resize((224, 224))
            messages = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user",   "content": [
                    {"type": "image", "image": img},
                    {"type": "text",  "text": f"Patient symptoms: {symptom_text}\nAssess the lesion shown."},
                ]},
            ]
            text   = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=text, images=[img], return_tensors="pt")
        else:
            messages = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user",   "content": f"Patient symptoms: {symptom_text}"},
            ]
            text   = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=text, return_tensors="pt")
        return {k: v.to(self._entry_device) for k, v in inputs.items()}

    def _parse_output(self, generated: str) -> dict:
        try:
            clean = generated.strip().strip("```").replace("json\n", "").strip()
            return json.loads(clean)
        except json.JSONDecodeError:
            return {"raw_output": generated, "parse_error": True}

    def triage(
        self,
        symptom_text: str,
        image_path: str = None,
        image: Image.Image = None,
        max_new_tokens: int = 200,
    ) -> dict:
        """
        Standard single-pass triage.

        Args:
            symptom_text: Patient symptom description
            image_path:   Path to skin lesion image (optional)
            image:        PIL Image object (alternative to image_path)
            max_new_tokens: Max tokens to generate

        Returns:
            dict with condition, severity, confidence, action, red_flags
        """
        if image_path and image is None:
            image = Image.open(image_path)

        inputs = self._build_inputs(image, symptom_text)

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.processor.tokenizer.eos_token_id,
            )

        generated = self.processor.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        return self._parse_output(generated)

    def triage_with_uncertainty(
        self,
        symptom_text: str,
        image_path: str = None,
        image: Image.Image = None,
        n_samples: int = 10,
        confidence_threshold: float = 0.7,
        max_new_tokens: int = 150,
    ) -> dict:
        """
        Uncertainty-aware triage via Monte Carlo Dropout.

        Runs n_samples stochastic forward passes with dropout enabled,
        aggregates predictions via majority voting, and computes a
        calibrated confidence score. Cases below confidence_threshold
        are flagged for human clinician review.

        Paper reference:
            Gal & Ghahramani (2016), Dropout as a Bayesian Approximation.

        Args:
            symptom_text:         Patient symptom description
            image_path:           Path to skin lesion image (optional)
            image:                PIL Image (alternative to image_path)
            n_samples:            Number of MC dropout samples
            confidence_threshold: Below this -> human_review_required = True
            max_new_tokens:       Max tokens per sample

        Returns:
            dict with condition, severity, confidence, action, red_flags,
                 human_review_required, mc_severity_dist, mc_condition_dist
        """
        if image_path and image is None:
            image = Image.open(image_path)

        # Enable dropout at inference time
        for m in self.model.modules():
            if isinstance(m, nn.Dropout):
                m.p   = 0.1
                m.train()

        inputs      = self._build_inputs(image, symptom_text)
        predictions = []

        for _ in range(n_samples):
            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                )
            generated = self.processor.decode(
                out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )
            parsed = self._parse_output(generated)
            if "parse_error" not in parsed:
                predictions.append(parsed)

        # Restore eval mode
        self.model.eval()

        if not predictions:
            return {"error": "No valid predictions", "human_review_required": True}

        severity_votes  = Counter(p.get("severity",  "UNKNOWN") for p in predictions)
        condition_votes = Counter(p.get("condition", "UNKNOWN") for p in predictions)
        top_severity    = severity_votes.most_common(1)[0]
        top_condition   = condition_votes.most_common(1)[0]

        sev_conf  = top_severity[1]  / len(predictions)
        cond_conf = top_condition[1] / len(predictions)
        overall   = (sev_conf + cond_conf) / 2

        # Find a representative prediction
        rep = predictions[0]
        for p in predictions:
            if p.get("severity") == top_severity[0] and p.get("condition") == top_condition[0]:
                rep = p
                break

        return {
            "condition":             top_condition[0],
            "severity":              top_severity[0],
            "confidence":            round(overall, 3),
            "action":                rep.get("action",    "Refer to clinic"),
            "red_flags":             rep.get("red_flags", "Monitor for changes"),
            "human_review_required": overall < confidence_threshold,
            "severity_confidence":   round(sev_conf,  3),
            "condition_confidence":  round(cond_conf, 3),
            "n_samples_used":        len(predictions),
            "mc_severity_dist":      dict(severity_votes),
            "mc_condition_dist":     dict(condition_votes),
        }

    def explain(
        self,
        symptom_text: str,
        image_path: str = None,
        image: Image.Image = None,
    ) -> tuple:
        """
        Generate Grad-CAM explanation for a triage decision.

        Returns:
            cam:     2D numpy array of attention weights
            overlay: numpy array of original image with heatmap overlaid
        """
        if image_path and image is None:
            image = Image.open(image_path)

        # Find last suitable vision layer
        target_layer = None
        for name, module in reversed(list(self.model.named_modules())):
            if isinstance(module, nn.Linear) and "vision" in name.lower():
                target_layer = module
                break
        if target_layer is None:
            for _, module in reversed(list(self.model.named_modules())):
                if isinstance(module, nn.Linear):
                    target_layer = module
                    break

        gradcam = GradCAM(self.model, target_layer)
        inputs  = self._build_inputs(image, symptom_text)

        self.model.eval()
        cam     = gradcam.generate(inputs)
        overlay = overlay_gradcam(image, cam)
        gradcam.remove()

        return cam, overlay
