"""
train.py -- RuralMed Vision QLoRA Fine-tuning

Usage:
    python src/train.py --config configs/config.yaml
    python src/train.py --config configs/config.yaml --smoke-test
"""

import os, sys, json, math, time, logging, argparse, traceback, yaml, torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset as TorchDataset
from dataclasses import dataclass
from typing import List, Dict, Any

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# ---------------------------------------------------------------------------
# Logging -- works in VSCode terminal, JupyterLab, and plain cmd
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
log = logging.getLogger("ruralmed")

def section(title):
    log.info("")
    log.info("=" * 60)
    log.info(f"  {title}")
    log.info("=" * 60)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TriageDataset(TorchDataset):
    def __init__(self, jsonl_path, processor, cfg):
        with open(jsonl_path) as f:
            self.samples = [json.loads(l) for l in f]
        self.processor  = processor
        self.max_length = cfg["training"]["max_seq_length"]
        self.image_size = cfg["training"]["image_size"]
        log.info(f"Dataset loaded: {len(self.samples)} samples from {jsonl_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        try:
            if s["image_path"] and Path(s["image_path"]).exists():
                image = Image.open(s["image_path"]).convert("RGB").resize(
                    (self.image_size, self.image_size)
                )
                messages = [
                    {"role": "system",    "content": [{"type": "text",  "text": s["system"]}]},
                    {"role": "user",      "content": [{"type": "image", "image": image},
                                                      {"type": "text",  "text": s["user_text"]}]},
                    {"role": "assistant", "content": [{"type": "text",  "text": s["assistant"]}]},
                ]
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
                enc = self.processor(
                    text=text, images=[image], return_tensors="pt",
                    max_length=self.max_length, truncation=True, padding="max_length",
                )
            else:
                messages = [
                    {"role": "system",    "content": [{"type": "text", "text": s["system"]}]},
                    {"role": "user",      "content": [{"type": "text", "text": s["user_text"]}]},
                    {"role": "assistant", "content": [{"type": "text", "text": s["assistant"]}]},
                ]
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
                enc = self.processor(
                    text=text, return_tensors="pt",
                    max_length=self.max_length, truncation=True, padding="max_length",
                )

            input_ids      = enc["input_ids"].squeeze(0)
            attention_mask = enc["attention_mask"].squeeze(0)
            pixel_values   = enc["pixel_values"].squeeze(0) if "pixel_values" in enc else None
            # image_grid_thw must stay as [N, 3] — do NOT squeeze, model iterates rows as (t,h,w)
            image_grid_thw = enc.get("image_grid_thw")  # shape: [num_images, 3]
            image_position_ids = enc.get("image_position_ids")
            if image_position_ids is not None:
                image_position_ids = image_position_ids.squeeze(0)

            labels = input_ids.clone()
            turn_token = self.processor.tokenizer.encode(
                "<|im_start|>assistant", add_special_tokens=False
            )
            seq = input_ids.tolist()
            mask_until = 0
            for i in range(len(seq) - len(turn_token)):
                if seq[i:i + len(turn_token)] == turn_token:
                    mask_until = i + len(turn_token)
            labels[:mask_until] = -100
            pad_id = self.processor.tokenizer.pad_token_id
            if pad_id is not None:
                labels[labels == pad_id] = -100

            out = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
            if pixel_values is not None:
                out["pixel_values"] = pixel_values
            if image_grid_thw is not None:
                out["image_grid_thw"] = image_grid_thw  # [num_images, 3]
            if image_position_ids is not None:
                out["image_position_ids"] = image_position_ids
            return out

        except Exception as e:
            log.warning(f"Sample {idx} failed: {e} -- using fallback text-only sample")
            dummy_text = "Patient presents with a skin condition."
            enc = self.processor(
                text=dummy_text, return_tensors="pt",
                max_length=self.max_length, truncation=True, padding="max_length",
            )
            input_ids = enc["input_ids"].squeeze(0)
            return {
                "input_ids":      input_ids,
                "attention_mask": enc["attention_mask"].squeeze(0),
                "labels":         torch.full_like(input_ids, -100),
            }


@dataclass
class TriageCollator:
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = {
            "input_ids":      torch.stack([f["input_ids"]      for f in features]).long(),
            "attention_mask": torch.stack([f["attention_mask"] for f in features]).long(),
            "labels":         torch.stack([f["labels"]         for f in features]).long(),
        }
        if "pixel_values" in features[0] and features[0]["pixel_values"] is not None:
            try:
                batch["pixel_values"] = torch.stack([f["pixel_values"] for f in features])
            except Exception:
                pass
        # image_grid_thw: each sample is [num_images, 3], cat along dim=0 to get [total_images, 3]
        if any("image_grid_thw" in f for f in features):
            try:
                grids = [f["image_grid_thw"] for f in features if "image_grid_thw" in f]
                batch["image_grid_thw"] = torch.cat(grids, dim=0)
            except Exception:
                pass
        if "image_position_ids" in features[0] and features[0]["image_position_ids"] is not None:
            try:
                batch["image_position_ids"] = torch.stack([f["image_position_ids"] for f in features])
            except Exception:
                pass
        return batch


# ---------------------------------------------------------------------------
# Training callback -- VSCode-friendly plain text output
# ---------------------------------------------------------------------------

class RuralMedCallback:
    """
    Plain logging callback -- works in any terminal.
    Prints a line every N steps with loss, ppl, lr, vram.
    Also shows a simple ASCII progress bar.
    """
    def __init__(self, log_every: int = 10):
        self.log_every        = log_every
        self._logs            = {}
        self._t_start         = None
        self._t_step          = None
        self._total_steps     = None
        self._steps_per_epoch = None
        self._current_epoch   = 0
        self.history          = {"step": [], "train_loss": [], "eval_loss": [], "eval_ppl": []}

    # HF Trainer calls these as on_*
    def on_train_begin(self, args, state, control, **kw):
        self._total_steps     = state.max_steps
        self._steps_per_epoch = math.ceil(state.max_steps / int(args.num_train_epochs))
        self._t_start         = time.perf_counter()
        log.info(f"Training started")
        log.info(f"  Total steps    : {self._total_steps}")
        log.info(f"  Steps/epoch    : {self._steps_per_epoch}")
        log.info(f"  Epochs         : {int(args.num_train_epochs)}")
        log.info(f"  Effective batch: {args.per_device_train_batch_size * args.gradient_accumulation_steps}")

    def on_train_end(self, args, state, control, **kw):
        elapsed = time.perf_counter() - self._t_start
        log.info("")
        log.info(f"Training complete in {elapsed/60:.1f} min | Best eval loss: {state.best_metric}")

    def on_step_begin(self, args, state, control, **kw):
        self._t_step = time.perf_counter()

    def on_step_end(self, args, state, control, **kw):
        step = state.global_step
        if step % self.log_every != 0:
            return

        # ASCII progress bar
        pct   = step / max(self._total_steps, 1)
        width = 30
        filled = int(width * pct)
        bar   = "[" + "#" * filled + "-" * (width - filled) + "]"

        # Time estimate
        elapsed = time.perf_counter() - self._t_start
        eta_s   = (elapsed / max(step, 1)) * (self._total_steps - step)
        eta_str = f"{eta_s/60:.1f}min" if eta_s > 60 else f"{eta_s:.0f}s"

        # Step speed
        dt      = time.perf_counter() - self._t_step if self._t_step else 1
        its_str = f"{1/dt:.2f} it/s"

        # Loss and ppl
        loss_str = ""
        if "loss" in self._logs:
            loss = self._logs["loss"]
            try:    ppl = round(math.exp(loss), 2)
            except: ppl = float("inf")
            loss_str = f" | loss={loss:.4f} ppl={ppl}"

        # LR
        lr_str = ""
        if "learning_rate" in self._logs:
            lr_str = f" | lr={self._logs['learning_rate']:.2e}"

        # VRAM
        vram_str = ""
        if torch.cuda.is_available():
            parts = [f"GPU{i}:{torch.cuda.memory_allocated(i)/1e9:.1f}GB"
                     for i in range(torch.cuda.device_count())]
            vram_str = f" | VRAM={' '.join(parts)}"

        epoch_str = f"E{self._current_epoch+1}"
        log.info(f"  {epoch_str} {bar} {step}/{self._total_steps} ({pct*100:.0f}%) "
                 f"ETA={eta_str} {its_str}{loss_str}{lr_str}{vram_str}")

    def on_log(self, args, state, control, logs=None, **kw):
        if not logs:
            return
        self._logs.update(logs)
        if "loss" in logs and state.global_step > 0:
            self.history["step"].append(state.global_step)
            self.history["train_loss"].append(logs["loss"])

    def on_evaluate(self, args, state, control, metrics=None, **kw):
        if not metrics or "eval_loss" not in metrics:
            return
        el = metrics["eval_loss"]
        try:    ppl = round(math.exp(el), 2)
        except: ppl = float("inf")
        self.history["eval_loss"].append(el)
        self.history["eval_ppl"].append(ppl)
        vram_str = ""
        if torch.cuda.is_available():
            parts = [f"GPU{i}:{torch.cuda.memory_allocated(i)/1e9:.1f}GB"
                     for i in range(torch.cuda.device_count())]
            vram_str = " | VRAM=" + " ".join(parts)
        log.info(f"  >> EVAL step={state.global_step} | loss={el:.4f} | ppl={ppl}{vram_str}")

        # Detect epoch boundary
        if self._steps_per_epoch and state.global_step % self._steps_per_epoch < 5:
            self._current_epoch += 1
            log.info(f"  >> Epoch {self._current_epoch} complete")

    def save_history(self, path):
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
        log.info(f"Training history saved -> {path}")

    # Make it compatible with HF Trainer callback protocol
    def __call__(self, *args, **kwargs):
        pass


# HF Trainer needs a TrainerCallback subclass
from transformers import TrainerCallback

class RuralMedTrainerCallback(TrainerCallback):
    def __init__(self, cb: RuralMedCallback):
        self.cb = cb

    def on_train_begin(self, args, state, control, **kw):
        self.cb.on_train_begin(args, state, control, **kw)

    def on_train_end(self, args, state, control, **kw):
        self.cb.on_train_end(args, state, control, **kw)

    def on_step_begin(self, args, state, control, **kw):
        self.cb.on_step_begin(args, state, control, **kw)

    def on_step_end(self, args, state, control, **kw):
        self.cb.on_step_end(args, state, control, **kw)

    def on_log(self, args, state, control, logs=None, **kw):
        self.cb.on_log(args, state, control, logs=logs, **kw)

    def on_evaluate(self, args, state, control, metrics=None, **kw):
        self.cb.on_evaluate(args, state, control, metrics=metrics, **kw)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="configs/config.yaml")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    # Load config
    log.info(f"Loading config from {args.config}")
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.smoke_test:
        cfg["training"]["num_epochs"]        = 1
        cfg["training"]["max_train_samples"] = 50
        cfg["training"]["max_val_samples"]   = 10
        cfg["training"]["logging_steps"]     = 5
        cfg["training"]["eval_steps"]        = 25
        cfg["training"]["save_steps"]        = 50
        log.info("SMOKE TEST MODE: 50 samples, 1 epoch")

    section("SYSTEM")
    log.info(f"Python         : {sys.version.split()[0]}")
    log.info(f"PyTorch        : {torch.__version__}")
    log.info(f"CUDA available : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            log.info(f"  GPU {i}: {p.name} | {p.total_memory/1e9:.1f} GB VRAM")
    else:
        log.warning("No GPU detected -- training will be very slow on CPU")

    section("LOADING PROCESSOR")
    log.info(f"Model: {cfg['model']['name']}")
    log.info("Downloading processor (first run: ~1 min)...")
    from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
    processor = AutoProcessor.from_pretrained(cfg["model"]["name"])
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    log.info("Processor ready")

    section("LOADING MODEL (4-bit QLoRA)")
    log.info("Downloading model weights (first run: ~15GB, takes several minutes)...")
    log.info("You will see HuggingFace download progress bars below:")
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=cfg["quantization"]["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=cfg["quantization"]["bnb_4bit_use_double_quant"],
    )
    model = AutoModelForImageTextToText.from_pretrained(
        cfg["model"]["name"],
        quantization_config=bnb_cfg,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    model.enable_input_require_grads()
    total_params = sum(p.numel() for p in model.parameters()) / 1e9
    log.info(f"Model loaded | {total_params:.2f}B params")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            used  = torch.cuda.memory_allocated(i) / 1e9
            total = torch.cuda.get_device_properties(i).total_memory / 1e9
            log.info(f"  GPU {i}: {used:.1f} / {total:.1f} GB used")

    section("APPLYING LoRA")
    from peft import LoraConfig, get_peft_model, TaskType
    lang_layers = [
        name for name, mod in model.named_modules()
        if mod.__class__.__name__ == "Linear"
        and not any(x in name for x in ["vision_tower", "visual", "patch_embed"])
    ]
    log.info(f"Found {len(lang_layers)} targetable language model layers")
    lora_cfg = LoraConfig(
        r=cfg["qlora"]["r"],
        lora_alpha=cfg["qlora"]["lora_alpha"],
        lora_dropout=cfg["qlora"]["lora_dropout"],
        bias=cfg["qlora"]["bias"],
        task_type=TaskType.CAUSAL_LM,
        target_modules=lang_layers,
    )
    model = get_peft_model(model, lora_cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    log.info(f"Trainable params: {trainable/1e6:.1f}M / {total/1e6:.0f}M ({100*trainable/total:.2f}%)")

    section("LOADING DATASETS")
    train_ds = TriageDataset(cfg["data"]["train_jsonl"], processor, cfg)
    val_ds   = TriageDataset(cfg["data"]["val_jsonl"],   processor, cfg)

    if args.smoke_test:
        from torch.utils.data import Subset
        train_ds = Subset(train_ds, range(min(50,  len(train_ds))))
        val_ds   = Subset(val_ds,   range(min(10,  len(val_ds))))
        log.info(f"Smoke test: using {len(train_ds)} train / {len(val_ds)} val samples")

    eff_batch = cfg["training"]["batch_size"] * cfg["training"]["gradient_accumulation_steps"]
    steps_per_epoch = math.ceil(len(train_ds) / eff_batch)
    total_steps     = steps_per_epoch * cfg["training"]["num_epochs"]
    log.info(f"Effective batch size : {eff_batch}")
    log.info(f"Steps per epoch      : {steps_per_epoch}")
    log.info(f"Total steps          : {total_steps}")
    log.info(f"Estimated time       : ~{total_steps * 2 / 60:.0f} min on RTX 4060")

    section("TRAINING")
    from transformers import TrainingArguments, Trainer
    t_args = TrainingArguments(
        output_dir=cfg["training"]["output_dir"],
        num_train_epochs=cfg["training"]["num_epochs"],
        per_device_train_batch_size=cfg["training"]["batch_size"],
        per_device_eval_batch_size=cfg["training"]["batch_size"],
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
        gradient_checkpointing=cfg["training"]["gradient_checkpointing"],
        learning_rate=cfg["training"]["learning_rate"],
        lr_scheduler_type=cfg["training"]["lr_scheduler"],
        warmup_steps=cfg["training"]["warmup_steps"],
        weight_decay=cfg["training"]["weight_decay"],
        bf16=cfg["training"]["bf16"],
        fp16=cfg["training"]["fp16"],
        logging_steps=cfg["training"]["logging_steps"],
        eval_strategy="steps",
        eval_steps=cfg["training"]["eval_steps"],
        save_steps=cfg["training"]["save_steps"],
        save_total_limit=cfg["training"]["save_total_limit"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        remove_unused_columns=False,
        torch_compile=False,
        disable_tqdm=False,     # keep HF default bars too
    )

    cb = RuralMedCallback(log_every=cfg["training"]["logging_steps"])
    trainer = Trainer(
        model=model, args=t_args,
        train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=TriageCollator(),
        callbacks=[RuralMedTrainerCallback(cb)],
    )

    log.info("Starting training -- progress printed every logging_steps steps")
    try:
        trainer.train()
    except Exception as e:
        log.error(f"Training failed: {e}")
        log.error(traceback.format_exc())
        sys.exit(1)

    section("SAVING")
    adapter_path = cfg["model"]["adapter_path"]
    Path(adapter_path).mkdir(parents=True, exist_ok=True)
    log.info(f"Saving adapter -> {adapter_path}")
    model.save_pretrained(adapter_path)
    processor.save_pretrained(adapter_path)
    cb.save_history(str(Path(adapter_path) / "training_history.json"))
    log.info("Done.")


if __name__ == "__main__":
    main()