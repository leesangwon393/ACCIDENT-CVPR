"""RB-FT stage-1 rationale fine-tuning for Qwen3-VL.

Training contract:
- input prompt: the canonical prompt from `1. Making Rationle.py` with only the
  `Ground Truth Label: {label}` line removed
- target: the rationale text generated offline by `1. Making Rationle.py`

Runtime assumptions are the same Qwen3-VL environment used by the generation
script: `transformers` with `Qwen3VLForConditionalGeneration`,
`qwen_vl_utils.process_vision_info`, `peft`, and optional `bitsandbytes`.

Dataset and video paths are intentionally provided at runtime instead of being
hardcoded into the script.
"""

import argparse
import gc
import importlib.util
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

torch = __import__("torch")
qwen_vl_utils = __import__("qwen_vl_utils")
process_vision_info = getattr(qwen_vl_utils, "process_vision_info")
lr_scheduler_mod = __import__("torch.optim.lr_scheduler", fromlist=["CosineAnnealingLR", "LambdaLR", "SequentialLR"])
CosineAnnealingLR = getattr(lr_scheduler_mod, "CosineAnnealingLR")
LambdaLR = getattr(lr_scheduler_mod, "LambdaLR")
SequentialLR = getattr(lr_scheduler_mod, "SequentialLR")
Optimizer = getattr(__import__("torch.optim.optimizer", fromlist=["Optimizer"]), "Optimizer")
AdamW = getattr(__import__("torch.optim.adamw", fromlist=["AdamW"]), "AdamW")
data_mod = __import__("torch.utils.data", fromlist=["DataLoader", "Dataset"])
DataLoader = getattr(data_mod, "DataLoader")
Dataset = getattr(data_mod, "Dataset")
tqdm = getattr(__import__("tqdm.auto", fromlist=["tqdm"]), "tqdm")


def load_module(module_path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SCRIPT_DIR = Path(__file__).resolve().parent
baseline = load_module(SCRIPT_DIR / "0. baseline.py", "baseline_local_stage1")
stage1_spec = load_module(SCRIPT_DIR / "1. Making Rationle.py", "stage1_prompt_spec")
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs" / "stage1_rationale_lora"

peft = __import__("peft")
transformers = __import__("transformers")
AutoProcessor = getattr(__import__("transformers", fromlist=["AutoProcessor"]), "AutoProcessor")
BitsAndBytesConfig = getattr(__import__("transformers", fromlist=["BitsAndBytesConfig"]), "BitsAndBytesConfig")
pd = __import__("pandas")

Qwen3VLForConditionalGeneration = getattr(transformers, "Qwen3VLForConditionalGeneration", None)
LoraConfig = getattr(peft, "LoraConfig")
get_peft_model = getattr(peft, "get_peft_model")
prepare_model_for_kbit_training = getattr(peft, "prepare_model_for_kbit_training")


STAGE1_TRAIN_USER_PROMPT = """Role: Traffic Accident Analysis Expert. 
Task: Based on the video content, analyze the traffic scene and explain the accident-related event or abnormal driving situation shown in the video.
Keep your total response concise, under 400 tokens. Do NOT repeat the same observation. Each fact should appear only once.
Structure your reasoning across the following four dimensions:
Subjects: Identify the main entities involved in the scene, such as vehicles, guardrails, lane markings, traffic lights, road boundaries, and other relevant objects.
Attributes: Describe the important characteristics of these entities, including relative positions, lane placement, estimated speed changes, heading direction, vehicle type/color, road geometry, visibility, lighting, and weather or surface conditions.
Actions: Detail the key motions and interactions between entities. Focus on behaviors related to accident occurrence or accident precursors, such as sudden braking, rapid lane departure, failure to yield, unsafe turning, rear-end approach, side impact, collision, near-collision, rollover, or loss of control.
Scenes: Describe the overall traffic environment, such as intersection, highway, merge area, curved road, narrow street, crosswalk, roadside area, traffic density, and surrounding visual conditions.
Output: Provide a detailed rationale explaining why these elements indicate a traffic accident or accident-related event requiring attention.
"""


class RowDataset(Dataset):
    def __init__(self, rows: List[Dict[str, Any]]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.rows[idx]


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "t", "1", "yes", "y"}:
        return True
    if normalized in {"false", "f", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage-1 rationale fine-tuning for Qwen3-VL")
    parser.add_argument("--model-name", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--rationale-jsonl", required=True, help="JSONL file produced by 1. Making Rationle.py")
    parser.add_argument("--video-base-path", required=True, help="Base directory that contains training videos")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--language-learning-rate", type=float, default=1e-5)
    parser.add_argument("--vision-learning-rate", type=float, default=2e-6)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--eta-min", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=0, help="0 means run full epoch schedule")
    parser.add_argument("--max-train-samples", type=int, default=0, help="0 means use all rows")
    parser.add_argument("--train-max-frames", type=int, default=32)
    parser.add_argument("--train-max-side", type=int, default=480)
    parser.add_argument("--train-mini-batch", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--save-every", type=int, default=0, help="0 disables intermediate saves")
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--finetune-vision-layers", type=parse_bool, default=True)
    parser.add_argument("--finetune-language-layers", type=parse_bool, default=True)
    parser.add_argument("--finetune-attention-modules", type=parse_bool, default=True)
    parser.add_argument("--finetune-mlp-modules", type=parse_bool, default=False)
    parser.add_argument("--finetune-merger-modules", type=parse_bool, default=True)
    parser.add_argument("--base-precision", choices=["4bit", "8bit", "bf16"], default="4bit")
    parser.add_argument("--wandb", type=parse_bool, default=False)
    parser.add_argument("--wandb-project", default="rb-ft")
    parser.add_argument("--wandb-run-name", default="")
    parser.add_argument("--wandb-mode", choices=["online", "offline"], default="online")
    return parser.parse_args()


def is_main_process() -> bool:
    return True


def log_info(message: str) -> None:
    if is_main_process():
        print(message)


def maybe_init_wandb(args: argparse.Namespace, output_dir: Path, stage_name: str) -> Any:
    if not args.wandb or not is_main_process():
        return None
    try:
        wandb = __import__("wandb")
    except ImportError as exc:
        raise RuntimeError("wandb is not installed but --wandb was enabled.") from exc

    run_name = args.wandb_run_name or f"{stage_name}-{output_dir.name}"
    config = vars(args).copy()
    config["stage_name"] = stage_name
    run = wandb.init(
        project=args.wandb_project,
        name=run_name,
        mode=args.wandb_mode,
        config=config,
        dir=str(output_dir),
    )
    return run


def build_warmup_cosine_scheduler(
    optimizer: Any,
    total_optimizer_steps: int,
    warmup_ratio: float,
    eta_min: float,
) -> Any:
    warmup_steps = min(total_optimizer_steps - 1, int(total_optimizer_steps * warmup_ratio)) if total_optimizer_steps > 1 else 0
    cosine_kwargs: Dict[str, Any] = {"eta_min": eta_min}
    if warmup_steps <= 0:
        return CosineAnnealingLR(optimizer, T_max=max(1, total_optimizer_steps), **cosine_kwargs)

    warmup_scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda current_step: float(current_step + 1) / float(max(1, warmup_steps)),
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(1, total_optimizer_steps - warmup_steps),
        **cosine_kwargs,
    )
    return SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps],
    )


def initialize_runtime() -> Any:
    if torch.cuda.is_available():
        return torch.device("cuda", 0)
    return torch.device("cpu")


def load_rationale_rows(args: argparse.Namespace) -> Any:
    rows: List[Dict[str, Any]] = []
    with Path(args.rationale_jsonl).open("r", encoding="utf-8") as reader:
        for line in reader:
            stripped = line.strip()
            if not stripped:
                continue
            rows.append(cast(Dict[str, Any], json.loads(stripped)))

    if args.max_train_samples > 0:
        rows = rows[: args.max_train_samples]

    if not rows:
        raise ValueError("No training rows loaded from rationale JSONL.")

    required = {"video_rel_path", "rationale"}
    for idx, row in enumerate(rows, start=1):
        missing = required.difference(row.keys())
        if missing:
            raise ValueError(f"Row {idx} is missing required keys: {sorted(missing)}")

    return pd.DataFrame(rows)


def select_lora_target_modules(model: Any, args: argparse.Namespace) -> List[str]:
    target_modules: List[str] = []

    for module_name, _ in model.named_modules():
        if args.finetune_language_layers:
            if args.finetune_attention_modules and module_name.startswith("model.language_model.layers."):
                if module_name.endswith((".self_attn.q_proj", ".self_attn.k_proj", ".self_attn.v_proj", ".self_attn.o_proj")):
                    target_modules.append(module_name)
            if args.finetune_mlp_modules and module_name.startswith("model.language_model.layers."):
                if module_name.endswith((".mlp.gate_proj", ".mlp.up_proj", ".mlp.down_proj")):
                    target_modules.append(module_name)

        if args.finetune_vision_layers:
            if args.finetune_attention_modules and module_name.startswith("model.visual.blocks."):
                parts = module_name.split(".")
                block_index = int(parts[3]) if len(parts) > 3 and parts[3].isdigit() else -1
                if 20 <= block_index <= 26 and module_name.endswith((".attn.qkv", ".attn.proj")):
                    target_modules.append(module_name)
                if 20 <= block_index <= 26 and args.finetune_mlp_modules and module_name.endswith((".mlp.linear_fc1", ".mlp.linear_fc2")):
                    target_modules.append(module_name)
            if args.finetune_merger_modules:
                if module_name.startswith("model.visual.merger.") and module_name.endswith((".linear_fc1", ".linear_fc2")):
                    target_modules.append(module_name)
                if module_name.startswith("model.visual.deepstack_merger_list.") and module_name.endswith((".linear_fc1", ".linear_fc2")):
                    target_modules.append(module_name)

    deduped = sorted(set(target_modules))
    if not deduped:
        raise ValueError("No LoRA target modules selected. Enable at least one module group.")
    return deduped


def build_optimizer_param_groups(model: Any, args: argparse.Namespace) -> List[Dict[str, Any]]:
    language_params: List[Any] = []
    vision_merger_params: List[Any] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(token in name for token in ("visual", "merger", "deepstack_merger_list")):
            vision_merger_params.append(param)
        else:
            language_params.append(param)

    param_groups: List[Dict[str, Any]] = []
    if language_params:
        param_groups.append(
            {
                "params": language_params,
                "lr": args.language_learning_rate,
                "weight_decay": args.weight_decay,
                "group_name": "language",
            }
        )
    if vision_merger_params:
        param_groups.append(
            {
                "params": vision_merger_params,
                "lr": args.vision_learning_rate,
                "weight_decay": args.weight_decay,
                "group_name": "vision_merger",
            }
        )

    if not param_groups:
        raise ValueError("No trainable parameters available for AdamW optimizer.")
    return param_groups


def build_model_and_processor(args: argparse.Namespace, runtime_device: Any) -> "tuple[Any, Any]":
    if Qwen3VLForConditionalGeneration is None:
        raise RuntimeError("Qwen3VLForConditionalGeneration is unavailable in the installed transformers version.")

    quant_config: Optional[Any] = None
    load_kwargs: Dict[str, Any] = {
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
    }

    if args.base_precision == "4bit":
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_storage=torch.bfloat16,
        )
        load_kwargs["quantization_config"] = quant_config
        load_kwargs["attn_implementation"] = "flash_attention_2"
    elif args.base_precision == "8bit":
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
        load_kwargs["quantization_config"] = quant_config
        load_kwargs["attn_implementation"] = "flash_attention_2"
    else:
        load_kwargs["attn_implementation"] = "sdpa"

    if runtime_device.type == "cuda":
        load_kwargs["device_map"] = "auto"

    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
    processor.tokenizer.padding_side = "left"

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_name,
        **load_kwargs,
    )
    model.config.use_cache = False
    if getattr(model.config, "text_config", None) is not None:
        model.config.text_config.use_cache = False
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.use_cache = False

    if quant_config is not None:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": True},
        )
    else:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
        model.enable_input_require_grads()

    if not args.finetune_vision_layers and hasattr(model, "model") and hasattr(model.model, "visual"):
        model.model.visual.gradient_checkpointing_disable()

    target_modules = select_lora_target_modules(model, args)
    log_info(f"[INFO] selected {len(target_modules)} LoRA target modules")

    lora_config = LoraConfig(
        r=args.r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.train()
    if is_main_process():
        model.print_trainable_parameters()
    return model, processor


def clean_rationale_text(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def build_training_messages(row: Dict[str, Any], args: argparse.Namespace) -> List[Dict[str, Any]]:
    video_rel_path = str(row["video_rel_path"])
    video_path = str(Path(args.video_base_path) / video_rel_path)

    duration = baseline.to_float_or(row.get("duration"), 10.0)
    no_frames = baseline.to_int_or(row.get("no_frames"), 0)
    height = baseline.to_int_or(row.get("height"), 720)
    width = baseline.to_int_or(row.get("width"), 1280)
    fps = baseline.compute_video_fps(duration, no_frames, height, width)

    frames_with_ts = baseline.sample_frames_with_timestamps(
        video_path,
        fps,
        duration,
        max_frames=args.train_max_frames,
        max_side=args.train_max_side,
    )

    content: List[Dict[str, Any]] = []
    for frame_image, ts in frames_with_ts:
        content.append({"type": "text", "text": f"[t={ts}s]"})
        content.append(
            {
                "type": "image",
                "image": frame_image,
                "max_pixels": baseline.MAX_PIXELS,
                "min_pixels": baseline.MIN_PIXELS,
            }
        )
    content.append({"type": "text", "text": STAGE1_TRAIN_USER_PROMPT})

    rationale_target = clean_rationale_text(str(row["rationale"]))
    return [
        {"role": "user", "content": content},
        {"role": "assistant", "content": rationale_target},
    ]


def build_training_features(processor: Any, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    prompt_messages = messages[:-1]

    full_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    prompt_text = processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)

    image_inputs, video_inputs = process_vision_info(prompt_messages)

    processor_kwargs: Dict[str, Any] = {
        "padding": True,
        "return_tensors": "pt",
    }
    if image_inputs is not None:
        processor_kwargs["images"] = image_inputs
    if video_inputs is not None:
        processor_kwargs["videos"] = video_inputs

    full_inputs = processor(text=[full_text], **processor_kwargs)
    prompt_inputs = processor(text=[prompt_text], **processor_kwargs)

    prompt_len = prompt_inputs["input_ids"].shape[1]
    labels = full_inputs["input_ids"].clone()
    labels[:, :prompt_len] = -100
    labels[full_inputs["attention_mask"] == 0] = -100

    features: Dict[str, Any] = {}
    for key, value in full_inputs.items():
        if isinstance(value, torch.Tensor):
            features[key] = value
    features["labels"] = labels
    return features


def move_to_device(batch: Dict[str, Any], model: Any) -> Dict[str, Any]:
    device = next(model.parameters()).device
    return {key: value.to(device) for key, value in batch.items()}


def save_training_artifacts(model: Any, processor: Any, save_dir: Path) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_dir)
    processor.save_pretrained(save_dir)


def is_cuda_oom_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "cuda error: out of memory" in text


def clear_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def run_training(args: argparse.Namespace) -> None:
    runtime_device = initialize_runtime()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    wandb_run = maybe_init_wandb(args, output_dir, "stage1")

    log_info(f"[INFO] model={args.model_name}")
    log_info(f"[INFO] rationale_jsonl={args.rationale_jsonl}")
    log_info(f"[INFO] output_dir={output_dir}")
    log_info(f"[INFO] base_precision={args.base_precision}")
    log_info("[INFO] training_objective=causal_lm_rationale_teacher_forcing")
    log_info(f"[INFO] train_mini_batch={args.train_mini_batch}")
    log_info(f"[INFO] gradient_accumulation_steps={args.gradient_accumulation_steps}")
    log_info(f"[INFO] language_learning_rate={args.language_learning_rate}")
    log_info(f"[INFO] vision_learning_rate={args.vision_learning_rate}")
    log_info(f"[INFO] weight_decay={args.weight_decay}")
    log_info(f"[INFO] warmup_ratio={args.warmup_ratio}")
    log_info(f"[INFO] num_epochs={args.num_epochs}")
    log_info(f"[INFO] finetune_attention_modules={args.finetune_attention_modules}")
    log_info(f"[INFO] finetune_mlp_modules={args.finetune_mlp_modules}")
    log_info(f"[INFO] finetune_merger_modules={args.finetune_merger_modules}")

    labels_df = load_rationale_rows(args)
    model, processor = build_model_and_processor(args, runtime_device)

    dataset = RowDataset(labels_df.to_dict("records"))
    dataloader = DataLoader(
        dataset,
        batch_size=args.train_mini_batch,
        shuffle=True,
        sampler=None,
        collate_fn=lambda batch: batch,
    )

    optimizer_param_groups = build_optimizer_param_groups(model, args)
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = AdamW(optimizer_param_groups)

    total_batch_steps = len(dataloader) * args.num_epochs
    if args.max_steps > 0:
        total_batch_steps = min(total_batch_steps, args.max_steps)
    total_optimizer_steps = max(1, math.ceil(total_batch_steps / args.gradient_accumulation_steps))
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        total_optimizer_steps=total_optimizer_steps,
        warmup_ratio=args.warmup_ratio,
        eta_min=args.eta_min,
    )
    log_info(f"[INFO] train_rows={len(dataset)}")
    log_info(f"[INFO] total_batch_steps={total_batch_steps}")
    log_info(f"[INFO] total_optimizer_steps={total_optimizer_steps}")
    for group in optimizer.param_groups:
        log_info(
            f"[INFO] optimizer_group={group.get('group_name', 'unknown')} "
            f"lr={group['lr']:.8f} weight_decay={group['weight_decay']:.4f} params={len(group['params'])}"
        )

    train_model = model
    optimizer.zero_grad(set_to_none=True)

    step = 0
    optimizer_step = 0
    progress_bar = tqdm(total=total_batch_steps, desc="FT-stage1", unit="batch") if is_main_process() else None

    try:
        try:
            for epoch in range(args.num_epochs):
                if is_main_process():
                    tqdm.write(f"[INFO] starting epoch {epoch + 1}/{args.num_epochs}")
                for mini_batch_rows in dataloader:
                    if args.max_steps > 0 and step >= args.max_steps:
                        break

                    batch_rows = cast(List[Dict[str, Any]], mini_batch_rows)
                    batch_loss_total = 0.0

                    for row in batch_rows:
                        messages = build_training_messages(cast(Dict[str, Any], row), args)
                        batch = build_training_features(processor, messages)
                        batch = move_to_device(batch, train_model)

                        outputs = train_model(**batch)
                        raw_loss = outputs.loss
                        loss = raw_loss / float(len(batch_rows) * args.gradient_accumulation_steps)
                        loss.backward()
                        batch_loss_total += raw_loss.detach().item()

                        del loss
                        del raw_loss
                        del outputs
                        del batch
                        del messages

                    avg_batch_loss = batch_loss_total / max(1, len(batch_rows))
                    step += 1
                    if step % args.gradient_accumulation_steps == 0:
                        if args.grad_clip > 0:
                            torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad(set_to_none=True)
                        optimizer_step += 1

                    if progress_bar is not None:
                        progress_bar.update(1)
                        progress_bar.set_postfix(loss=f"{avg_batch_loss:.4f}")

                    if wandb_run is not None:
                        wandb_run.log(
                            {
                                "train/loss": avg_batch_loss,
                                "train/batch_step": step,
                                "train/optimizer_step": optimizer_step,
                                "train/language_lr": optimizer.param_groups[0]["lr"],
                                "train/vision_lr": optimizer.param_groups[1]["lr"] if len(optimizer.param_groups) > 1 else 0.0,
                                "train/epoch": epoch + 1,
                            },
                            step=step,
                        )

                    if step % args.log_every == 0 and is_main_process():
                        tqdm.write(
                            f"[TRAIN] batch_step={step}/{total_batch_steps} "
                            f"optimizer_step={optimizer_step}/{total_optimizer_steps} "
                            f"lang_lr={optimizer.param_groups[0]['lr']:.8f} "
                            f"loss={avg_batch_loss:.4f}"
                        )

                    if args.save_every > 0 and optimizer_step > 0 and optimizer_step % args.save_every == 0 and step % args.gradient_accumulation_steps == 0:
                        save_dir = output_dir / f"step_{optimizer_step}"
                        save_training_artifacts(train_model, processor, save_dir)
                        log_info(f"[INFO] checkpoint saved -> {save_dir}")

                clear_cuda_memory()
                log_info(f"[INFO] cleared CUDA memory after epoch {epoch + 1}/{args.num_epochs}")

                if args.max_steps > 0 and step >= args.max_steps:
                    break

            pending_micro_steps = step % args.gradient_accumulation_steps
            if pending_micro_steps != 0:
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1
                log_info(f"[INFO] flushed final partial accumulation window ({pending_micro_steps} micro_steps)")
        finally:
            if progress_bar is not None:
                progress_bar.close()

        final_dir = output_dir / "final"
        save_training_artifacts(train_model, processor, final_dir)
        log_info(f"[INFO] final adapter saved -> {final_dir}")
        if wandb_run is not None:
            wandb_run.summary["final_adapter_dir"] = str(final_dir)
            wandb_run.summary["total_batch_steps"] = step
            wandb_run.summary["total_optimizer_steps"] = optimizer_step
    finally:
        if wandb_run is not None:
            wandb_run.finish()


def main() -> None:
    args = parse_args()
    try:
        run_training(args)
    except RuntimeError as exc:
        if not is_cuda_oom_error(exc):
            raise

        print(f"[ERROR] CUDA OOM detected with train_max_frames={args.train_max_frames}: {exc}")
        print("[ERROR] Exiting without automatic restart. Re-run manually with adjusted settings.")
        clear_cuda_memory()
        raise


if __name__ == "__main__":
    main()
