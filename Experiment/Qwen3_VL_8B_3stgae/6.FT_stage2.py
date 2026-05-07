"""RB-FT stage-2 task fine-tuning for Qwen3-VL.

Training contract:
- initialize from the base Qwen3-VL model or from a stage-1 adapter via
  `--stage1-adapter-path`
- input: traffic-accident video frames plus the task prompt for final output
- target: final competition JSON only

This script assumes the same Qwen3-VL runtime used throughout this repo:
`transformers` with `Qwen3VLForConditionalGeneration`,
`qwen_vl_utils.process_vision_info`, `peft`, and optional `bitsandbytes`.

Dataset and video paths are intentionally required as CLI arguments rather than
being hardcoded, because stage-1 and stage-2 data sources may differ.
"""

import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import gc
import importlib.util
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

pd = __import__("pandas")
peft = __import__("peft")
torch = __import__("torch")
transformers = __import__("transformers")
qwen_vl_utils = __import__("qwen_vl_utils")
process_vision_info = getattr(qwen_vl_utils, "process_vision_info")
lr_scheduler_mod = __import__(
    "torch.optim.lr_scheduler",
    fromlist=["CosineAnnealingLR", "LambdaLR", "SequentialLR"],
)
CosineAnnealingLR = getattr(lr_scheduler_mod, "CosineAnnealingLR")
LambdaLR = getattr(lr_scheduler_mod, "LambdaLR")
SequentialLR = getattr(lr_scheduler_mod, "SequentialLR")
Optimizer = getattr(
    __import__("torch.optim.optimizer", fromlist=["Optimizer"]), "Optimizer"
)
AdamW = getattr(__import__("torch.optim.adamw", fromlist=["AdamW"]), "AdamW")
data_mod = __import__(
    "torch.utils.data", fromlist=["DataLoader", "Dataset", "WeightedRandomSampler"]
)
DataLoader = getattr(data_mod, "DataLoader")
Dataset = getattr(data_mod, "Dataset")
WeightedRandomSampler = getattr(data_mod, "WeightedRandomSampler")
tqdm = getattr(__import__("tqdm.auto", fromlist=["tqdm"]), "tqdm")
AutoProcessor = getattr(
    __import__("transformers", fromlist=["AutoProcessor"]), "AutoProcessor"
)
BitsAndBytesConfig = getattr(
    __import__("transformers", fromlist=["BitsAndBytesConfig"]), "BitsAndBytesConfig"
)


def load_local_module(module_path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SCRIPT_DIR = Path(__file__).resolve().parent
baseline = load_local_module(SCRIPT_DIR / "0. baseline.py", "baseline_local_stage2")
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs" / "stage2_task_lora"


Qwen3VLForConditionalGeneration = getattr(
    transformers, "Qwen3VLForConditionalGeneration", None
)
LoraConfig = getattr(peft, "LoraConfig")
get_peft_model = getattr(peft, "get_peft_model")
prepare_model_for_kbit_training = getattr(peft, "prepare_model_for_kbit_training")


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
    parser = argparse.ArgumentParser(
        description="Conservative generative QLoRA FT for Qwen3-VL-8B-Instruct"
    )
    parser.add_argument("--model-name", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--labels-csv", required=True)
    parser.add_argument("--video-base-path", required=True)
    parser.add_argument(
        "--skip-list", default="", help="Optional path to skip-list text file"
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--stage1-adapter-path",
        default="/root/Desktop/workspace/yuyeon/Experiments/16. RB-FT/outputs/experiment_artifacts/adapter_stage1_fix_version_with_mlp_e3_lang5e5_vis1e5_wd001_b1_ga8/final",
        help="Optional stage-1 adapter/checkpoint path for stage-2 initialization",
    )
    parser.add_argument("--language-learning-rate", type=float, default=5e-5)
    parser.add_argument("--vision-learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--eta-min", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument(
        "--max-steps", type=int, default=0, help="0 means run full epoch schedule"
    )
    parser.add_argument(
        "--max-train-samples", type=int, default=0, help="0 means use all rows"
    )
    parser.add_argument("--train-max-frames", type=int, default=32)
    parser.add_argument("--train-max-side", type=int, default=480)
    parser.add_argument(
        "--class-balance", choices=["none", "weighted"], default="weighted"
    )
    parser.add_argument(
        "--class-balance-strength",
        choices=["sqrt_inverse", "inverse"],
        default="sqrt_inverse",
    )
    parser.add_argument("--class-balance-seed", type=int, default=42)
    parser.add_argument("--train-mini-batch", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument(
        "--save-every", type=int, default=0, help="0 disables intermediate saves"
    )
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--finetune-vision-layers", type=parse_bool, default=True)
    parser.add_argument("--finetune-language-layers", type=parse_bool, default=True)
    parser.add_argument("--finetune-attention-modules", type=parse_bool, default=True)
    parser.add_argument("--finetune-mlp-modules", type=parse_bool, default=True)
    parser.add_argument(
        "--base-precision", choices=["4bit", "8bit", "bf16"], default="4bit"
    )
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


def maybe_init_wandb(
    args: argparse.Namespace, output_dir: Path, stage_name: str
) -> Any:
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
    warmup_steps = (
        min(total_optimizer_steps - 1, int(total_optimizer_steps * warmup_ratio))
        if total_optimizer_steps > 1
        else 0
    )
    cosine_kwargs: Dict[str, Any] = {"eta_min": eta_min}
    if warmup_steps <= 0:
        return CosineAnnealingLR(
            optimizer, T_max=max(1, total_optimizer_steps), **cosine_kwargs
        )

    warmup_scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda current_step: (
            float(current_step + 1) / float(max(1, warmup_steps))
        ),
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


def load_training_rows(args: argparse.Namespace) -> Any:
    df = pd.read_csv(args.labels_csv)
    if args.skip_list:
        skip_list_path = Path(args.skip_list)
        if not skip_list_path.is_file():
            print(f"[WARN] skip-list path is not a file, ignoring: {skip_list_path}")
            skip_list_path = Path()
    else:
        skip_list_path = Path()

    if skip_list_path.is_file():
        skip_paths = {
            line.strip()
            for line in skip_list_path.read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        }
        if skip_paths and "rgb_path" in df.columns:
            before = len(df)
            df = cast(Any, df.loc[~df["rgb_path"].isin(list(skip_paths))].copy())
            print(
                f"[INFO] excluded {before - len(df)} samples using skip list: {skip_list_path}"
            )
    if args.max_train_samples > 0:
        df = df.iloc[: args.max_train_samples].copy()
    return df.reset_index(drop=True)


def build_class_balance_weights(
    labels_df: Any,
    strength: str,
    seed: int,
) -> "tuple[Any, Dict[str, int], Dict[str, float], Dict[str, int]]":
    if "type" not in labels_df.columns:
        raise ValueError(
            "Class balancing requires a 'type' column in the training dataframe."
        )
    if labels_df.empty:
        raise ValueError("Class balancing requires at least one training sample.")

    class_counts_series = labels_df["type"].astype(str).value_counts()
    if strength == "sqrt_inverse":
        class_weights = {
            label: 1.0 / math.sqrt(int(count))
            for label, count in class_counts_series.items()
        }
    elif strength == "inverse":
        class_weights = {
            label: 1.0 / float(count) for label, count in class_counts_series.items()
        }
    else:
        raise ValueError(f"Unsupported class balance strength: {strength}")

    row_weights = torch.tensor(
        [class_weights[str(label)] for label in labels_df["type"].astype(str)],
        dtype=torch.double,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)
    sanity_sampler = WeightedRandomSampler(
        weights=row_weights.tolist(),
        num_samples=len(labels_df),
        replacement=True,
        generator=generator,
    )
    sampled_indices = list(sanity_sampler)
    sampled_types = labels_df.iloc[sampled_indices]["type"].astype(str).value_counts()

    return (
        row_weights,
        {str(label): int(count) for label, count in class_counts_series.items()},
        {str(label): float(weight) for label, weight in class_weights.items()},
        {
            str(label): int(sampled_types.get(label, 0))
            for label in class_counts_series.index
        },
    )


def log_class_balance_summary(
    class_counts: Dict[str, int],
    class_weights: Dict[str, float],
    sampled_counts: Dict[str, int],
) -> None:
    ordered_labels = [label for label in baseline.VALID_TYPES if label in class_counts]
    ordered_labels.extend(
        label for label in class_counts if label not in ordered_labels
    )

    min_weight = min(class_weights.values())
    total_weight = sum(class_weights.values())
    print("[INFO] class distribution before sampling:")
    for label in ordered_labels:
        count = class_counts[label]
        weight = class_weights[label]
        relative = weight / min_weight if min_weight > 0 else 1.0
        normalized = weight / total_weight if total_weight > 0 else 0.0
        sampled = sampled_counts.get(label, 0)
        print(
            f"[INFO]   {label:10s} count={count:4d} "
            f"weight={weight:.6f} relative={relative:.2f}x normalized={normalized:.4f} "
            f"sampled_epoch={sampled:4d}"
        )


def build_target_payload(row: Any) -> Dict[str, Any]:
    collision_type = baseline.normalize_type(str(row["type"]))
    return {
        "type": collision_type,
        "accident_time": round(float(row["accident_time"]), 2),
        "center_x": round(float(row["center_x"]), 3),
        "center_y": round(float(row["center_y"]), 3),
    }


def build_target_json(row: Any) -> str:
    payload = build_target_payload(row)
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"))


TRAIN_USER_PROMPT_TEMPLATE = baseline.USER_PROMPT_TEMPLATE


def thin_user_frame_content(
    content: List[Dict[str, Any]], train_max_frames: int
) -> List[Dict[str, Any]]:
    if train_max_frames <= 0:
        return content

    image_indices = [
        idx for idx, item in enumerate(content) if item.get("type") == "image"
    ]
    if len(image_indices) <= train_max_frames:
        return content

    keep_positions = {
        round(i * (len(image_indices) - 1) / (train_max_frames - 1))
        for i in range(train_max_frames)
    }
    keep_image_indices = {image_indices[pos] for pos in sorted(keep_positions)}

    thinned: List[Dict[str, Any]] = []
    for idx, item in enumerate(content):
        if item.get("type") == "image":
            if idx in keep_image_indices:
                thinned.append(item)
            continue
        if item.get("type") == "text":
            if idx + 1 < len(content) and content[idx + 1].get("type") == "image":
                if idx + 1 in keep_image_indices:
                    thinned.append(item)
                continue
        thinned.append(item)

    return thinned


def build_training_messages(
    row: Any, video_base_path: str, train_max_frames: int, train_max_side: int
) -> List[Dict[str, Any]]:
    video_rel_path = str(row["rgb_path"])
    video_path = str(Path(video_base_path) / video_rel_path)
    duration = baseline.to_float_or(row.get("duration"), 10.0)
    no_frames = baseline.to_int_or(row.get("no_frames"), 0)
    height = baseline.to_int_or(row.get("height"), 720)
    width = baseline.to_int_or(row.get("width"), 1280)
    fps = baseline.compute_video_fps(duration, no_frames, height, width)
    messages, train_n_frames = baseline.build_frame_messages(
        video_path,
        fps,
        duration,
        max_frames=train_max_frames,
        max_side=train_max_side,
    )
    if messages[1]["content"] and messages[1]["content"][-1].get("type") == "text":
        messages[1]["content"][-1] = {
            "type": "text",
            "text": TRAIN_USER_PROMPT_TEMPLATE.format(
                duration=duration, n_frames=train_n_frames
            ),
        }
    messages.append({"role": "assistant", "content": build_target_json(row)})
    return messages


def select_lora_target_modules(model: Any, args: argparse.Namespace) -> List[str]:
    target_modules: List[str] = []

    for module_name, _ in model.named_modules():
        if args.finetune_language_layers:
            if args.finetune_attention_modules and module_name.startswith(
                "model.language_model.layers."
            ):
                if module_name.endswith(
                    (
                        ".self_attn.q_proj",
                        ".self_attn.k_proj",
                        ".self_attn.v_proj",
                        ".self_attn.o_proj",
                    )
                ):
                    target_modules.append(module_name)

        if args.finetune_vision_layers:
            if args.finetune_attention_modules and module_name.startswith(
                "model.visual.blocks."
            ):
                parts = module_name.split(".")
                block_index = (
                    int(parts[3]) if len(parts) > 3 and parts[3].isdigit() else -1
                )
                if 20 <= block_index <= 26 and module_name.endswith(
                    (".attn.qkv", ".attn.proj")
                ):
                    target_modules.append(module_name)
            if args.finetune_mlp_modules:
                if module_name.startswith(
                    "model.visual.merger."
                ) and module_name.endswith((".linear_fc1", ".linear_fc2")):
                    target_modules.append(module_name)
                if module_name.startswith(
                    "model.visual.deepstack_merger_list."
                ) and module_name.endswith((".linear_fc1", ".linear_fc2")):
                    target_modules.append(module_name)

    deduped = sorted(set(target_modules))
    if not deduped:
        raise ValueError(
            "No LoRA target modules selected. Enable at least one of vision/language and attention/MLP toggles."
        )
    return deduped


def build_optimizer_param_groups(
    model: Any, args: argparse.Namespace
) -> List[Dict[str, Any]]:
    language_params: List[Any] = []
    vision_merger_params: List[Any] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(
            token in name for token in ("visual", "merger", "deepstack_merger_list")
        ):
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


def build_model_and_processor(
    args: argparse.Namespace, runtime_device: Any
) -> "tuple[Any, Any]":
    if Qwen3VLForConditionalGeneration is None:
        raise RuntimeError(
            "Qwen3VLForConditionalGeneration is unavailable in the installed transformers version."
        )

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
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": True}
        )
        model.enable_input_require_grads()

    if (
        not args.finetune_vision_layers
        and hasattr(model, "model")
        and hasattr(model.model, "visual")
    ):
        model.model.visual.gradient_checkpointing_disable()

    if args.stage1_adapter_path:
        stage1_adapter = Path(args.stage1_adapter_path)
        if not stage1_adapter.exists():
            raise FileNotFoundError(
                f"Stage-1 adapter path does not exist: {stage1_adapter}"
            )
        peft_model_cls = getattr(peft, "PeftModel")
        model = peft_model_cls.from_pretrained(
            model, str(stage1_adapter), is_trainable=True
        )
        log_info(f"[INFO] initialized stage-2 from stage-1 adapter: {stage1_adapter}")
    else:
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


def build_training_features(
    processor: Any, messages: List[Dict[str, Any]]
) -> Dict[str, Any]:
    prompt_messages = messages[:-1]

    full_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    prompt_text = processor.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )

    vision_result = process_vision_info(prompt_messages)
    image_inputs = vision_result[0]
    video_inputs = vision_result[1]

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


def save_training_artifacts(
    model: Any,
    processor: Any,
    save_dir: Path,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_dir)
    processor.save_pretrained(save_dir)
    if metadata is not None:
        metadata_path = save_dir / "checkpoint_metrics.json"
        metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=True) + "\n"
        )


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
    wandb_run = maybe_init_wandb(args, output_dir, "stage2")

    log_info(f"[INFO] model={args.model_name}")
    log_info(f"[INFO] labels={args.labels_csv}")
    log_info(f"[INFO] output_dir={output_dir}")
    log_info(f"[INFO] base_precision={args.base_precision}")
    log_info(f"[INFO] stage1_adapter_path={args.stage1_adapter_path or 'none'}")
    log_info("[INFO] device_map=auto")
    log_info(f"[INFO] train_max_frames={args.train_max_frames}")
    log_info(
        f"[INFO] optimizer=AdamW warmup_ratio={args.warmup_ratio} grad_clip={args.grad_clip} eta_min={args.eta_min}"
    )
    log_info(f"[INFO] language_learning_rate={args.language_learning_rate}")
    log_info(f"[INFO] vision_learning_rate={args.vision_learning_rate}")
    log_info(f"[INFO] weight_decay={args.weight_decay}")
    log_info(f"[INFO] train_mini_batch={args.train_mini_batch}")
    log_info(f"[INFO] gradient_accumulation_steps={args.gradient_accumulation_steps}")
    log_info(f"[INFO] num_epochs={args.num_epochs}")
    log_info(f"[INFO] class_balance={args.class_balance}")
    log_info(f"[INFO] class_balance_strength={args.class_balance_strength}")
    log_info(f"[INFO] finetune_vision_layers={args.finetune_vision_layers}")
    log_info(
        "[INFO] target_json_format="
        '{"type":"<rear-end|head-on|sideswipe|t-bone|single>",'
        '"accident_time":<float>,"center_x":<float>,"center_y":<float>}'
    )
    log_info("[INFO] type_loss=included_in_assistant_target")
    log_info("[INFO] training_objective=causal_lm_json_teacher_forcing")

    labels_df = load_training_rows(args)
    model, processor = build_model_and_processor(args, runtime_device)

    dataset = RowDataset(labels_df.to_dict("records"))

    if args.class_balance == "weighted":
        row_weights, class_counts, class_weights, sampled_counts = (
            build_class_balance_weights(
                labels_df,
                args.class_balance_strength,
                args.class_balance_seed,
            )
        )
        sampler_generator = torch.Generator()
        sampler_generator.manual_seed(args.class_balance_seed)
        sampler = WeightedRandomSampler(
            weights=row_weights.tolist(),
            num_samples=len(dataset),
            replacement=True,
            generator=sampler_generator,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=args.train_mini_batch,
            sampler=sampler,
            shuffle=False,
            collate_fn=lambda batch: batch,
        )
        log_info(
            "[INFO] weighted sampling preserves epoch length while changing class composition"
        )
        if is_main_process():
            log_class_balance_summary(class_counts, class_weights, sampled_counts)
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=args.train_mini_batch,
            shuffle=True,
            sampler=None,
            collate_fn=lambda batch: batch,
        )
        class_counts = {
            str(label): int(count)
            for label, count in labels_df["type"].astype(str).value_counts().items()
        }
        log_info("[INFO] class distribution before sampling:")
        for label in [t for t in baseline.VALID_TYPES if t in class_counts]:
            log_info(f"[INFO]   {label:10s} count={class_counts[label]:4d}")

    optimizer_param_groups = build_optimizer_param_groups(model, args)
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = AdamW(optimizer_param_groups)

    total_batch_steps = len(dataloader) * args.num_epochs
    if args.max_steps > 0:
        total_batch_steps = min(total_batch_steps, args.max_steps)
    total_optimizer_steps = max(
        1, math.ceil(total_batch_steps / args.gradient_accumulation_steps)
    )
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
    latest_avg_batch_loss: Optional[float] = None
    progress_bar = (
        tqdm(total=total_batch_steps, desc="FT", unit="batch")
        if is_main_process()
        else None
    )

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
                        messages = build_training_messages(
                            pd.Series(row),
                            args.video_base_path,
                            args.train_max_frames,
                            args.train_max_side,
                        )
                        batch = build_training_features(processor, messages)
                        batch = move_to_device(batch, train_model)

                        outputs = train_model(**batch)
                        raw_loss = outputs.loss
                        loss = raw_loss / float(
                            len(batch_rows) * args.gradient_accumulation_steps
                        )
                        loss.backward()
                        batch_loss_total += raw_loss.detach().item()

                        del loss
                        del raw_loss
                        del outputs
                        del batch
                        del messages

                    avg_batch_loss = batch_loss_total / max(1, len(batch_rows))
                    latest_avg_batch_loss = avg_batch_loss
                    step += 1
                    if step % args.gradient_accumulation_steps == 0:
                        if args.grad_clip > 0:
                            torch.nn.utils.clip_grad_norm_(
                                trainable_params, args.grad_clip
                            )
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
                                "train/vision_lr": optimizer.param_groups[1]["lr"]
                                if len(optimizer.param_groups) > 1
                                else 0.0,
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

                    if (
                        args.save_every > 0
                        and optimizer_step > 0
                        and optimizer_step % args.save_every == 0
                        and step % args.gradient_accumulation_steps == 0
                    ):
                        save_dir = output_dir / f"step_{optimizer_step}"
                        checkpoint_metadata: Dict[str, Any] = {
                            "epoch": epoch + 1,
                            "batch_step": step,
                            "optimizer_step": optimizer_step,
                            "avg_batch_loss": avg_batch_loss,
                        }
                        save_training_artifacts(
                            train_model,
                            processor,
                            save_dir,
                            metadata=checkpoint_metadata,
                        )
                        log_info(
                            f"[INFO] checkpoint saved -> {save_dir} (loss={avg_batch_loss:.4f})"
                        )

                clear_cuda_memory()
                log_info(
                    f"[INFO] cleared CUDA memory after epoch {epoch + 1}/{args.num_epochs}"
                )

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
                log_info(
                    f"[INFO] flushed final partial accumulation window ({pending_micro_steps} micro_steps)"
                )
        finally:
            if progress_bar is not None:
                progress_bar.close()

        final_dir = output_dir / "final"
        final_metadata: Dict[str, Any] = {
            "total_batch_steps": step,
            "total_optimizer_steps": optimizer_step,
        }
        if latest_avg_batch_loss is not None:
            final_metadata["last_avg_batch_loss"] = latest_avg_batch_loss
        save_training_artifacts(
            train_model,
            processor,
            final_dir,
            metadata=final_metadata,
        )
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

        print(
            f"[ERROR] CUDA OOM detected with train_max_frames={args.train_max_frames}: {exc}"
        )
        print(
            "[ERROR] Exiting without automatic restart. Re-run manually with adjusted settings."
        )
        clear_cuda_memory()
        raise


if __name__ == "__main__":
    main()
