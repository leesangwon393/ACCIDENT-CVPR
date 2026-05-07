import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import gc
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple, cast

import pandas as pd
import peft
import torch
from qwen_vl_utils import process_vision_info
from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm.auto import tqdm
from transformers import AutoProcessor, BitsAndBytesConfig, get_cosine_schedule_with_warmup
import transformers

import baseline


Qwen3VLForConditionalGeneration = getattr(transformers, "Qwen3VLForConditionalGeneration")
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
    parser = argparse.ArgumentParser(description="Basic QLoRA FT for Qwen3-VL-8B-Instruct")
    parser.add_argument("--model-name", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--labels-csv", default="/workspace/yuyeon/raw/accident/sim_dataset/labels.csv")
    parser.add_argument("--video-base-path", default="/workspace/yuyeon/raw/accident/sim_dataset")
    parser.add_argument("--skip-list", default="/workspace/yuyeon/raw/accident/sim_dataset/skip_list.txt")
    parser.add_argument("--output-dir", default="/workspace/yuyeon/outputs/lora_qwen3vl8b_basic")
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--head-learning-rate", type=float, default=1e-4)
    parser.add_argument("--head-weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.10)
    parser.add_argument("--num-epochs", type=int, default=2)
    parser.add_argument(
        "--eval-every",
        type=int,
        default=300,
        help="Unused; validation is disabled in this script",
    )
    parser.add_argument("--max-steps", type=int, default=0, help="0 means run full epoch schedule")
    parser.add_argument("--max-train-samples", type=int, default=0, help="0 means use all rows")
    parser.add_argument("--train-max-frames", type=int, default=64)
    parser.add_argument("--oom-frame-step", type=int, default=8)
    parser.add_argument("--min-train-max-frames", type=int, default=8)
    parser.add_argument("--class-balance", choices=["none", "weighted"], default="weighted")
    parser.add_argument("--class-balance-strength", choices=["sqrt_inverse", "inverse"], default="sqrt_inverse")
    parser.add_argument("--class-balance-seed", type=int, default=42)
    parser.add_argument("--type-loss-weight", type=float, default=3.0)
    parser.add_argument("--time-loss-weight", type=float, default=1.0)
    parser.add_argument("--xy-loss-weight", type=float, default=1.0)
    parser.add_argument("--readout-last-k", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=0, help="0 disables intermediate saves")
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--r", type=int, default=4)
    parser.add_argument("--lora-alpha", type=int, default=8)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--finetune-vision-layers", type=parse_bool, default=False)
    parser.add_argument("--finetune-language-layers", type=parse_bool, default=True)
    parser.add_argument("--finetune-attention-modules", type=parse_bool, default=True)
    parser.add_argument("--finetune-mlp-modules", type=parse_bool, default=False)
    return parser.parse_args()


def load_training_rows(args: argparse.Namespace) -> pd.DataFrame:
    df = pd.read_csv(args.labels_csv)
    skip_list_path = Path(args.skip_list)
    if skip_list_path.exists():
        skip_paths = {
            line.strip()
            for line in skip_list_path.read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        }
        if skip_paths and "rgb_path" in df.columns:
            before = len(df)
            df = cast(pd.DataFrame, df.loc[~df["rgb_path"].isin(list(skip_paths))].copy())
            print(f"[INFO] excluded {before - len(df)} samples using skip list: {skip_list_path}")
    if args.max_train_samples > 0:
        df = df.iloc[: args.max_train_samples].copy()
    return df.reset_index(drop=True)


def build_reasoning(collision_type: str) -> str:
    mapping = {
        "rear-end": "The first impact is a rear-end collision where a following vehicle hits the vehicle ahead.",
        "head-on": "The first impact is a head-on collision where two vehicles strike front-to-front.",
        "sideswipe": "The first impact is a sideswipe collision with contact along the vehicle sides.",
        "t-bone": "The first impact is a perpendicular collision where one vehicle hits the side of another.",
        "single": "The first impact is a single-vehicle crash into a non-vehicle object or roadside structure.",
    }
    return mapping.get(collision_type, "The first impact matches the labeled crash type in the sampled frames.")


def build_class_balance_weights(
    labels_df: pd.DataFrame,
    strength: str,
    seed: int,
) -> "tuple[torch.Tensor, Dict[str, int], Dict[str, float], Dict[str, int]]":
    if "type" not in labels_df.columns:
        raise ValueError("Class balancing requires a 'type' column in the training dataframe.")
    if labels_df.empty:
        raise ValueError("Class balancing requires at least one training sample.")

    class_counts_series = labels_df["type"].astype(str).value_counts()
    if strength == "sqrt_inverse":
        class_weights = {label: 1.0 / math.sqrt(int(count)) for label, count in class_counts_series.items()}
    elif strength == "inverse":
        class_weights = {label: 1.0 / float(count) for label, count in class_counts_series.items()}
    else:
        raise ValueError(f"Unsupported class balance strength: {strength}")

    row_weights = torch.tensor(
        [class_weights[str(label)] for label in labels_df["type"].astype(str)],
        dtype=torch.double,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)
    sanity_sampler = WeightedRandomSampler(
        weights=row_weights,
        num_samples=len(labels_df),
        replacement=True,
        generator=generator,
    )
    sampled_indices = list(sanity_sampler)
    sampled_types = labels_df.iloc[sampled_indices]["type"].astype(str).value_counts()

    return (
        row_weights,
        {label: int(count) for label, count in class_counts_series.items()},
        class_weights,
        {label: int(sampled_types.get(label, 0)) for label in class_counts_series.index},
    )


def log_class_balance_summary(
    class_counts: Dict[str, int],
    class_weights: Dict[str, float],
    sampled_counts: Dict[str, int],
) -> None:
    ordered_labels = [label for label in baseline.VALID_TYPES if label in class_counts]
    ordered_labels.extend(label for label in class_counts if label not in ordered_labels)

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


def build_target_payload(row: pd.Series) -> Dict[str, Any]:
    return {
        "t": float(row["accident_time"]),
        "x": float(row["center_x"]),
        "y": float(row["center_y"]),
        "c": baseline.encode_type_code(str(row["type"])),
    }


FT_USER_PROMPT_TEMPLATE = """The following {n_frames} frames are sampled from a {duration:.1f}-second dashcam video.
Each frame is labeled with its exact timestamp [t=X.Xs].

Return exactly one compact JSON object with these keys in this exact order:
{{
  "t": <first physical impact time in seconds from start, between 0.0 and {duration:.1f}>,
  "x": <normalized x in [0,1]>,
  "y": <normalized y in [0,1]>,
  "c": <"HO"|"RE"|"SW"|"SI"|"TB">
}}

Crash type code definitions:
- SI : single   : vehicle vs non-vehicle object (wall, guardrail, pole, tree, barrier, curb)
- RE : rear-end : same direction - front of A hits rear of B
- HO : head-on  : opposite direction - front of A hits front of B
- TB : t-bone   : perpendicular (~90 degrees) - front of A hits side of B
- SW : sideswipe: parallel - side of A hits side of B (same or opposite direction)

Rules:
- Every video contains one crash, so do not say no crash, unknown, or none.
- Use the first physical impact, not the aftermath.
- Use `t` for impact time in seconds from the start of the video.
- Use `x` and `y` for the normalized impact point in [0,1].
- Choose exactly one crash code from the allowed list.
- Do not output any extra keys.
- Use the exact key order `t`, `x`, `y`, `c`.
- Output JSON only, on one line, with no spaces, markdown fences, or extra text.
"""


def build_target_json(row: pd.Series) -> str:
    payload = build_target_payload(row)
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"))


def thin_user_frame_content(content: List[Dict[str, Any]], train_max_frames: int) -> List[Dict[str, Any]]:
    if train_max_frames <= 0:
        return content

    image_indices = [idx for idx, item in enumerate(content) if item.get("type") == "image"]
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


def build_training_messages(row: pd.Series, video_base_path: str, train_max_frames: int) -> List[Dict[str, Any]]:
    video_rel_path = str(row["rgb_path"])
    video_path = str(Path(video_base_path) / video_rel_path)
    duration = baseline.to_float_or(row.get("duration"), 10.0)
    no_frames = baseline.to_int_or(row.get("no_frames"), 0)
    height = baseline.to_int_or(row.get("height"), 720)
    width = baseline.to_int_or(row.get("width"), 1280)
    fps = baseline.compute_video_fps(duration, no_frames, height, width)
    messages, _ = baseline.build_frame_messages(video_path, fps, duration)
    user_content = messages[1]["content"]
    messages[1]["content"] = thin_user_frame_content(user_content, train_max_frames)
    train_n_frames = sum(1 for item in messages[1]["content"] if item.get("type") == "image")
    if messages[1]["content"] and messages[1]["content"][-1].get("type") == "text":
        messages[1]["content"][-1] = {
            "type": "text",
            "text": FT_USER_PROMPT_TEMPLATE.format(
                duration=duration,
                n_frames=train_n_frames,
            ),
        }
    return messages


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
                if module_name.endswith((".attn.qkv", ".attn.proj")):
                    target_modules.append(module_name)
            if args.finetune_mlp_modules and module_name.startswith("model.visual"):
                if module_name.endswith((".mlp.linear_fc1", ".mlp.linear_fc2", ".merger.linear_fc1", ".merger.linear_fc2")):
                    target_modules.append(module_name)
                if ".deepstack_merger_list." in module_name and module_name.endswith((".linear_fc1", ".linear_fc2")):
                    target_modules.append(module_name)

    deduped = sorted(set(target_modules))
    if not deduped:
        raise ValueError(
            "No LoRA target modules selected. Enable at least one of vision/language and attention/MLP toggles."
        )
    return deduped


def build_model_and_processor(args: argparse.Namespace) -> "tuple[Any, Any, Any]":
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
    processor.tokenizer.padding_side = "left"

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_name,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    target_modules = select_lora_target_modules(model, args)
    print(f"[INFO] selected {len(target_modules)} LoRA target modules")

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
    model.print_trainable_parameters()
    hidden_size = baseline.get_hidden_size(model)
    heads = baseline.CrashPredictionHeads(hidden_size).to(next(model.parameters()).device)
    heads.train()
    return model, processor, heads


def build_training_features(processor: Any, messages: List[Dict[str, Any]], row: pd.Series) -> Dict[str, torch.Tensor]:
    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    image_inputs, video_inputs = process_vision_info(messages)

    processor_kwargs: Dict[str, Any] = {
        "padding": True,
        "return_tensors": "pt",
    }
    if image_inputs is not None:
        processor_kwargs["images"] = image_inputs
    if video_inputs is not None:
        processor_kwargs["videos"] = video_inputs

    prompt_inputs = processor(text=[prompt_text], **processor_kwargs)

    features: Dict[str, torch.Tensor] = {}
    for key, value in prompt_inputs.items():
        if isinstance(value, torch.Tensor):
            features[key] = value
    features["duration"] = torch.tensor([baseline.to_float_or(row.get("duration"), 10.0)], dtype=torch.float32)
    features["accident_time_target"] = torch.tensor([float(row["accident_time"])], dtype=torch.float32)
    features["center_x_target"] = torch.tensor([float(row["center_x"])], dtype=torch.float32)
    features["center_y_target"] = torch.tensor([float(row["center_y"])], dtype=torch.float32)
    features["type_target"] = torch.tensor(
        [baseline.CODE_TO_INDEX[baseline.encode_type_code(str(row["type"]))]],
        dtype=torch.long,
    )
    return features


def move_to_device(batch: Dict[str, torch.Tensor], model: Any) -> Dict[str, torch.Tensor]:
    device = next(model.parameters()).device
    return {key: value.to(device) for key, value in batch.items()}


def focal_loss(inputs: torch.Tensor, targets: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    ce_loss = torch.nn.functional.cross_entropy(inputs, targets, reduction='none')
    pt = torch.exp(-ce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * ce_loss
    return focal_loss.mean()


def compute_multitask_loss(
    model: Any,
    heads: Any,
    batch: Dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    model_inputs = {
        key: value
        for key, value in batch.items()
        if key
        not in {
            "duration",
            "accident_time_target",
            "center_x_target",
            "center_y_target",
            "type_target",
        }
    }
    outputs = model(**model_inputs, output_hidden_states=True, return_dict=True)
    hidden = outputs.hidden_states[-1]
    pooled = baseline.pool_last_k_hidden(hidden, model_inputs["attention_mask"], k=args.readout_last_k)
    if next(heads.parameters()).device != pooled.device:
        heads = heads.to(pooled.device)
    head_outputs = heads(pooled)

    type_logits = head_outputs["type_logits"]
    type_target = batch["type_target"]
    duration = batch["duration"]
    time_target = batch["accident_time_target"]
    x_target = batch["center_x_target"]
    y_target = batch["center_y_target"]

    type_loss = focal_loss(type_logits, type_target, alpha=0.25, gamma=2.0)

    time_pred = torch.sigmoid(head_outputs["time_raw"]) * duration
    x_pred = torch.sigmoid(head_outputs["x_raw"])
    y_pred = torch.sigmoid(head_outputs["y_raw"])

    time_loss = torch.nn.functional.smooth_l1_loss(
        (time_pred - time_target) / 1.0,
        torch.zeros_like(time_target),
    )
    xy_loss = torch.nn.functional.smooth_l1_loss(x_pred, x_target) + \
              torch.nn.functional.smooth_l1_loss(y_pred, y_target)

    total_loss = (
        args.type_loss_weight * type_loss
        + args.time_loss_weight * time_loss
        + args.xy_loss_weight * xy_loss
    )

    metrics = {
        "loss": float(total_loss.detach().item()),
        "type_loss": float(type_loss.detach().item()),
        "time_loss": float(time_loss.detach().item()),
        "xy_loss": float(xy_loss.detach().item()),
        "type_acc": float((type_logits.argmax(dim=-1) == type_target).float().mean().detach().item()),
        "time_mae": float((time_pred - time_target).abs().mean().detach().item()),
        "x_mae": float((x_pred - x_target).abs().mean().detach().item()),
        "y_mae": float((y_pred - y_target).abs().mean().detach().item()),
        "xy_euclidean": float(torch.sqrt((x_pred - x_target) ** 2 + (y_pred - y_target) ** 2).mean().detach().item()),
    }
    return total_loss, metrics


def save_training_artifacts(model: Any, processor: Any, heads: Any, save_dir: Path, args: argparse.Namespace) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_dir)
    processor.save_pretrained(save_dir)
    torch.save(heads.state_dict(), save_dir / "aux_heads.pt")
    (save_dir / "aux_heads_config.json").write_text(
        json.dumps(
            {
                "type_codes": baseline.TYPE_CODES,
                "output_schema": "txyc_continuous_v1",
                "time_output": "seconds",
                "x_output": "normalized_0_to_1",
                "y_output": "normalized_0_to_1",
                "loss": "multitask_heads_v1",
                "readout_strategy": "last_k_mean_pooling",
                "readout_last_k": args.readout_last_k,
                "lora_learning_rate": args.learning_rate,
                "lora_weight_decay": args.weight_decay,
                "head_learning_rate": args.head_learning_rate,
                "head_weight_decay": args.head_weight_decay,
                "warmup_ratio": args.warmup_ratio,
                "scheduler": "cosine_with_warmup",
            },
            indent=2,
        )
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
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] model={args.model_name}")
    print(f"[INFO] labels={args.labels_csv}")
    print(f"[INFO] output_dir={output_dir}")
    print(f"[INFO] train_max_frames={args.train_max_frames}")
    print(f"[INFO] lora_learning_rate={args.learning_rate}")
    print(f"[INFO] lora_weight_decay={args.weight_decay}")
    print(f"[INFO] head_learning_rate={args.head_learning_rate}")
    print(f"[INFO] head_weight_decay={args.head_weight_decay}")
    print(f"[INFO] warmup_ratio={args.warmup_ratio}")
    print(f"[INFO] class_balance={args.class_balance}")
    print(f"[INFO] class_balance_strength={args.class_balance_strength}")
    print(f"[INFO] type_loss_weight={args.type_loss_weight}")
    print(f"[INFO] time_loss_weight={args.time_loss_weight}")
    print(f"[INFO] xy_loss_weight={args.xy_loss_weight}")
    print("[INFO] readout_strategy=last_k_mean_pooling")
    print(f"[INFO] readout_last_k={args.readout_last_k}")
    print(f"[INFO] finetune_vision_layers={args.finetune_vision_layers}")
    print(
        "[INFO] canonical_target_format="
        f'{{"t":<float_seconds>,"x":<float_0_to_1>,"y":<float_0_to_1>,'
        f'"c":"<{"|".join(baseline.TYPE_TO_CODE.values())}>"}}'
    )
    print("[INFO] regression_targets time=seconds x=normalized y=normalized")
    print("[INFO] imbalance_strategy=weighted_sampler_only")
    print(f"[INFO] class_code_mapping={baseline.TYPE_TO_CODE}")

    labels_df = load_training_rows(args)

    model, processor, heads = build_model_and_processor(args)

    dataset = RowDataset(labels_df.to_dict("records"))
    
    if args.class_balance == "weighted":
        row_weights, class_counts, class_weights, sampled_counts = build_class_balance_weights(
            labels_df,
            args.class_balance_strength,
            args.class_balance_seed,
        )
        sampler_generator = torch.Generator()
        sampler_generator.manual_seed(args.class_balance_seed)
        sampler = WeightedRandomSampler(
            weights=row_weights,
            num_samples=len(dataset),
            replacement=True,
            generator=sampler_generator,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            sampler=sampler,
            shuffle=False,
            collate_fn=lambda batch: batch[0],
        )
        print("[INFO] weighted sampling preserves epoch length while changing class composition")
        log_class_balance_summary(class_counts, class_weights, sampled_counts)
    else:
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=lambda batch: batch[0])
        class_counts = {
            str(label): int(count)
            for label, count in labels_df["type"].astype(str).value_counts().items()
        }
        print("[INFO] class distribution before sampling:")
        for label in [t for t in baseline.VALID_TYPES if t in class_counts]:
            print(f"[INFO]   {label:10s} count={class_counts[label]:4d}")

    lora_params = [param for param in model.parameters() if param.requires_grad]
    head_params = [param for param in heads.parameters() if param.requires_grad]
    optimizer = AdamW(
        [
            {
                "params": lora_params,
                "lr": args.learning_rate,
                "weight_decay": args.weight_decay,
            },
            {
                "params": head_params,
                "lr": args.head_learning_rate,
                "weight_decay": args.head_weight_decay,
            },
        ]
    )

    total_micro_steps = len(dataloader) * args.num_epochs
    if args.max_steps > 0:
        total_micro_steps = min(total_micro_steps, args.max_steps)
    total_optimizer_steps = max(1, math.ceil(total_micro_steps / args.gradient_accumulation_steps))
    warmup_steps = min(total_optimizer_steps - 1, int(total_optimizer_steps * args.warmup_ratio))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(0, warmup_steps),
        num_training_steps=total_optimizer_steps,
    )
    print(f"[INFO] train_rows={len(dataset)}")
    print(f"[INFO] total_micro_steps={total_micro_steps}")
    print(f"[INFO] total_optimizer_steps={total_optimizer_steps}")
    print(f"[INFO] warmup_steps={max(0, warmup_steps)}")

    optimizer.zero_grad(set_to_none=True)
    step = 0
    optimizer_step = 0
    progress_bar = tqdm(total=total_micro_steps, desc="FT", unit="step")

    try:
        for epoch in range(args.num_epochs):
            tqdm.write(f"[INFO] starting epoch {epoch + 1}/{args.num_epochs}")
            for row in dataloader:
                if args.max_steps > 0 and step >= args.max_steps:
                    break

                messages = build_training_messages(pd.Series(row), args.video_base_path, args.train_max_frames)
                batch = build_training_features(processor, messages, pd.Series(row))
                batch = move_to_device(batch, model)

                heads.train()
                raw_loss, metrics = compute_multitask_loss(model, heads, batch, args)
                loss = raw_loss / args.gradient_accumulation_steps
                loss.backward()
                step += 1
                progress_bar.update(1)
                progress_bar.set_postfix(
                    loss=f"{metrics['loss']:.4f}",
                    type_acc=f"{metrics['type_acc']:.2f}",
                    t_mae=f"{metrics['time_mae']:.2f}",
                )

                if step % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_step += 1

                if step % args.log_every == 0:
                    tqdm.write(
                        f"[TRAIN] micro_step={step}/{total_micro_steps} "
                        f"optimizer_step={optimizer_step}/{total_optimizer_steps} "
                        f"loss={metrics['loss']:.4f} "
                        f"type_loss={metrics['type_loss']:.4f} "
                        f"time_loss={metrics['time_loss']:.4f} "
                        f"xy_loss={metrics['xy_loss']:.4f} "
                        f"type_acc={metrics['type_acc']:.3f} "
                        f"time_mae={metrics['time_mae']:.3f} "
                        f"xy_euclidean={metrics['xy_euclidean']:.3f} "
                    )

                if args.save_every > 0 and optimizer_step > 0 and optimizer_step % args.save_every == 0 and step % args.gradient_accumulation_steps == 0:
                    save_dir = output_dir / f"step_{optimizer_step}"
                    save_training_artifacts(model, processor, heads, save_dir, args)
                    print(f"[INFO] checkpoint saved -> {save_dir}")

                del loss
                del raw_loss
                del batch
                del messages
            
            if args.max_steps > 0 and step >= args.max_steps:
                break
    finally:
        progress_bar.close()

    final_dir = output_dir / "final"
    save_training_artifacts(model, processor, heads, final_dir, args)
    print(f"[INFO] final adapter saved -> {final_dir}")


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
