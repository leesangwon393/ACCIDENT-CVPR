import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, cast

import pandas as pd
import peft
import torch
from qwen_vl_utils import process_vision_info
from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoProcessor, BitsAndBytesConfig
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
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=0, help="0 means run full epoch schedule")
    parser.add_argument("--max-train-samples", type=int, default=0, help="0 means use all rows")
    parser.add_argument("--train-max-frames", type=int, default=48)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=0, help="0 disables intermediate saves")
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--r", type=int, default=2)
    parser.add_argument("--lora-alpha", type=int, default=4)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
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


def build_target_json(row: pd.Series) -> str:
    payload = {
        "accident_time": round(float(row["accident_time"]), 2),
        "center_x": round(float(row["center_x"]), 4),
        "center_y": round(float(row["center_y"]), 4),
        "type": str(row["type"]),
        "confidence": 1.0,
        "reasoning": build_reasoning(str(row["type"])),
    }
    return json.dumps(payload, ensure_ascii=True, indent=2)


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
    messages.append({"role": "assistant", "content": build_target_json(row)})
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


def build_model_and_processor(args: argparse.Namespace) -> "tuple[Any, Any]":
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
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
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
    return model, processor


def build_training_features(processor: Any, messages: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
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

    features: Dict[str, torch.Tensor] = {}
    for key, value in full_inputs.items():
        if isinstance(value, torch.Tensor):
            features[key] = value
    features["labels"] = labels
    return features


def move_to_device(batch: Dict[str, torch.Tensor], model: Any) -> Dict[str, torch.Tensor]:
    device = next(model.parameters()).device
    return {key: value.to(device) for key, value in batch.items()}


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] model={args.model_name}")
    print(f"[INFO] labels={args.labels_csv}")
    print(f"[INFO] output_dir={output_dir}")

    labels_df = load_training_rows(args)
    model, processor = build_model_and_processor(args)

    dataset = RowDataset(labels_df.to_dict("records"))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=lambda batch: batch[0])

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)

    total_micro_steps = len(dataloader) * args.num_epochs
    if args.max_steps > 0:
        total_micro_steps = min(total_micro_steps, args.max_steps)
    total_optimizer_steps = max(1, math.ceil(total_micro_steps / args.gradient_accumulation_steps))
    print(f"[INFO] train_rows={len(dataset)}")
    print(f"[INFO] total_micro_steps={total_micro_steps}")
    print(f"[INFO] total_optimizer_steps={total_optimizer_steps}")

    optimizer.zero_grad(set_to_none=True)
    step = 0
    optimizer_step = 0
    progress_bar = tqdm(total=total_micro_steps, desc="FT", unit="step")

    try:
        for epoch in range(args.num_epochs):
            print(f"[INFO] starting epoch {epoch + 1}/{args.num_epochs}")
            for row in dataloader:
                if args.max_steps > 0 and step >= args.max_steps:
                    break

                messages = build_training_messages(pd.Series(row), args.video_base_path, args.train_max_frames)
                batch = build_training_features(processor, messages)
                batch = move_to_device(batch, model)

                outputs = model(**batch)
                loss = outputs.loss / args.gradient_accumulation_steps
                loss.backward()
                step += 1
                progress_bar.update(1)
                progress_bar.set_postfix(loss=f"{loss.detach().item() * args.gradient_accumulation_steps:.4f}")

                if step % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_step += 1

                if step % args.log_every == 0:
                    print(
                        f"[TRAIN] micro_step={step}/{total_micro_steps} "
                        f"optimizer_step={optimizer_step}/{total_optimizer_steps} "
                        f"loss={loss.detach().item() * args.gradient_accumulation_steps:.4f}"
                    )

                if args.save_every > 0 and optimizer_step > 0 and optimizer_step % args.save_every == 0 and step % args.gradient_accumulation_steps == 0:
                    save_dir = output_dir / f"step_{optimizer_step}"
                    model.save_pretrained(save_dir)
                    processor.save_pretrained(save_dir)
                    print(f"[INFO] checkpoint saved -> {save_dir}")

            if args.max_steps > 0 and step >= args.max_steps:
                break
    finally:
        progress_bar.close()

    final_dir = output_dir / "final"
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)
    print(f"[INFO] final adapter saved -> {final_dir}")


if __name__ == "__main__":
    main()
