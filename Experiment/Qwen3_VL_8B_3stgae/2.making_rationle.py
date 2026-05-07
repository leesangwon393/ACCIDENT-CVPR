"""RB-FT stage-1 rationale generation for Qwen3-VL.

This script is the canonical rationale prompt/output definition for this repo.
It assumes a Qwen/Qwen3-VL runtime with:
- `transformers` exposing `Qwen3VLForConditionalGeneration`
- `qwen_vl_utils.process_vision_info`
- optional `peft` for adapter loading
- optional `bitsandbytes` when using 4-bit or 8-bit base precision

Dataset and video paths are intentionally not hardcoded. Pass them at runtime.
"""

STAGE1_USER_PROMPT_TEMPLATE = """Role: Traffic Accident Analysis Expert. Ground Truth Label: {label}
Task: Based on the provided label and the video content, analyze the traffic scene and explain the accident-related event or abnormal driving situation shown in the video.
Keep your total response concise, under 400 tokens. Do NOT repeat the same observation. Each fact should appear only once.
Structure your reasoning across the following four dimensions:
Subjects: Identify the main entities involved in the scene, such as vehicles, guardrails, lane markings, traffic lights, road boundaries, and other relevant objects.
Attributes: Describe the important characteristics of these entities, including relative positions, lane placement, estimated speed changes, heading direction, vehicle type/color, road geometry, visibility, lighting, and weather or surface conditions.
Actions: Detail the key motions and interactions between entities. Focus on behaviors related to accident occurrence or accident precursors, such as sudden braking, rapid lane departure, failure to yield, unsafe turning, rear-end approach, side impact, collision, near-collision, rollover, or loss of control.
Scenes: Describe the overall traffic environment, such as intersection, highway, merge area, curved road, narrow street, crosswalk, roadside area, traffic density, and surrounding visual conditions.
Output: Provide a detailed rationale explaining why these elements indicate a traffic accident or accident-related event requiring attention.
"""

import argparse
import importlib.util
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

pd = __import__("pandas")
torch = __import__("torch")
transformers = __import__("transformers")
qwen_vl_utils = __import__("qwen_vl_utils")
process_vision_info = getattr(qwen_vl_utils, "process_vision_info")
AutoProcessor = getattr(__import__("transformers", fromlist=["AutoProcessor"]), "AutoProcessor")
AutoModelForImageTextToText = getattr(transformers, "AutoModelForImageTextToText", None)
AutoModelForVision2Seq = getattr(transformers, "AutoModelForVision2Seq", None)
BitsAndBytesConfig = getattr(__import__("transformers", fromlist=["BitsAndBytesConfig"]), "BitsAndBytesConfig")


Qwen3VLForConditionalGeneration = getattr(transformers, "Qwen3VLForConditionalGeneration", None)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_JSONL = SCRIPT_DIR / "outputs" / "stage1_rationales.jsonl"


def load_local_module(module_path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate stage-1 rationale records from labeled videos")
    parser.add_argument("--labels-csv", required=True, help="Path to training labels CSV")
    parser.add_argument("--video-base-path", required=True, help="Base directory that contains video files")
    parser.add_argument("--output-jsonl", default=str(DEFAULT_OUTPUT_JSONL), help="Output rationale dataset JSONL path")
    parser.add_argument("--path-column", default="rgb_path", help="CSV column that stores the video-relative path")
    parser.add_argument("--label-column", default="type", help="CSV column used for normalized label metadata")
    parser.add_argument(
        "--prompt-label-column",
        default="",
        help="Optional CSV column used only for the prompt Ground Truth Label line; defaults to label-column",
    )
    parser.add_argument(
        "--use-raw-prompt-label",
        action="store_true",
        help="Use the prompt-label-column value verbatim instead of normalizing it through baseline.normalize_type",
    )
    parser.add_argument("--model-name", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--processor-name", default="", help="Optional processor id/path; defaults to model-name")
    parser.add_argument("--adapter-path", default="", help="Optional LoRA adapter to load before generation")
    parser.add_argument("--base-precision", choices=["4bit", "8bit", "bf16"], default="4bit")
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all rows")
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--max-side", type=int, default=480)
    parser.add_argument("--max-pixels", type=int, default=480 * 480)
    parser.add_argument("--min-pixels", type=int, default=28 * 28)
    parser.add_argument("--max-new-tokens", type=int, default=400)
    parser.add_argument("--max-duration-seconds", type=float, default=60.0)
    return parser.parse_args()


def build_quantization_config(base_precision: str) -> Tuple[Optional[Any], Dict[str, Any]]:
    load_kwargs: Dict[str, Any] = {
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
    }
    quant_config: Optional[Any] = None

    if base_precision == "4bit":
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_storage=torch.bfloat16,
        )
        load_kwargs["quantization_config"] = quant_config
        load_kwargs["attn_implementation"] = "flash_attention_2"
    elif base_precision == "8bit":
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
        load_kwargs["quantization_config"] = quant_config
        load_kwargs["attn_implementation"] = "flash_attention_2"
    else:
        load_kwargs["attn_implementation"] = "sdpa"

    if torch.cuda.is_available():
        load_kwargs["device_map"] = "auto"

    return quant_config, load_kwargs


def load_model_and_processor(args: argparse.Namespace) -> Tuple[Any, Any]:
    _, load_kwargs = build_quantization_config(args.base_precision)
    processor_name = args.processor_name or args.model_name

    processor = AutoProcessor.from_pretrained(processor_name, trust_remote_code=True)
    tokenizer_like = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    if tokenizer_like.pad_token_id is None:
        tokenizer_like.pad_token_id = tokenizer_like.eos_token_id
    tokenizer_like.padding_side = "left"

    model_cls = Qwen3VLForConditionalGeneration
    if model_cls is not None:
        model = model_cls.from_pretrained(args.model_name, **load_kwargs)
    else:
        if AutoModelForImageTextToText is not None:
            model = AutoModelForImageTextToText.from_pretrained(args.model_name, **load_kwargs)
        elif AutoModelForVision2Seq is not None:
            model = AutoModelForVision2Seq.from_pretrained(args.model_name, **load_kwargs)
        else:
            raise RuntimeError("No compatible multimodal model loader is available in the installed transformers version.")
    model.eval()

    if args.adapter_path:
        peft_model_cls = getattr(__import__("peft"), "PeftModel")
        model = peft_model_cls.from_pretrained(model, args.adapter_path)
        model.eval()

    return model, processor


def clean_generated_text(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def generate_rationale(model: Any, processor: Any, messages: List[Dict[str, Any]], max_new_tokens: int) -> str:
    text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info([messages])

    processor_kwargs: Dict[str, Any] = {
        "text": [text_input],
        "padding": True,
        "return_tensors": "pt",
    }
    if image_inputs is not None:
        processor_kwargs["images"] = image_inputs
    if video_inputs is not None:
        processor_kwargs["videos"] = video_inputs

    inputs = processor(**processor_kwargs).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0,
            top_p=None,
            repetition_penalty=1.2,
        )

    trimmed = generated_ids[:, inputs["input_ids"].shape[1] :]
    decoded = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return clean_generated_text(decoded)


def build_messages_for_row(
    row: Any,
    baseline: Any,
    video_base_path: Path,
    path_column: str,
    label_column: str,
    prompt_label_column: str,
    use_raw_prompt_label: bool,
    max_frames: int,
    max_side: int,
    max_pixels: int,
    min_pixels: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    video_rel_path = str(row[path_column])
    video_path = str(video_base_path / video_rel_path)

    duration = baseline.to_float_or(row.get("duration"), 10.0)
    no_frames = baseline.to_int_or(row.get("no_frames"), 0)
    height = baseline.to_int_or(row.get("height"), 720)
    width = baseline.to_int_or(row.get("width"), 1280)
    fps = baseline.compute_video_fps(duration, no_frames, height, width)

    frames_with_ts = baseline.sample_frames_with_timestamps(
        video_path,
        fps,
        duration,
        max_frames=max_frames,
        max_side=max_side,
    )

    raw_label = str(row[label_column])
    normalized_label = baseline.normalize_type(raw_label)
    prompt_label_value = str(row[prompt_label_column]) if prompt_label_column else raw_label
    if not use_raw_prompt_label:
        prompt_label_value = normalized_label
    prompt_text = STAGE1_USER_PROMPT_TEMPLATE.format(label=prompt_label_value)

    user_content: List[Dict[str, Any]] = []
    for frame_image, ts in frames_with_ts:
        user_content.append({"type": "text", "text": f"[t={ts}s]"})
        user_content.append(
            {
                "type": "image",
                "image": frame_image,
                "max_pixels": max_pixels,
                "min_pixels": min_pixels,
            }
        )
    user_content.append({"type": "text", "text": prompt_text})

    messages = [{"role": "user", "content": user_content}]
    metadata = {
        "video_rel_path": video_rel_path,
        "prompt_label": prompt_label_value,
        "duration": float(duration),
        "no_frames": int(no_frames),
        "height": int(height),
        "width": int(width),
        "fps": float(fps),
        "n_sampled_frames": int(len(frames_with_ts)),
    }
    return messages, metadata


def load_rows(labels_csv: str, path_column: str, label_column: str, prompt_label_column: str, max_samples: int) -> Any:
    df = pd.read_csv(labels_csv)
    required_columns = {path_column, label_column}
    if prompt_label_column:
        required_columns.add(prompt_label_column)
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"labels CSV missing required columns: {sorted(missing)}")
    if max_samples > 0:
        df = df.iloc[:max_samples].copy()
    return df.reset_index(drop=True)


def run(args: argparse.Namespace) -> None:
    script_dir = Path(__file__).resolve().parent
    baseline = load_local_module(script_dir / "0. baseline.py", "baseline_local")

    labels_df = load_rows(
        args.labels_csv,
        args.path_column,
        args.label_column,
        args.prompt_label_column,
        args.max_samples,
    )
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model, processor = load_model_and_processor(args)

    success_count = 0
    skipped_count = 0
    with output_path.open("w", encoding="utf-8") as writer:
        for idx, row in labels_df.iterrows():
            row_idx = int(idx) + 1
            duration = baseline.to_float_or(row.get("duration"), 10.0)
            if duration > args.max_duration_seconds:
                skipped_count += 1
                print(
                    f"[SKIP] {row_idx}/{len(labels_df)} {row.get(args.path_column, 'unknown')} "
                    f"duration={duration:.2f}s exceeds max_duration_seconds={args.max_duration_seconds:.2f}"
                )
                continue
            try:
                messages, metadata = build_messages_for_row(
                    row,
                    baseline,
                    Path(args.video_base_path),
                    args.path_column,
                    args.label_column,
                    args.prompt_label_column,
                    args.use_raw_prompt_label,
                    args.max_frames,
                    args.max_side,
                    args.max_pixels,
                    args.min_pixels,
                )
                rationale = generate_rationale(model, processor, messages, args.max_new_tokens)
                record = {
                    **metadata,
                    "rationale": rationale,
                }
                writer.write(json.dumps(record, ensure_ascii=True) + "\n")
                success_count += 1
                print(f"[OK] {row_idx}/{len(labels_df)} {metadata['video_rel_path']}")
            except Exception as exc:
                print(f"[WARN] failed {row_idx}/{len(labels_df)} {row.get('rgb_path', 'unknown')}: {exc}")

    print(f"[INFO] skipped {skipped_count} rows longer than {args.max_duration_seconds:.2f}s")
    print(f"[INFO] wrote {success_count} rationale records -> {output_path}")


if __name__ == "__main__":
    run(parse_args())
