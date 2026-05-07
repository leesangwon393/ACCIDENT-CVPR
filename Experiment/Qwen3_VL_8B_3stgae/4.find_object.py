import argparse
import gc
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import torch
from PIL import Image

try:
    transformers = __import__("transformers")
    AutoProcessor = getattr(transformers, "AutoProcessor")
    AutoModelForVision2Seq = getattr(transformers, "AutoModelForVision2Seq", None)
    BitsAndBytesConfig = getattr(transformers, "BitsAndBytesConfig")
    Qwen3VLForConditionalGeneration = getattr(
        transformers, "Qwen3VLForConditionalGeneration", None
    )

    import peft
    PeftModel = getattr(peft, "PeftModel")
except ImportError as e:
    raise RuntimeError(f"Dependencies missing: {e}")

DEFAULT_BASE_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_ADAPTER_PATH = "/root/Desktop/workspace/yuyeon/Experiments/16. RB-FT/outputs/experiment_artifacts/adapter_stage1_fix_version_with_mlp_e3_lang5e5_vis1e5_wd001_b1_ga8"

# 너무 세분화 되지 않도록 수정
INFER_PROMPT = """You are analyzing traffic accident video frames.

Task:
Return only the most important object categories that are directly relevant to vehicle accidents in the provided frames.

Constraints:
1. Always include "vehicle" if any road vehicle is visible.
2. Besides "vehicle", include only objects that can directly affect, constrain, or explain a vehicle accident.
3. Focus on accident-relevant objects such as road, barrier, divider, traffic sign, traffic light, pole, guardrail, bridge, or other physical obstacles.
4. Exclude less relevant context objects such as sidewalk, crosswalk, snow, weather, shoulder, vegetation, building, or background scenery unless they are clearly essential to explaining the accident.
5. Prefer broad, general categories over specific or fine-grained names.
6. Merge visually similar or functionally similar objects into one general category when possible.
7. Avoid using multiple names for nearly the same object type.
8. Use concise, standard English object names.
9. Use singular nouns only.
10. Output valid JSON only.
11. Output only a JSON array of strings.
12. Do not include any explanation or extra text.

Output:
"""

def load_model_and_processor(args: argparse.Namespace) -> Tuple[Any, Any]:
    if Qwen3VLForConditionalGeneration is None and AutoModelForVision2Seq is None:
        raise RuntimeError(
            "Neither Qwen3VLForConditionalGeneration nor AutoModelForVision2Seq is available."
        )

    print(f"[INFO] Loading base model: {args.base_model}")

    load_kwargs = {
        "trust_remote_code": True,
        "device_map": "auto" if torch.cuda.is_available() else "cpu",
    }

    if args.precision == "4bit":
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        load_kwargs["torch_dtype"] = torch.bfloat16
    else:
        load_kwargs["torch_dtype"] = torch.bfloat16

    processor = AutoProcessor.from_pretrained(
        args.base_model,
        trust_remote_code=True
    )
    tokenizer = getattr(processor, "tokenizer", processor)
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model_loader: Any = Qwen3VLForConditionalGeneration
    if model_loader is None:
        print("[WARN] Falling back to AutoModelForVision2Seq for current transformers version.")
        model_loader = AutoModelForVision2Seq

    base_model = model_loader.from_pretrained(args.base_model, **load_kwargs)

    if not os.path.exists(args.adapter_path):
        raise FileNotFoundError(f"Adapter path not found: {args.adapter_path}")

    print(f"[INFO] Loading adapter: {args.adapter_path}")
    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    model.eval()

    return model, processor


def preprocess_video(video_path: str, max_frames: int = 5) -> List[Dict[str, Any]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise ValueError(f"Video has no readable frames: {video_path}")

    step = max(1, total_frames // max_frames)

    samples = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % step == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            samples.append({
                "image": Image.fromarray(frame_rgb),
                "timestamp": frame_count
            })

            if len(samples) >= max_frames:
                break

        frame_count += 1

    cap.release()
    return samples


def extract_json_list(text: str) -> List[str]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```json\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^```\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    start_idx = cleaned.find("[")
    end_idx = cleaned.rfind("]")

    if start_idx == -1 or end_idx == -1 or end_idx < start_idx:
        return []

    candidate = cleaned[start_idx:end_idx + 1]

    try:
        data = json.loads(candidate)
        if isinstance(data, list):
            return [str(item).strip() for item in data if str(item).strip()]
    except Exception:
        return []

    return []


def detect_objects(
    model: Any,
    processor: Any,
    video_path: str,
    max_frames: int = 5,
    max_new_tokens: int = 256,
    temperature: float = 0.2,
) -> Dict[str, Any]:
    samples = preprocess_video(video_path, max_frames=max_frames)
    detected_objects_all = set()
    frame_results = []

    print(f"\n🎬 Analyzing: {Path(video_path).name}")
    print(f"[INFO] Sampled {len(samples)} frames")

    device = next(model.parameters()).device

    for i, sample in enumerate(samples):
        image = sample["image"]
        timestamp = sample["timestamp"]

        content = [
            {"type": "text", "text": INFER_PROMPT},
            {"type": "image", "image": image, "max_pixels": 1474560},
        ]
        messages = [{"role": "user", "content": content}]

        text_input = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = processor(
            text=[text_input],
            images=[image],
            padding=True,
            return_tensors="pt",
        )
        inputs.pop("token_type_ids", None)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=(temperature > 0),
                # 이런 방법도 고려해볼수 있다.
                #temperature=0.0,
                #do_sample=False, 
            )

        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        response = processor.decode(generated_ids, skip_special_tokens=True)
        obj_list = extract_json_list(response)

        detected_objects_all.update(obj_list)

        frame_results.append({
            "frame_idx": i,
            "timestamp_frame": int(timestamp),
            "detected": obj_list,
            "raw_response": response
        })

        print(f"  - frame {i} (frame_id={timestamp}): {obj_list}")

        del inputs, outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        "unique_objects": sorted(list(detected_objects_all)),
        "samples": frame_results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage1 object extraction inspection")
    parser.add_argument("--base-model", type=str, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter-path", type=str, default=DEFAULT_ADAPTER_PATH)
    parser.add_argument("--video-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="results/object_eval")
    parser.add_argument("--precision", choices=["4bit", "bf16", "8bit"], default="4bit")
    parser.add_argument("--max-frames", type=int, default=5)
    parser.add_argument("--max-videos", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.2)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("[SYSTEM] Initializing inference pipeline...")
    print(f"[SYSTEM] Base model   : {args.base_model}")
    print(f"[SYSTEM] Adapter path : {args.adapter_path}")
    print(f"[SYSTEM] Video dir    : {args.video_dir}")
    print(f"[SYSTEM] Output dir   : {args.output_dir}")

    model, processor = load_model_and_processor(args)

    video_files = sorted(str(path) for path in Path(args.video_dir).rglob("*.mp4"))

    if not video_files:
        raise FileNotFoundError(f"No .mp4 files found in: {args.video_dir}")

    video_files = video_files[:args.max_videos]
    print(f"[SYSTEM] Found {len(video_files)} video(s) to analyze.")

    final_report: Dict[str, Any] = {
        "base_model": args.base_model,
        "adapter_path": args.adapter_path,
        "prompt": INFER_PROMPT,
        "videos_analyzed": [],
    }

    for idx, v_path in enumerate(video_files, start=1):
        v_name = os.path.relpath(v_path, args.video_dir)
        print(f"\n{'=' * 70}")
        print(f"[{idx}/{len(video_files)}] Processing: {v_name}")

        try:
            res = detect_objects(
                model=model,
                processor=processor,
                video_path=v_path,
                max_frames=args.max_frames,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )

            video_result = {
                "filename": v_name,
                "total_detected": len(res["unique_objects"]),
                "unique_objects": res["unique_objects"],
                "frame_results": res["samples"],
            }

            final_report["videos_analyzed"].append(video_result)

            print(f"✅ Unique Objects : {res['unique_objects']}")

            save_path = os.path.join(
                args.output_dir,
                f"{Path(v_name).stem}_report.json"
            )
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(video_result, f, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"❌ Error processing {v_name}: {e}")
            final_report["videos_analyzed"].append({
                "filename": v_name,
                "error": str(e),
            })

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    report_path = os.path.join(args.output_dir, "final_summary.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Done. Results saved to: {report_path}")


if __name__ == "__main__":
    main()
