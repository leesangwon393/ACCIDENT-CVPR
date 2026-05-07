"""Shared Qwen3-VL inference helpers for this repo.

This module is imported by the RB-FT stage scripts and assumes a Qwen/Qwen3-VL
runtime with `transformers`, `qwen_vl_utils`, and optional quantization
support through `bitsandbytes`.
"""

import glob
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

cv2 = __import__("cv2")
np = __import__("numpy")
pd = __import__("pandas")
torch = __import__("torch")
transformers = __import__("transformers")
pil_image = __import__("PIL.Image", fromlist=["Image", "fromarray"])
qwen_vl_utils = __import__("qwen_vl_utils")
process_vision_info = getattr(qwen_vl_utils, "process_vision_info")
AutoProcessor = getattr(
    __import__("transformers", fromlist=["AutoProcessor"]), "AutoProcessor"
)
BitsAndBytesConfig = getattr(
    __import__("transformers", fromlist=["BitsAndBytesConfig"]), "BitsAndBytesConfig"
)
import_utils = __import__("transformers.utils.import_utils", fromlist=["*"])

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


Qwen3VLForConditionalGeneration = getattr(
    transformers, "Qwen3VLForConditionalGeneration", None
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs" / "inference"

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
PROCESSOR_ID = MODEL_ID
VIDEO_DIR = os.environ.get("VIDEO_DIR", "./raw/accident/videos")
METADATA_CSV = os.environ.get("METADATA_CSV", "./raw/accident/test_metadata.csv")
OUTPUT_CSV = os.environ.get("OUTPUT_CSV", str(DEFAULT_OUTPUT_DIR / "submission.csv"))
DEBUG_CSV = os.environ.get("DEBUG_CSV", str(DEFAULT_OUTPUT_DIR / "debug_results.csv"))

# -- VRAM optimization (24GB GPU, 4-bit quantized model inference memory reduction) --
# Strategy: save tokens with fps=2 and invest the budget in resolution.

MAX_FRAMES = 64
TARGET_FPS = 2.0
MIN_FRAMES = 8
MAX_PIXELS = 480 * 480
MAX_SIDE = 480
MIN_PIXELS = 28 * 28
MAX_NEW_TOKENS = 220
INFERENCE_BATCH_SIZE = 2

VALID_TYPES = ["rear-end", "head-on", "sideswipe", "t-bone", "single"]


def to_float_or(value: Any, default: float) -> float:
    return float(default if value is None else value)


def to_int_or(value: Any, default: int) -> int:
    return int(default if value is None else value)


SYSTEM_PROMPT = (
    "You are a traffic crash analyst. The video frames are overlaid with semi-transparent semantic masks to highlight key objects: "
    "- BLUE: Vehicles (active participants) "
    "- RED: Fixed objects (poles, guardrails, barriers, walls) "
    "- GREEN: Infrastructure (road surface, bridges) "
    "Your task is to localize the first physical impact moment, estimate the impact point, and classify the crash type using both the visual details and these color cues."
)

USER_PROMPT_TEMPLATE = """The following {n_frames} frames are sampled from a {duration:.1f}-second traffic surveillance video.
Each frame is labeled with its exact timestamp [t=X.Xs].

Return exactly one JSON object with these keys:
{{
  "type": <"rear-end"|"head-on"|"sideswipe"|"t-bone"|"single">,
  "accident_time": <seconds from start where first physical impact occurs, between 0.0 and {duration:.1f}>,
  "center_x": <normalized x in [0,1]>,
  "center_y": <normalized y in [0,1]>
}}

Crash type definitions:
- single   : BLUE (vehicle) vs RED (fixed object like pole or guardrail)
- rear-end : BLUE vs BLUE (front hitting rear)
- head-on  : BLUE vs BLUE (front hitting front)
- t-bone   : BLUE vs BLUE (perpendicular contact)
- sideswipe: BLUE vs BLUE (parallel side contact)

Rules:
- Every video contains exactly one crash; do not say no crash, unknown, or none.
- A crash occurs when a BLUE region first contacts another BLUE or RED region.
- Use the first physical impact, not the aftermath.
- accident_time MUST match one of the [t=X.Xs] timestamps shown, or interpolate between two adjacent ones.
- center_x and center_y should pinpoint the exact intersection of the colored regions.
- Use the actual impact point, not the frame center.
- Choose exactly one crash type from the allowed list.
- Output JSON only, with no markdown fences or extra text.
"""


@dataclass
class Prediction:
    path: str
    accident_time: float
    center_x: float
    center_y: float
    type: str


def load_metadata(csv_path: str) -> Dict[str, Dict[str, Any]]:
    if not os.path.exists(csv_path):
        print(f"[WARN] Metadata CSV not found: {csv_path}")
        return {}
    df = pd.read_csv(csv_path)
    meta = {}
    for _, row in df.iterrows():
        duration = row.get("duration")
        no_frames = row.get("no_frames")
        height = row.get("height")
        width = row.get("width")
        quality = row.get("quality")
        meta[row["path"]] = {
            "duration": to_float_or(duration, 10.0),
            "no_frames": to_int_or(no_frames, 0),
            "height": to_int_or(height, 720),
            "width": to_int_or(width, 1280),
            "quality": str(quality if quality is not None else "Fine"),
        }
    return meta


def compute_video_fps(
    duration: float, no_frames: int, height: int, width: int
) -> float:
    if duration <= 0:
        return TARGET_FPS

    fps = TARGET_FPS
    if duration * fps > MAX_FRAMES:
        fps = MAX_FRAMES / duration
    if duration * fps < MIN_FRAMES:
        fps = min(MIN_FRAMES / duration, 4.0)

    fps = max(0.5, min(4.0, fps))
    return round(fps, 2)


def load_model() -> Tuple[Any, Any]:
    if Qwen3VLForConditionalGeneration is None:
        raise RuntimeError(
            "Qwen3VLForConditionalGeneration is unavailable in the installed transformers version."
        )

    if getattr(import_utils, "is_flash_attn_3_available", lambda: False)():
        attn_implementation = "flash_attention_3"
    elif import_utils.is_flash_attn_2_available():
        attn_implementation = "flash_attention_2"
    else:
        attn_implementation = "sdpa"

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    print(f"[INFO] Loading {MODEL_ID} in 4-bit ...")
    print(f"[INFO] Attention backend: {attn_implementation}")
    device_map: Any = "auto"

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map=device_map,
        attn_implementation=attn_implementation,
        quantization_config=quantization_config,
    )

    lora_path = os.environ.get("LORA_PATH")
    if lora_path:
        print(f"[INFO] Loading LoRA adapter from {lora_path}")
        peft = __import__("peft")
        peft_model = getattr(peft, "PeftModel")
        model = peft_model.from_pretrained(model, lora_path)

    processor = AutoProcessor.from_pretrained(PROCESSOR_ID)
    processor.tokenizer.padding_side = "left"
    model.eval()
    print("[INFO] Model ready.\n")
    return model, processor


def sample_frames_with_timestamps(
    video_path: str,
    fps: float,
    duration: float,
    max_frames: int = MAX_FRAMES,
    max_side: int = MAX_SIDE,
) -> List[Tuple[Any, float]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    original_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames_actual = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    actual_duration = (
        total_frames_actual / original_fps if original_fps > 0 else duration
    )

    n_frames = max(MIN_FRAMES, min(max_frames, int(duration * fps)))
    frame_indices = np.linspace(0, total_frames_actual - 1, n_frames, dtype=int)

    results = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = pil_image.fromarray(frame_rgb)
        w, h = pil_img.size
        if max(w, h) > max_side:
            scale = max_side / max(w, h)
            pil_img = pil_img.resize(
                (int(w * scale), int(h * scale)), pil_image.LANCZOS
            )
        timestamp = (idx / max(total_frames_actual - 1, 1)) * duration
        results.append((pil_img, round(timestamp, 1)))

    cap.release()
    return results


def build_frame_messages(
    video_path: str,
    fps: float,
    duration: float,
    max_frames: int = MAX_FRAMES,
    max_side: int = MAX_SIDE,
) -> Tuple[List[Dict[str, Any]], int]:
    frames_with_ts = sample_frames_with_timestamps(
        video_path, fps, duration, max_frames=max_frames, max_side=max_side
    )
    n_frames = len(frames_with_ts)

    user_prompt = USER_PROMPT_TEMPLATE.format(
        duration=duration, fps=fps, n_frames=n_frames
    )

    content_parts: List[Dict[str, Any]] = []
    for pil_img, ts in frames_with_ts:
        content_parts.append({"type": "text", "text": f"[t={ts}s]"})
        content_parts.append(
            {
                "type": "image",
                "image": pil_img,
                "max_pixels": MAX_PIXELS,
                "min_pixels": MIN_PIXELS,
            }
        )

    content_parts.append({"type": "text", "text": user_prompt})

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content_parts},
    ]
    return messages, n_frames


def generate_raw_batch(
    model: Any,
    processor: Any,
    messages_batch: List[List[Dict[str, Any]]],
) -> List[str]:
    text_inputs = [
        processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        for messages in messages_batch
    ]
    vision_result = process_vision_info(messages_batch)
    image_inputs = vision_result[0]
    video_inputs = vision_result[1]
    processor_kwargs: Dict[str, Any] = {
        "text": text_inputs,
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
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=0,
            top_p=None,
        )

    trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]
    return [
        text.strip()
        for text in processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    ]


def generate_raw(model: Any, processor: Any, messages: List[Dict[str, Any]]) -> str:
    return generate_raw_batch(model, processor, [messages])[0]


def extract_json(clean: str) -> Dict[str, Any]:
    match = re.search(r"```json\s*(\{.*?\})\s*```", clean, re.DOTALL)
    if not match:
        match = re.search(r"(\{.*?\})", clean, re.DOTALL)
    if not match:
        raise ValueError("json not found")
    return json.loads(match.group(1))


def normalize_type(value: Any) -> str:
    if not isinstance(value, str):
        return "single"
    text = value.strip().lower()
    aliases = {
        "rear end": "rear-end",
        "rear_end": "rear-end",
        "rear-ended": "rear-end",
        "head on": "head-on",
        "head_on": "head-on",
        "side swipe": "sideswipe",
        "side-swipe": "sideswipe",
        "tbone": "t-bone",
        "t bone": "t-bone",
    }
    text = aliases.get(text, text)
    return text if text in VALID_TYPES else "single"


def coerce_time(value: Any, duration: float, sampled_frames: int) -> float:
    try:
        candidate = float(value)
    except (TypeError, ValueError):
        return duration / 2

    return max(0.0, min(duration, candidate))


def parse_output(
    raw: str, video_path: str, duration: float, method: str, sampled_frames: int
) -> Prediction:
    clean = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    try:
        data = extract_json(clean)
    except (ValueError, json.JSONDecodeError):
        data = {}

    accident_time = coerce_time(data.get("accident_time"), duration, sampled_frames)
    center_x = float(data.get("center_x", 0.5) or 0.5)
    center_y = float(data.get("center_y", 0.5) or 0.5)
    collision_type = normalize_type(data.get("type", "single"))

    accident_time = max(0.0, min(duration, accident_time))
    center_x = max(0.0, min(1.0, center_x))
    center_y = max(0.0, min(1.0, center_y))

    lowered = clean.lower()
    if (
        "no crash" in lowered
        or "no accident" in lowered
        or "not detectable" in lowered
        or "no physical impact" in lowered
    ):
        accident_time = duration / 2

    relative_base = Path(VIDEO_DIR).parent.resolve()
    resolved_path = Path(video_path).resolve()
    try:
        relative_path = resolved_path.relative_to(relative_base)
    except ValueError:
        relative_path = resolved_path.name

    return Prediction(
        path=str(relative_path),
        accident_time=accident_time,
        center_x=center_x,
        center_y=center_y,
        type=collision_type,
    )


def infer_video(
    model: Any,
    processor: Any,
    video_path: str,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if meta:
        duration = meta["duration"]
        no_frames = meta["no_frames"]
        height = meta["height"]
        width = meta["width"]
    else:
        duration = 10.0
        no_frames = 0
        height = 720
        width = 1280

    fps = compute_video_fps(duration, no_frames, height, width)

    try:
        messages, sampled_frames = build_frame_messages(video_path, fps, duration)
        raw = generate_raw(model, processor, messages)
        prediction = parse_output(raw, video_path, duration, "frames", sampled_frames)
    except Exception as exc:
        prediction = Prediction(
            path=str(Path(video_path).name),
            accident_time=duration / 2,
            center_x=0.5,
            center_y=0.5,
            type="single",
        )

    return asdict(prediction)


def infer_videos_batch(
    model: Any,
    processor: Any,
    video_paths: List[str],
    metas: List[Optional[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    prepared: List[Dict[str, Any]] = []
    messages_batch: List[List[Dict[str, Any]]] = []

    try:
        for video_path, meta in zip(video_paths, metas):
            if meta:
                duration = meta["duration"]
                no_frames = meta["no_frames"]
                height = meta["height"]
                width = meta["width"]
            else:
                duration = 10.0
                no_frames = 0
                height = 720
                width = 1280

            fps = compute_video_fps(duration, no_frames, height, width)
            messages, sampled_frames = build_frame_messages(video_path, fps, duration)
            messages_batch.append(messages)
            prepared.append(
                {
                    "video_path": video_path,
                    "duration": duration,
                    "sampled_frames": sampled_frames,
                }
            )

        raws = generate_raw_batch(model, processor, messages_batch)
        return [
            asdict(
                parse_output(
                    raw,
                    item["video_path"],
                    item["duration"],
                    f"frames_batch{len(video_paths)}",
                    item["sampled_frames"],
                )
            )
            for item, raw in zip(prepared, raws)
        ]
    except Exception:
        return [
            infer_video(model, processor, video_path, meta=meta)
            for video_path, meta in zip(video_paths, metas)
        ]


def run_all(model: Any, processor: Any, video_dir: str) -> Any:
    metadata = load_metadata(METADATA_CSV)
    print(f"[INFO] Loaded metadata for {len(metadata)} videos")

    paths = sorted(glob.glob(os.path.join(video_dir, "*.mp4")))
    if not paths:
        raise FileNotFoundError(f"No .mp4 files found in: {video_dir}")

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    if world_size > 1:
        paths = paths[rank::world_size]
        print(f"[INFO] WORKER {rank}/{world_size} - Processing {len(paths)} videos")

    print(f"[INFO] {len(paths)} videos found. Starting...\n")
    records: List[Dict[str, Any]] = []
    progress_bar = (
        tqdm(total=len(paths), desc="Video inference", unit="video")
        if tqdm is not None
        else None
    )

    try:
        for batch_start in range(0, len(paths), INFERENCE_BATCH_SIZE):
            batch_paths = paths[batch_start : batch_start + INFERENCE_BATCH_SIZE]
            batch_metas: List[Optional[Dict[str, Any]]] = []

            for offset, video_path in enumerate(batch_paths, start=batch_start + 1):
                video_name = Path(video_path).name
                meta_key = f"videos/{video_name}"
                meta = metadata.get(meta_key)
                batch_metas.append(meta)

                if meta:
                    fps = compute_video_fps(
                        meta["duration"],
                        meta["no_frames"],
                        meta["height"],
                        meta["width"],
                    )
                    est_frames = int(meta["duration"] * fps)
                    print(
                        f"[{offset}/{len(paths)}] {video_name} | {meta['duration']:.1f}s {meta['width']}x{meta['height']} -> fps={fps} (~{est_frames}f)"
                    )
                else:
                    print(f"[{offset}/{len(paths)}] {video_name} | no metadata")

            batch_records = infer_videos_batch(
                model, processor, batch_paths, batch_metas
            )
            for record in batch_records:
                print(
                    f"  -> {record['accident_time']:.2f}s | "
                    f"({record['center_x']:.3f}, {record['center_y']:.3f}) | "
                    f"{record['type']}"
                )
                records.append(record)

            if progress_bar is not None:
                progress_bar.update(len(batch_paths))
    finally:
        if progress_bar is not None:
            progress_bar.close()

    return pd.DataFrame(records)


def save_submission(df: Any, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(DEBUG_CSV).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DEBUG_CSV, index=False)
    print(f"[INFO] debug saved -> {DEBUG_CSV}")
    submission = df.copy()
    submission["accident_time"] = submission["accident_time"].astype(float).round(2)
    submission["center_x"] = submission["center_x"].astype(float).round(3)
    submission["center_y"] = submission["center_y"].astype(float).round(3)
    submission.to_csv(path, index=False)
    print(f"[INFO] submission saved -> {path}")
    print(submission.head(10).to_string(index=False))


if __name__ == "__main__":
    model, processor = load_model()
    dataframe = run_all(model, processor, VIDEO_DIR)
    save_submission(dataframe, OUTPUT_CSV)
    print("\nDone")
