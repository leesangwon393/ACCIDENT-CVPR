import glob
import importlib.util
import json
import os
import re
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import transformers
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, BitsAndBytesConfig
from transformers.utils import import_utils

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


Qwen3VLForConditionalGeneration = getattr(transformers, "Qwen3VLForConditionalGeneration")

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
PROCESSOR_ID = MODEL_ID
VIDEO_DIR = "./raw/accident/videos"
METADATA_CSV = "./raw/accident/test_metadata.csv"
OUTPUT_CSV = "./submission.csv"
DEBUG_CSV = "./debug_results.csv"

# ── VRAM 최적화 (24GB GPU, 4-bit 양자화 모델 기준 추론 메모리 절감) ──
# 전략: stage1은 저 fps overview로 시간 탐지 → stage2/3은 3-frame montage로 공간/유형 정밀화
MAX_FRAMES = 64
TARGET_FPS = 2.0
MIN_FRAMES = 8
MAX_PIXELS = 512 * 768
MIN_PIXELS = 28 * 28
MAX_NEW_TOKENS = 220
INFERENCE_BATCH_SIZE = 2

VALID_TYPES = ["rear-end", "head-on", "sideswipe", "t-bone", "single"]
STAGE2_CONTEXT_SEC = 0.25



def to_float_or(value: Any, default: float) -> float:
    return float(default if value is None else value)



def to_int_or(value: Any, default: int) -> int:
    return int(default if value is None else value)


SYSTEM_PROMPT = (
    "You are a traffic-accident video analyst. "
    "Assume each input contains one primary accident. "
    "Your job is to localize ONLY the first primary collision. "
    "Definitions: accident_time is the time in seconds of the first visible physical contact that initiates the primary accident. "
    "collision_point is the normalized (x, y) midpoint of the first-contact region in the contact frame, not the center of a vehicle and not the final resting position. "
    "collision_type must be exactly one of [head-on, rear-end, sideswipe, single, t-bone]. "
    "Use initial impact geometry and motion BEFORE contact, not aftermath. "
    "If multiple impacts occur, use the earliest primary impact. "
    "Ignore smoke, debris, subtitles, logos, camera shake, and emergency response after impact. "
    "Keep reasoning internal. Return JSON only."
)

TIME_PROMPT_TEMPLATE = """The following {n_frames} overview frames are sampled from a {duration:.1f}-second dashcam video.
Each frame is labeled with its exact timestamp [t=X.Xs].

Task:
Find the first visible physical contact that initiates the primary accident.

Instructions:
- Use the earliest primary impact only.
- Focus on first visible physical contact, not peak deformation, spin, smoke, debris, or aftermath.
- Do NOT choose a near miss, hard braking, steering, or pre-impact overlap caused by perspective.
- If multiple impacts occur, use the earliest primary impact.
- Every video contains one crash, so make the best grounded estimate from the shown frames.
- accident_time MUST match one of the shown timestamps or be interpolated between two adjacent timestamps.

Return JSON only:
{{
  "accident_time": <seconds from start where first visible physical contact occurs, between 0.0 and {duration:.1f}>,
  "confidence": <float in [0,1]>,
  "reasoning": <one short sentence>
}}
"""

LOCATION_PROMPT_TEMPLATE = """This montage corresponds to the first primary collision at approximately {accident_time:.2f}s.
It has 3 panels from left to right:
- LEFT: pre-impact
- CENTER: contact frame
- RIGHT: post-impact

Task:
Locate the midpoint of the FIRST-CONTACT region.

Definitions:
- collision_point is the normalized (x, y) midpoint of the touching region in the CENTER panel.
- This is NOT the center of a vehicle.
- This is NOT the final resting position.

Instructions:
- Identify the FIRST visible physical contact region in the CENTER panel.
- Use LEFT and RIGHT only to resolve blur, occlusion, or trajectory ambiguity.
- Return the midpoint of the touching region, not the vehicle center.
- Use impact geometry, not aftermath.

Return JSON only:
{{
  "center_x": <normalized x in [0,1] for the CENTER panel only>,
  "center_y": <normalized y in [0,1] for the CENTER panel only>,
  "confidence": <float in [0,1]>,
  "reasoning": <one short sentence>
}}
"""

TYPE_PROMPT_TEMPLATE = """This montage corresponds to the first primary collision at approximately {accident_time:.2f}s.
It has 3 panels from left to right:
- LEFT: pre-impact
- CENTER: contact frame
- RIGHT: post-impact

Task:
Classify the first primary collision type.

Choose exactly one:
[head-on, rear-end, sideswipe, single, t-bone]

Collision type definitions:
- head-on: front-to-front impact between two moving vehicles approaching each other.
- rear-end: the front of a following vehicle hits the rear of a leading vehicle traveling in roughly the same direction.
- sideswipe: side-to-side contact or scraping, usually with lateral overlap; not a perpendicular front-to-side strike.
- single: a vehicle first impacts a fixed object or loses control without first colliding with another vehicle.
- t-bone: the front of one vehicle strikes the side of another vehicle at roughly perpendicular directions.

Decision rules:
- Use the initial impact geometry and motion BEFORE contact, not aftermath.
- Use LEFT for approach direction, CENTER for first visible contact, and RIGHT only as supporting evidence.
- perpendicular front-to-side impact is t-bone, not sideswipe.
- same-direction front-to-rear impact is rear-end, not head-on.
- if another vehicle is the first impacted object, do not output single.

Return JSON only:
{{
  "type": <"rear-end"|"head-on"|"sideswipe"|"t-bone"|"single">,
  "confidence": <float in [0,1]>,
  "reasoning": <one short sentence>
}}
"""


@dataclass
class Prediction:
    path: str
    accident_time: float
    center_x: float
    center_y: float
    type: str
    confidence: float
    reasoning: str
    raw: str
    method: str
    fallback_used: bool
    issues: str


# ═══════════════════════════════════════════════════════
#  메타데이터 로드
# ═══════════════════════════════════════════════════════

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



def compute_video_fps(duration: float, no_frames: int, height: int, width: int) -> float:
    if duration <= 0:
        return TARGET_FPS

    fps = TARGET_FPS
    if duration * fps > MAX_FRAMES:
        fps = MAX_FRAMES / duration
    if duration * fps < MIN_FRAMES:
        fps = min(MIN_FRAMES / duration, 4.0)

    fps = max(0.5, min(4.0, fps))
    return round(fps, 2)


def format_gib(num_bytes: int) -> str:
    return f"{num_bytes / (1024 ** 3):.2f} GiB"


def pick_best_cuda_device() -> str:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    best_index = 0
    best_free_bytes = -1
    for index in range(torch.cuda.device_count()):
        free_bytes, _ = torch.cuda.mem_get_info(index)
        if free_bytes > best_free_bytes:
            best_index = index
            best_free_bytes = free_bytes

    return f"cuda:{best_index}"


def resolve_device_settings() -> Tuple[str, Any, torch.dtype, Optional[BitsAndBytesConfig]]:
    request = os.environ.get("BASEMODEL_DEVICE", "auto").strip().lower()

    if request == "cpu":
        return request, "cpu", torch.float32, None

    if request in {"cuda-best", "gpu-best", "gpu-free"}:
        selected_device = pick_best_cuda_device()
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        return selected_device, {"": selected_device}, torch.bfloat16, quantization_config

    if request in {"cuda", "gpu"}:
        request = "cuda:0"

    if request.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError("BASEMODEL_DEVICE requested CUDA, but CUDA is not available")

        try:
            device_index = int(request.split(":", 1)[1])
        except ValueError as exc:
            raise ValueError(f"Invalid BASEMODEL_DEVICE value: {request}") from exc

        if device_index < 0 or device_index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device index out of range: {request}")

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        return request, {"": request}, torch.bfloat16, quantization_config

    if request != "auto":
        raise ValueError(f"Unsupported BASEMODEL_DEVICE value: {request}")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    return request, "auto", torch.bfloat16, quantization_config


# ═══════════════════════════════════════════════════════
#  모델 로드 & 추론
# ═══════════════════════════════════════════════════════

def load_model() -> Tuple[Any, Any]:
    if getattr(import_utils, "is_flash_attn_3_available", lambda: False)():
        attn_implementation = "flash_attention_3"
    elif import_utils.is_flash_attn_2_available():
        attn_implementation = "flash_attention_2"
    else:
        attn_implementation = "sdpa"

    selected_device, device_map, model_dtype, quantization_config = resolve_device_settings()

    if selected_device.startswith("cuda:"):
        device_index = int(selected_device.split(":", 1)[1])
        free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
        print(
            f"[INFO] Selected {selected_device} "
            f"({format_gib(free_bytes)} free / {format_gib(total_bytes)} total)"
        )

    print(f"[INFO] Loading {MODEL_ID} on {selected_device} ...")
    print(f"[INFO] Attention backend: {attn_implementation}")
    load_kwargs: Dict[str, Any] = {
        "dtype": model_dtype,
        "device_map": device_map,
        "attn_implementation": attn_implementation,
    }
    if quantization_config is not None:
        load_kwargs["quantization_config"] = quantization_config

    model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL_ID, **load_kwargs)
    processor = AutoProcessor.from_pretrained(PROCESSOR_ID)
    processor.tokenizer.padding_side = "left"
    model.eval()
    print("[INFO] Model ready.\n")
    return model, processor



def sample_frames_with_timestamps(video_path: str, fps: float, duration: float) -> List[Tuple[Image.Image, float]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    original_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames_actual = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    n_frames = max(MIN_FRAMES, min(MAX_FRAMES, int(duration * fps)))
    frame_indices = np.linspace(0, max(total_frames_actual - 1, 0), n_frames, dtype=int)

    results = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        max_side = 640
        w, h = pil_img.size
        if max(w, h) > max_side:
            scale = max_side / max(w, h)
            pil_img = pil_img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        timestamp = (idx / max(total_frames_actual - 1, 1)) * duration
        results.append((pil_img, round(timestamp, 1)))

    cap.release()
    return results



def build_time_messages(video_path: str, fps: float, duration: float) -> Tuple[List[Dict[str, Any]], int]:
    frames_with_ts = sample_frames_with_timestamps(video_path, fps, duration)
    n_frames = len(frames_with_ts)
    user_prompt = TIME_PROMPT_TEMPLATE.format(duration=duration, n_frames=n_frames)

    content_parts: List[Dict[str, Any]] = []
    for pil_img, ts in frames_with_ts:
        content_parts.append({"type": "text", "text": f"[t={ts}s]"})
        content_parts.append({"type": "image", "image": pil_img, "max_pixels": MAX_PIXELS, "min_pixels": MIN_PIXELS})

    content_parts.append({"type": "text", "text": user_prompt})
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content_parts},
    ]
    return messages, n_frames



def build_single_image_messages(image: Image.Image, user_prompt: str) -> List[Dict[str, Any]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image, "max_pixels": MAX_PIXELS, "min_pixels": MIN_PIXELS},
                {"type": "text", "text": user_prompt},
            ],
        },
    ]



def generate_raw_batch(
    model: Any,
    processor: Any,
    messages_batch: List[List[Dict[str, Any]]],
) -> List[str]:
    text_inputs = [
        processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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

    trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)]
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


# ═══════════════════════════════════════════════════════
#  출력 파싱 & 후처리
# ═══════════════════════════════════════════════════════

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



def coerce_time(value: Any, duration: float) -> float:
    try:
        candidate = float(value)
    except (TypeError, ValueError):
        return duration / 2
    return max(0.0, min(duration, candidate))



def extract_stage_json(raw: str) -> Tuple[Dict[str, Any], str]:
    clean = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    try:
        return extract_json(clean), clean
    except (ValueError, json.JSONDecodeError):
        return {}, clean



def get_video_frame_count_and_fps(video_path: str) -> Tuple[int, float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count, fps



def read_frame_at_index(video_path: str, frame_index: int) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ret, frame = cap.read()
    if not ret or frame is None:
        for delta in range(1, 6):
            for candidate in (frame_index - delta, frame_index + delta):
                if candidate < 0:
                    continue
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(candidate))
                ret, frame = cap.read()
                if ret and frame is not None:
                    cap.release()
                    return frame
        cap.release()
        raise RuntimeError(f"Failed to read frame near index {frame_index}")

    cap.release()
    return frame



def draw_panel_label(frame: np.ndarray, text: str) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    bar_h = max(28, h // 14)
    cv2.rectangle(out, (0, 0), (w, bar_h), (0, 0, 0), thickness=-1)
    cv2.putText(
        out,
        text,
        (10, int(bar_h * 0.72)),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(0.6, min(w, h) / 600.0),
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out



def resize_to_same_height(frames: List[np.ndarray], target_height: int) -> List[np.ndarray]:
    resized = []
    for frame in frames:
        h, w = frame.shape[:2]
        if h == target_height:
            resized.append(frame)
            continue
        new_w = max(1, int(round(w * (target_height / h))))
        resized.append(cv2.resize(frame, (new_w, target_height), interpolation=cv2.INTER_AREA))
    return resized



def build_context_montage_image(
    video_path: str,
    accident_time: float,
    duration: float,
    no_frames: int,
) -> Image.Image:
    frame_count, actual_fps = get_video_frame_count_and_fps(video_path)

    if duration > 0 and no_frames > 0:
        meta_fps = no_frames / duration
        fps = meta_fps if meta_fps > 0 else actual_fps
    else:
        fps = actual_fps

    contact_idx = int(round(accident_time * fps))
    contact_idx = max(0, min(frame_count - 1, contact_idx))
    offset = max(1, int(round(fps * STAGE2_CONTEXT_SEC)))
    pre_idx = max(0, contact_idx - offset)
    post_idx = min(frame_count - 1, contact_idx + offset)

    pre_frame = draw_panel_label(read_frame_at_index(video_path, pre_idx), "PRE-IMPACT")
    contact_frame = draw_panel_label(read_frame_at_index(video_path, contact_idx), "CONTACT")
    post_frame = draw_panel_label(read_frame_at_index(video_path, post_idx), "POST-IMPACT")

    target_height = min(pre_frame.shape[0], contact_frame.shape[0], post_frame.shape[0])
    pre_frame, contact_frame, post_frame = resize_to_same_height(
        [pre_frame, contact_frame, post_frame],
        target_height=target_height,
    )
    montage = cv2.hconcat([pre_frame, contact_frame, post_frame])
    montage_rgb = cv2.cvtColor(montage, cv2.COLOR_BGR2RGB)
    return Image.fromarray(montage_rgb)



def build_prediction(
    video_path: str,
    duration: float,
    time_raw: str,
    location_raw: str,
    type_raw: str,
    fallback_used: bool = False,
) -> Prediction:
    time_data, time_clean = extract_stage_json(time_raw)
    loc_data, loc_clean = extract_stage_json(location_raw)
    type_data, type_clean = extract_stage_json(type_raw)

    accident_time = coerce_time(time_data.get("accident_time"), duration)
    center_x = float(loc_data.get("center_x", 0.5) or 0.5)
    center_y = float(loc_data.get("center_y", 0.5) or 0.5)
    confidence_candidates = [
        float(time_data.get("confidence", 0.0) or 0.0),
        float(loc_data.get("confidence", 0.0) or 0.0),
        float(type_data.get("confidence", 0.0) or 0.0),
    ]
    confidence = max(0.0, min(1.0, sum(confidence_candidates) / len(confidence_candidates)))
    collision_type = normalize_type(type_data.get("type", "single"))

    center_x = max(0.0, min(1.0, center_x))
    center_y = max(0.0, min(1.0, center_y))

    issues: List[str] = []
    if not time_data:
        issues.append("time_parse_failed")
    if not loc_data:
        issues.append("location_parse_failed")
    if not type_data:
        issues.append("type_parse_failed")
    if abs(center_x - 0.5) < 1e-6 and abs(center_y - 0.5) < 1e-6:
        issues.append("center_default")
    if confidence < 0.2:
        issues.append("low_confidence")

    relative_base = Path(VIDEO_DIR).parent.resolve()
    resolved_path = Path(video_path).resolve()
    try:
        relative_path = resolved_path.relative_to(relative_base)
    except ValueError:
        relative_path = resolved_path.name

    reasoning_parts = [
        str(time_data.get("reasoning", "")).strip(),
        str(loc_data.get("reasoning", "")).strip(),
        str(type_data.get("reasoning", "")).strip(),
    ]
    reasoning = " | ".join([part for part in reasoning_parts if part])[:300]
    raw = json.dumps(
        {
            "time_raw": time_clean,
            "location_raw": loc_clean,
            "type_raw": type_clean,
        },
        ensure_ascii=False,
    )

    return Prediction(
        path=str(relative_path),
        accident_time=accident_time,
        center_x=center_x,
        center_y=center_y,
        type=collision_type,
        confidence=confidence,
        reasoning=reasoning,
        raw=raw,
        method="three_stage_montage",
        fallback_used=fallback_used,
        issues=",".join(issues),
    )


# ═══════════════════════════════════════════════════════
#  메인 추론 파이프라인
# ═══════════════════════════════════════════════════════

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
        # Stage 1: overview frames -> accident_time
        time_messages, sampled_frames = build_time_messages(video_path, fps, duration)
        time_raw = generate_raw(model, processor, time_messages)
        time_data, _ = extract_stage_json(time_raw)
        accident_time = coerce_time(time_data.get("accident_time"), duration)

        # Stage 2: pre/contact/post montage -> collision point
        montage_image = build_context_montage_image(video_path, accident_time, duration, no_frames)
        location_messages = build_single_image_messages(
            montage_image,
            LOCATION_PROMPT_TEMPLATE.format(accident_time=accident_time),
        )
        location_raw = generate_raw(model, processor, location_messages)

        # Stage 3: same montage -> collision type
        type_messages = build_single_image_messages(
            montage_image,
            TYPE_PROMPT_TEMPLATE.format(accident_time=accident_time),
        )
        type_raw = generate_raw(model, processor, type_messages)

        prediction = build_prediction(video_path, duration, time_raw, location_raw, type_raw)
    except Exception as exc:
        prediction = Prediction(
            path=str(Path(video_path).name),
            accident_time=duration / 2,
            center_x=0.5,
            center_y=0.5,
            type="single",
            confidence=0.0,
            reasoning=f"three_stage_error: {str(exc)[:160]}",
            raw="",
            method="three_stage_error",
            fallback_used=False,
            issues="three_stage_error,parse_failed",
        )

    return asdict(prediction)



def infer_videos_batch(
    model: Any,
    processor: Any,
    video_paths: List[str],
    metas: List[Optional[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    # 3-stage는 영상별 montage 생성/재추론이 필요해서 per-video 처리 유지
    return [infer_video(model, processor, video_path, meta=meta) for video_path, meta in zip(video_paths, metas)]



def run_all(model: Any, processor: Any, video_dir: str) -> pd.DataFrame:
    metadata = load_metadata(METADATA_CSV)
    print(f"[INFO] Loaded metadata for {len(metadata)} videos")

    paths = sorted(glob.glob(os.path.join(video_dir, "*.mp4")))
    if not paths:
        raise FileNotFoundError(f"No .mp4 files found in: {video_dir}")

    print(f"[INFO] {len(paths)} videos found. Starting...\n")
    records: List[Dict[str, Any]] = []
    progress_bar = tqdm(total=len(paths), desc="Video inference", unit="video") if tqdm is not None else None

    try:
        for batch_start in range(0, len(paths), INFERENCE_BATCH_SIZE):
            batch_paths = paths[batch_start: batch_start + INFERENCE_BATCH_SIZE]
            batch_metas: List[Optional[Dict[str, Any]]] = []

            for offset, video_path in enumerate(batch_paths, start=batch_start + 1):
                video_name = Path(video_path).name
                meta_key = f"videos/{video_name}"
                meta = metadata.get(meta_key)
                batch_metas.append(meta)

                if meta:
                    fps = compute_video_fps(meta["duration"], meta["no_frames"], meta["height"], meta["width"])
                    est_frames = int(meta["duration"] * fps)
                    print(f"[{offset}/{len(paths)}] {video_name} | {meta['duration']:.1f}s {meta['width']}x{meta['height']} → stage1 fps={fps} (~{est_frames}f)")
                else:
                    print(f"[{offset}/{len(paths)}] {video_name} | no metadata")

            batch_records = infer_videos_batch(model, processor, batch_paths, batch_metas)
            for record in batch_records:
                print(
                    f"  -> {record['accident_time']:.2f}s | "
                    f"({record['center_x']:.3f}, {record['center_y']:.3f}) | "
                    f"{record['type']} | conf={record['confidence']:.2f} | {record['method']}"
                )
                records.append(record)

            if progress_bar is not None:
                progress_bar.update(len(batch_paths))
    finally:
        if progress_bar is not None:
            progress_bar.close()

    return pd.DataFrame(records)



def save_submission(df: pd.DataFrame, path: str) -> None:
    df.to_csv(DEBUG_CSV, index=False)
    print(f"[INFO] debug saved -> {DEBUG_CSV}")
    submission = df[["path", "accident_time", "center_x", "center_y", "type"]].copy()
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
