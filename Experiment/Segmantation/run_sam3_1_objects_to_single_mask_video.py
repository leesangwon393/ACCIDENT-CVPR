#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import cv2
import numpy as np

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent
LOCAL_SAM3_ROOT = PROJECT_ROOT / "sam3"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "sam3.1" / "sam3.1_multiplex.pt"
DEFAULT_VIDEO = Path(
    "/root/Desktop/workspace/yuyeon/Experiments/16. RB-FT/sim_dataset_aug_fixed3_aug2/videos/single/Town10HD_single_sunset_22_aug2.mp4"
)
DEFAULT_CLASS_YAML = Path(
    "/root/Desktop/workspace/yuyeon/Experiments/16. RB-FT/sim_dataset_aug_fixed3_aug2/annotation_classes.yaml"
)
DEFAULT_RUNS_DIR = PROJECT_ROOT / "runs"

TRAIN_TRAFFIC_V2_LABELS = {
    "vehicle",
    "road",
    "wall",
    "fence",
    "pole",
    "vegetation",
}

TEST_TRAFFIC_V2_LABELS = {
    "vehicle",
    "road",
    "bridge",
    "tunnel",
    "pole",
    "traffic sign",
    "barrier",
    "guardrail",
    "cone",
    "tree",
    "wall",
    "fence",
}

TRAFFIC3_CATEGORY_BY_LABEL = {
    "vehicle": "Vehicle",
    "road": "Background",
    "bridge": "Background",
    "tunnel": "Background",
    "pole": "Obstacle",
    "traffic sign": "Obstacle",
    "barrier": "Obstacle",
    "guardrail": "Obstacle",
    "cone": "Obstacle",
    "tree": "Obstacle",
    "wall": "Obstacle",
    "fence": "Obstacle",
    "vegetation": "Obstacle",
}

TRAFFIC3_CATEGORY_COLORS = {
    "Vehicle": [30, 144, 255],
    "Background": [50, 205, 50],
    "Obstacle": [255, 99, 71],
}

LABEL_ALIASES = {
    "sign": "traffic sign",
    "traffic_sign": "traffic sign",
    "traffic signs": "traffic sign",
    "guard rail": "guardrail",
    "guard-rail": "guardrail",
    "vehicles": "vehicle",
    "roads": "road",
}

if str(LOCAL_SAM3_ROOT) not in sys.path:
    sys.path.insert(0, str(LOCAL_SAM3_ROOT))

build_sam3_predictor = getattr(importlib.import_module("sam3"), "build_sam3_predictor")


@dataclass(frozen=True)
class VideoInfo:
    frame_count: int
    width: int
    height: int
    fps: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run SAM 3.1 on a video using all labels from an annotation yaml and "
            "write a single combined masked output video."
        )
    )
    parser.add_argument("--video-path", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--class-yaml-path", type=Path, default=DEFAULT_CLASS_YAML)
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--prompt-frame-index", type=int, default=0)
    parser.add_argument(
        "--propagation-direction",
        choices=("forward", "backward", "both"),
        default="forward",
    )
    parser.add_argument("--output-prob-thresh", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--max-num-objects", type=int, default=48)
    parser.add_argument("--multiplex-count", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=1008)
    parser.add_argument("--postprocess-batch-size", type=int, default=4)
    parser.add_argument("--batched-grounding-batch-size", type=int, default=4)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--warm-up", action="store_true")
    parser.add_argument("--async-loading-frames", action="store_true", default=False)
    parser.add_argument(
        "--offload-video-to-cpu",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--use-fa3",
        action="store_true",
        default=False,
        help="Enable Flash Attention 3 path. Keep disabled unless flash_attn_interface is installed.",
    )
    parser.add_argument(
        "--label-policy",
        choices=("raw", "train_traffic_v2", "test_traffic_v2"),
        default="raw",
        help="Prompt filtering preset. raw=keep all labels; train/test apply preset allowed-label conditions.",
    )
    parser.add_argument(
        "--category-grouping",
        choices=("none", "traffic3", "auto"),
        default="auto",
        help="Render grouping mode. auto -> traffic3 for train/test policies, otherwise none.",
    )
    parser.add_argument(
        "--allowed-labels",
        nargs="*",
        default=[],
        help="Optional explicit allowed labels (overrides policy preset).",
    )
    parser.add_argument(
        "--exclude-labels",
        nargs="*",
        default=[],
        help="Optional labels to remove after loading all classes from yaml.",
    )
    return parser.parse_args()


def ensure_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def normalize_label(label: str) -> str:
    cleaned = re.sub(r"\s+", " ", label.strip().lower())
    return cleaned


def canonicalize_label(label: str) -> str:
    normalized = normalize_label(label)
    return LABEL_ALIASES.get(normalized, normalized)


def resolve_allowed_labels(label_policy: str, explicit_allowed: Sequence[str]) -> set[str] | None:
    explicit = {canonicalize_label(value) for value in explicit_allowed if value.strip()}
    if explicit:
        return explicit
    if label_policy == "train_traffic_v2":
        return set(TRAIN_TRAFFIC_V2_LABELS)
    if label_policy == "test_traffic_v2":
        return set(TEST_TRAFFIC_V2_LABELS)
    return None


def resolve_grouping_mode(label_policy: str, category_grouping: str) -> str:
    if category_grouping == "auto":
        return "traffic3" if label_policy in {"train_traffic_v2", "test_traffic_v2"} else "none"
    return category_grouping


def load_prompts(
    class_yaml_path: Path,
    exclude_labels: Sequence[str],
    allowed_labels: set[str] | None,
) -> List[str]:
    with class_yaml_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    names = data.get("names")
    if not isinstance(names, dict):
        raise ValueError(f"annotation yaml has no 'names' mapping: {class_yaml_path}")

    excluded = {canonicalize_label(label) for label in exclude_labels if label.strip()}
    prompts: List[str] = []
    seen: set[str] = set()
    for _, value in sorted(names.items(), key=lambda item: int(item[0])):
        raw_label = str(value).strip()
        if not raw_label:
            continue
        label = canonicalize_label(raw_label)
        if label in excluded:
            continue
        if allowed_labels is not None and label not in allowed_labels:
            continue
        if label in seen:
            continue
        prompts.append(label)
        seen.add(label)
    if not prompts:
        raise ValueError("No prompts loaded from annotation yaml after filtering.")
    return prompts


def read_video_info(video_path: Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")
    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
    finally:
        cap.release()

    if frame_count <= 0 or width <= 0 or height <= 0:
        raise RuntimeError(
            f"Invalid video metadata for {video_path}: frames={frame_count}, size={width}x{height}"
        )
    if fps <= 0:
        fps = 20.0
    return VideoInfo(frame_count=frame_count, width=width, height=height, fps=fps)


def build_prompt_category_map(prompts: Sequence[str], grouping_mode: str) -> Dict[str, str]:
    if grouping_mode != "traffic3":
        return {prompt: prompt for prompt in prompts}
    result: Dict[str, str] = {}
    for prompt in prompts:
        result[prompt] = TRAFFIC3_CATEGORY_BY_LABEL.get(prompt, "Obstacle")
    return result


def create_color_map(
    prompts: Sequence[str],
    grouping_mode: str,
    prompt_category_map: Dict[str, str],
) -> Dict[str, List[int]]:
    if grouping_mode == "traffic3":
        return {
            prompt: list(TRAFFIC3_CATEGORY_COLORS[prompt_category_map[prompt]])
            for prompt in prompts
        }
    palette = [
        (255, 99, 71),
        (30, 144, 255),
        (50, 205, 50),
        (255, 215, 0),
        (186, 85, 211),
        (255, 140, 0),
        (0, 206, 209),
        (220, 20, 60),
        (154, 205, 50),
        (255, 105, 180),
        (100, 149, 237),
        (255, 182, 193),
        (64, 224, 208),
        (238, 130, 238),
        (255, 160, 122),
        (0, 191, 255),
        (127, 255, 0),
        (255, 69, 0),
        (72, 209, 204),
        (147, 112, 219),
        (255, 228, 181),
        (60, 179, 113),
        (65, 105, 225),
        (255, 20, 147),
        (176, 196, 222),
        (210, 180, 140),
        (0, 250, 154),
        (218, 112, 214),
        (244, 164, 96),
        (46, 139, 87),
    ]
    return {prompt: list(palette[index % len(palette)]) for index, prompt in enumerate(prompts)}


def normalize_masks(mask_array: Any) -> np.ndarray:
    masks = np.asarray(mask_array)
    if masks.size == 0:
        return np.zeros((0, 0, 0), dtype=bool)
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0, :, :]
    if masks.ndim != 3:
        raise ValueError(f"Unexpected mask shape: {masks.shape}")
    return masks.astype(bool, copy=False)


def initialize_render_buffer(video_path: Path, video_info: VideoInfo, buffer_path: Path) -> np.memmap:
    buffer = np.memmap(
        buffer_path,
        dtype=np.uint8,
        mode="w+",
        shape=(video_info.frame_count, video_info.height, video_info.width, 3),
    )
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video for buffer initialization: {video_path}")

    frame_index = 0
    try:
        while True:
            success, frame = cap.read()
            if not success:
                break
            if frame_index >= video_info.frame_count:
                break
            buffer[frame_index] = frame
            frame_index += 1
    finally:
        cap.release()

    if frame_index != video_info.frame_count:
        raise RuntimeError(
            f"Read {frame_index} frames while metadata reported {video_info.frame_count} frames."
        )
    buffer.flush()
    return buffer


def blend_mask_into_frame(frame: np.ndarray, mask: np.ndarray, color_rgb: Sequence[int], alpha: float) -> None:
    if not mask.any():
        return
    color_bgr = np.array([color_rgb[2], color_rgb[1], color_rgb[0]], dtype=np.float32)
    pixels = frame[mask].astype(np.float32)
    blended = (pixels * (1.0 - alpha)) + (color_bgr * alpha)
    frame[mask] = np.clip(blended, 0, 255).astype(np.uint8)


def build_predictor(args: argparse.Namespace):
    return build_sam3_predictor(
        checkpoint_path=str(args.checkpoint_path),
        version="sam3.1",
        compile=args.compile,
        warm_up=args.warm_up,
        max_num_objects=args.max_num_objects,
        multiplex_count=args.multiplex_count,
        image_size=args.image_size,
        postprocess_batch_size=args.postprocess_batch_size,
        batched_grounding_batch_size=args.batched_grounding_batch_size,
        use_fa3=args.use_fa3,
        async_loading_frames=args.async_loading_frames,
    )


def prompt_stream(
    predictor: Any,
    session_id: str,
    propagation_direction: str,
    output_prob_thresh: float,
) -> Iterable[Dict[str, Any]]:
    request = {
        "type": "propagate_in_video",
        "session_id": session_id,
        "propagation_direction": propagation_direction,
        "output_prob_thresh": output_prob_thresh,
    }
    return predictor.handle_stream_request(request)


def run_prompt(
    predictor: Any,
    session_id: str,
    prompt: str,
    args: argparse.Namespace,
    render_buffer: np.memmap,
    color_map: Dict[str, List[int]],
    prompt_category_map: Dict[str, str],
    frames_with_union_mask: np.ndarray,
) -> Dict[str, Any]:
    start = time.perf_counter()
    predictor.handle_request({"type": "reset_session", "session_id": session_id})

    add_prompt_response = predictor.handle_request(
        {
            "type": "add_prompt",
            "session_id": session_id,
            "frame_index": args.prompt_frame_index,
            "text": prompt,
            "output_prob_thresh": args.output_prob_thresh,
        }
    )
    add_outputs = add_prompt_response.get("outputs", {})
    add_masks = normalize_masks(add_outputs.get("out_binary_masks", np.zeros((0, 0, 0))))
    objects_on_add_prompt_frame = int(add_masks.shape[0])

    processed_frame_indices: set[int] = set()
    frames_with_mask = 0
    max_objects_in_frame = objects_on_add_prompt_frame
    unique_object_ids: set[int] = set(int(value) for value in np.asarray(add_outputs.get("out_obj_ids", []), dtype=np.int64).tolist())

    for item in prompt_stream(
        predictor=predictor,
        session_id=session_id,
        propagation_direction=args.propagation_direction,
        output_prob_thresh=args.output_prob_thresh,
    ):
        frame_index = int(item["frame_index"])
        if frame_index in processed_frame_indices:
            continue

        outputs = item.get("outputs", {})
        masks = normalize_masks(outputs.get("out_binary_masks", np.zeros((0, 0, 0))))
        obj_ids = np.asarray(outputs.get("out_obj_ids", []), dtype=np.int64)
        unique_object_ids.update(int(value) for value in obj_ids.tolist())
        object_count = int(masks.shape[0])
        max_objects_in_frame = max(max_objects_in_frame, object_count)
        processed_frame_indices.add(frame_index)

        if object_count == 0:
            continue

        prompt_union_mask = np.any(masks, axis=0)
        if not prompt_union_mask.any():
            continue

        frames_with_mask += 1
        frames_with_union_mask[frame_index] = True
        blend_mask_into_frame(
            frame=render_buffer[frame_index],
            mask=prompt_union_mask,
            color_rgb=color_map[prompt],
            alpha=args.alpha,
        )

    runtime_sec = round(time.perf_counter() - start, 2)
    return {
        "prompt": prompt,
        "category": prompt_category_map.get(prompt, prompt),
        "objects_on_add_prompt_frame": objects_on_add_prompt_frame,
        "frames_with_mask": frames_with_mask,
        "max_objects_in_frame": max_objects_in_frame,
        "unique_object_ids": sorted(unique_object_ids),
        "total_unique_objects": len(unique_object_ids),
        "runtime_sec": runtime_sec,
    }


def encode_video(render_buffer: np.memmap, video_info: VideoInfo, output_path: Path) -> None:
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        video_info.fps,
        (video_info.width, video_info.height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Unable to open output writer: {output_path}")
    try:
        for index in range(video_info.frame_count):
            writer.write(np.asarray(render_buffer[index]))
    finally:
        writer.release()


def main() -> None:
    args = parse_args()
    ensure_file(args.video_path, "video")
    ensure_file(args.class_yaml_path, "class yaml")
    ensure_file(args.checkpoint_path, "checkpoint")

    if args.output_dir is None:
        args.output_dir = DEFAULT_RUNS_DIR / f"sam3_1_all_classes_{args.video_path.stem}"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    allowed_labels = resolve_allowed_labels(args.label_policy, args.allowed_labels)
    grouping_mode = resolve_grouping_mode(args.label_policy, args.category_grouping)
    prompts = load_prompts(args.class_yaml_path, args.exclude_labels, allowed_labels)
    prompt_category_map = build_prompt_category_map(prompts, grouping_mode)
    color_map = create_color_map(prompts, grouping_mode, prompt_category_map)
    video_info = read_video_info(args.video_path)

    if not 0 <= args.prompt_frame_index < video_info.frame_count:
        raise ValueError(
            f"prompt-frame-index must be within [0, {video_info.frame_count - 1}], got {args.prompt_frame_index}"
        )

    render_buffer_path = args.output_dir / "render_buffer.dat"
    output_video_path = args.output_dir / "all_classes_combined_mask.mp4"
    summary_json_path = args.output_dir / "combined_mask.json"

    print(f"[info] loading {len(prompts)} prompts from {args.class_yaml_path}")
    print(f"[info] video: {args.video_path}")
    print(f"[info] output dir: {args.output_dir}")
    print(f"[info] frame_count={video_info.frame_count}, size={video_info.width}x{video_info.height}, fps={video_info.fps:.3f}")

    render_buffer: np.memmap | None = None
    try:
        init_start = time.perf_counter()
        render_buffer = initialize_render_buffer(args.video_path, video_info, render_buffer_path)
        assert render_buffer is not None
        init_sec = round(time.perf_counter() - init_start, 2)
        frames_with_union_mask = np.zeros(video_info.frame_count, dtype=bool)

        predictor = build_predictor(args)
        response = predictor.handle_request(
            {
                "type": "start_session",
                "resource_path": str(args.video_path),
                "offload_video_to_cpu": args.offload_video_to_cpu,
            }
        )
        session_id = response["session_id"]

        prompt_start = time.perf_counter()
        per_prompt_stats: List[Dict[str, Any]] = []
        try:
            for index, prompt in enumerate(prompts, start=1):
                print(f"[prompt {index:02d}/{len(prompts):02d}] {prompt}")
                stats = run_prompt(
                    predictor=predictor,
                    session_id=session_id,
                    prompt=prompt,
                    args=args,
                    render_buffer=render_buffer,
                    color_map=color_map,
                    prompt_category_map=prompt_category_map,
                    frames_with_union_mask=frames_with_union_mask,
                )
                per_prompt_stats.append(stats)
                render_buffer.flush()
                print(
                    "  -> "
                    f"frames_with_mask={stats['frames_with_mask']}, "
                    f"max_objects_in_frame={stats['max_objects_in_frame']}, "
                    f"runtime_sec={stats['runtime_sec']}"
                )
        finally:
            predictor.handle_request({"type": "close_session", "session_id": session_id})
        prompt_loop_sec = round(time.perf_counter() - prompt_start, 2)

        render_start = time.perf_counter()
        encode_video(render_buffer, video_info, output_video_path)
        render_sec = round(time.perf_counter() - render_start, 2)
        total_runtime_sec = round(init_sec + prompt_loop_sec + render_sec, 2)

        summary = {
            "video_path": str(args.video_path),
            "class_yaml_path": str(args.class_yaml_path),
            "checkpoint_path": str(args.checkpoint_path),
            "objects": prompts,
            "object_count": len(prompts),
            "label_policy": args.label_policy,
            "category_grouping": grouping_mode,
            "allowed_labels": sorted(allowed_labels) if allowed_labels is not None else None,
            "excluded_labels": list(args.exclude_labels),
            "max_num_objects": args.max_num_objects,
            "multiplex_count": args.multiplex_count,
            "image_size": args.image_size,
            "postprocess_batch_size": args.postprocess_batch_size,
            "batched_grounding_batch_size": args.batched_grounding_batch_size,
            "use_fa3": args.use_fa3,
            "frame_index": args.prompt_frame_index,
            "offload_video_to_cpu": args.offload_video_to_cpu,
            "propagation_direction": args.propagation_direction,
            "frame_count": video_info.frame_count,
            "video_size": [video_info.width, video_info.height],
            "fps": video_info.fps,
            "frames_with_union_mask": int(frames_with_union_mask.sum()),
            "output_video_path": str(output_video_path),
            "prompt_color_map": color_map,
            "prompt_category_map": prompt_category_map,
            "category_color_map": TRAFFIC3_CATEGORY_COLORS if grouping_mode == "traffic3" else None,
            "per_prompt_stats": per_prompt_stats,
            "buffer_init_sec": init_sec,
            "prompt_loop_sec": prompt_loop_sec,
            "render_sec": render_sec,
            "total_runtime_sec": total_runtime_sec,
        }

        with summary_json_path.open("w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, ensure_ascii=False)
    finally:
        if render_buffer is not None:
            del render_buffer
        if render_buffer_path.exists():
            render_buffer_path.unlink()

    print(f"[done] summary: {summary_json_path}")
    print(f"[done] video:   {output_video_path}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
