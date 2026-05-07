#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import cv2
import numpy as np
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SAM_SCRIPT = Path(
    "/Users/isang-won/Desktop/CVPR/Experiment/Segmantation/run_sam3_1_objects_to_single_mask_video.py"
)
DEFAULT_PYTHON_BIN = "/workspace/yuyeon/Experiments/sam3_1_env/bin/python"
DEFAULT_LABELS_CSV = Path(
    "/root/Desktop/workspace/yuyeon/Experiments/16. RB-FT/sim_dataset_aug_fixed3_aug2/labels.csv"
)
DEFAULT_VIDEO_BASE_PATH = Path(
    "/root/Desktop/workspace/yuyeon/Experiments/16. RB-FT/sim_dataset_aug_fixed3_aug2"
)
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "runs" / "sam_train_sampled_traffic3"

TARGET_FPS = 2.0
MIN_FRAMES = 8
MAX_FRAMES = 64


@dataclass(frozen=True)
class VideoInfo:
    frame_count: int
    width: int
    height: int
    fps: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SAM on only train frames used by FT_stage2"
    )
    parser.add_argument("--labels-csv", type=Path, default=DEFAULT_LABELS_CSV)
    parser.add_argument("--video-base-path", type=Path, default=DEFAULT_VIDEO_BASE_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--python-bin", type=str, default=DEFAULT_PYTHON_BIN)
    parser.add_argument("--sam-script", type=Path, default=DEFAULT_SAM_SCRIPT)
    parser.add_argument("--train-max-frames", type=int, default=32)
    parser.add_argument("--sample-long-side", type=int, default=1008)
    parser.add_argument("--max-num-objects", type=int, default=48)
    parser.add_argument("--multiplex-count", type=int, default=16)
    parser.add_argument("--postprocess-batch-size", type=int, default=4)
    parser.add_argument("--batched-grounding-batch-size", type=int, default=4)
    parser.add_argument("--max-videos", type=int, default=0, help="0 means all")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    return parser.parse_args()


def ensure_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def to_float_or(value: str | None, default: float) -> float:
    if value is None or str(value).strip() == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def to_int_or(value: str | None, default: int) -> int:
    if value is None or str(value).strip() == "":
        return default
    try:
        return int(float(value))
    except ValueError:
        return default


def load_unique_rows(labels_csv: Path, max_videos: int) -> List[Dict[str, str]]:
    unique: List[Dict[str, str]] = []
    seen: set[str] = set()
    with labels_csv.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if "rgb_path" not in (reader.fieldnames or []):
            raise ValueError(f"labels csv must include rgb_path: {labels_csv}")
        for row in reader:
            rgb_path = str(row.get("rgb_path", "")).strip()
            if not rgb_path or rgb_path in seen:
                continue
            seen.add(rgb_path)
            unique.append(row)
            if max_videos > 0 and len(unique) >= max_videos:
                break
    return unique


def compute_video_fps(duration: float) -> float:
    if duration <= 0:
        return TARGET_FPS

    fps = TARGET_FPS
    if duration * fps > MAX_FRAMES:
        fps = MAX_FRAMES / duration
    if duration * fps < MIN_FRAMES:
        fps = min(MIN_FRAMES / duration, 4.0)

    fps = max(0.5, min(4.0, fps))
    return round(fps, 2)


def build_train_prompt_yaml(path: Path) -> None:
    names = {
        0: "vehicle",
        1: "wall",
        2: "fence",
        3: "pole",
        4: "vegetation",
        5: "road",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump({"names": names}, fh, sort_keys=False, allow_unicode=True)


def compute_sample_frame_indices(
    video_path: Path,
    duration: float,
    no_frames: int,
    height: int,
    width: int,
    train_max_frames: int,
) -> np.ndarray:
    del no_frames, height, width
    fps = compute_video_fps(duration)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    try:
        total_frames_actual = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()

    if total_frames_actual <= 0:
        raise RuntimeError(f"Video has no readable frames: {video_path}")

    n_frames = max(MIN_FRAMES, min(train_max_frames, int(duration * fps)))
    return np.linspace(0, total_frames_actual - 1, n_frames, dtype=int)


def read_video_info(video_path: Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
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


def build_sampled_video(
    src_video: Path,
    frame_indices: np.ndarray,
    dst_video: Path,
    sample_long_side: int,
) -> List[int]:
    cap = cv2.VideoCapture(str(src_video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source video: {src_video}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 20.0

    dst_video.parent.mkdir(parents=True, exist_ok=True)
    target_width, target_height = width, height
    if max(width, height) > 0 and sample_long_side > 0:
        scale = sample_long_side / float(max(width, height))
        target_width = max(1, int(round(width * scale)))
        target_height = max(1, int(round(height * scale)))

    writer = cv2.VideoWriter(
        str(dst_video),
        cv2.VideoWriter.fourcc(*"mp4v"),
        fps,
        (target_width, target_height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open writer for sampled video: {dst_video}")

    written_indices: List[int] = []
    try:
        for idx in frame_indices.tolist():
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok:
                continue
            if frame.shape[1] != target_width or frame.shape[0] != target_height:
                frame = cv2.resize(
                    frame, (target_width, target_height), interpolation=cv2.INTER_AREA
                )
            writer.write(frame)
            written_indices.append(int(idx))
    finally:
        cap.release()
        writer.release()

    if not written_indices:
        raise RuntimeError(f"No frames written for sampled video: {dst_video}")
    return written_indices


def rebuild_full_video_with_sam_frames(
    src_video: Path,
    sam_video: Path,
    sampled_frame_indices: Sequence[int],
    output_video: Path,
) -> Dict[str, Any]:
    source_info = read_video_info(src_video)

    sam_cap = cv2.VideoCapture(str(sam_video))
    if not sam_cap.isOpened():
        raise RuntimeError(f"Cannot open SAM output video: {sam_video}")

    replacement_frames: Dict[int, np.ndarray] = {}
    duplicate_replacements = 0
    sam_frame_count = 0
    try:
        for frame_index in sampled_frame_indices:
            ok, sam_frame = sam_cap.read()
            if not ok:
                break
            sam_frame_count += 1
            if (
                sam_frame.shape[1] != source_info.width
                or sam_frame.shape[0] != source_info.height
            ):
                sam_frame = cv2.resize(
                    sam_frame,
                    (source_info.width, source_info.height),
                    interpolation=cv2.INTER_LINEAR,
                )
            if int(frame_index) in replacement_frames:
                duplicate_replacements += 1
            replacement_frames[int(frame_index)] = sam_frame.copy()

        ok, extra_frame = sam_cap.read()
        extra_sam_frames = 1 if ok else 0
        while ok:
            ok, extra_frame = sam_cap.read()
            if ok:
                extra_sam_frames += 1
    finally:
        sam_cap.release()

    expected_frames = len(sampled_frame_indices)
    if sam_frame_count != expected_frames or extra_sam_frames != 0:
        raise RuntimeError(
            "SAM output frame count mismatch for reconstruction: "
            f"expected {expected_frames}, consumed {sam_frame_count}, extra {extra_sam_frames}"
        )

    src_cap = cv2.VideoCapture(str(src_video))
    if not src_cap.isOpened():
        raise RuntimeError(
            f"Cannot reopen source video for reconstruction: {src_video}"
        )

    output_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter.fourcc(*"mp4v"),
        source_info.fps,
        (source_info.width, source_info.height),
    )
    if not writer.isOpened():
        src_cap.release()
        raise RuntimeError(f"Cannot open reconstructed video writer: {output_video}")

    frames_written = 0
    replaced_positions = 0
    try:
        frame_position = 0
        while True:
            ok, frame = src_cap.read()
            if not ok:
                break
            replacement = replacement_frames.get(frame_position)
            if replacement is not None:
                writer.write(replacement)
                replaced_positions += 1
            else:
                writer.write(frame)
            frame_position += 1
            frames_written += 1
    finally:
        src_cap.release()
        writer.release()

    if frames_written != source_info.frame_count:
        raise RuntimeError(
            f"Reconstructed video frame count mismatch for {src_video}: "
            f"wrote {frames_written}, expected {source_info.frame_count}"
        )

    return {
        "output_video": str(output_video),
        "source_frame_count": source_info.frame_count,
        "source_width": source_info.width,
        "source_height": source_info.height,
        "source_fps": source_info.fps,
        "sampled_frames_expected": expected_frames,
        "sam_frames_consumed": sam_frame_count,
        "replaced_positions": replaced_positions,
        "unique_replaced_positions": len(replacement_frames),
        "duplicate_replacements": duplicate_replacements,
    }


def run_one_video(
    python_bin: str,
    sam_script: Path,
    sampled_video: Path,
    prompt_yaml: Path,
    run_dir: Path,
    args: argparse.Namespace,
) -> None:
    cmd = [
        python_bin,
        str(sam_script),
        "--video-path",
        str(sampled_video),
        "--class-yaml-path",
        str(prompt_yaml),
        "--output-dir",
        str(run_dir),
        "--label-policy",
        "train_traffic_v2",
        "--category-grouping",
        "traffic3",
        "--max-num-objects",
        str(args.max_num_objects),
        "--multiplex-count",
        str(args.multiplex_count),
        "--postprocess-batch-size",
        str(args.postprocess_batch_size),
        "--batched-grounding-batch-size",
        str(args.batched_grounding_batch_size),
    ]
    subprocess.run(cmd, check=True)


def flush_logs(
    logs_dir: Path,
    summary_records: Sequence[Dict[str, Any]],
    skipped_records: Sequence[Dict[str, Any]],
    error_records: Sequence[Dict[str, Any]],
) -> None:
    (logs_dir / "train_sampled_summary.json").write_text(
        json.dumps(list(summary_records), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (logs_dir / "train_sampled_skipped.json").write_text(
        json.dumps(list(skipped_records), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (logs_dir / "train_sampled_errors.json").write_text(
        json.dumps(list(error_records), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    ensure_file(args.labels_csv, "labels csv")
    ensure_file(args.sam_script, "sam script")

    output_root = args.output_root
    sampled_dir = output_root / "sampled_videos"
    runs_dir = output_root / "runs"
    videos_only_dir = output_root / "videos_only"
    restored_videos_only_dir = output_root / "restored_videos_only"
    prompts_dir = output_root / "prompts"
    logs_dir = output_root / "logs"
    for path in (
        sampled_dir,
        runs_dir,
        videos_only_dir,
        restored_videos_only_dir,
        prompts_dir,
        logs_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)

    prompt_yaml = prompts_dir / "train_traffic_v2.yaml"
    build_train_prompt_yaml(prompt_yaml)

    unique_rows = load_unique_rows(args.labels_csv, args.max_videos)

    summary_records: List[Dict[str, Any]] = []
    skipped_records: List[Dict[str, Any]] = []
    error_records: List[Dict[str, Any]] = []

    for row in unique_rows:
        rgb_path = str(row.get("rgb_path", "")).strip()
        src_video = (args.video_base_path / rgb_path).resolve()
        if not src_video.is_file():
            record = {"rgb_path": rgb_path, "status": "missing_video"}
            summary_records.append(record)
            skipped_records.append(record)
            flush_logs(logs_dir, summary_records, skipped_records, error_records)
            continue
        run_name = src_video.stem
        run_dir = runs_dir / run_name
        out_video = run_dir / "all_classes_combined_mask.mp4"
        summary_json = run_dir / "combined_mask.json"
        restored_video = run_dir / "all_classes_combined_mask_restored_full.mp4"
        restored_manifest = run_dir / "restored_full_video_manifest.json"
        sampled_video = sampled_dir / f"{run_name}_sampled_train.mp4"

        try:
            duration = to_float_or(row.get("duration"), 10.0)
            no_frames = to_int_or(row.get("no_frames"), 0)
            height = to_int_or(row.get("height"), 720)
            width = to_int_or(row.get("width"), 1280)

            frame_indices = compute_sample_frame_indices(
                video_path=src_video,
                duration=duration,
                no_frames=no_frames,
                height=height,
                width=width,
                train_max_frames=args.train_max_frames,
            )

            if (
                args.skip_existing
                and out_video.is_file()
                and summary_json.is_file()
                and restored_video.is_file()
                and restored_manifest.is_file()
            ):
                manifest_sampled_indices = frame_indices.tolist()
                try:
                    restored_payload = json.loads(
                        restored_manifest.read_text(encoding="utf-8")
                    )
                    manifest_indices_value = restored_payload.get(
                        "sampled_frame_indices"
                    )
                    if isinstance(manifest_indices_value, list):
                        manifest_sampled_indices = [
                            int(value) for value in manifest_indices_value
                        ]
                except (OSError, json.JSONDecodeError, TypeError, ValueError):
                    pass
                target_video = (
                    videos_only_dir / f"{run_name}_all_classes_combined_mask.mp4"
                )
                if not target_video.is_file():
                    shutil.copy2(out_video, target_video)
                restored_target_video = (
                    restored_videos_only_dir
                    / f"{run_name}_all_classes_combined_mask_restored_full.mp4"
                )
                if not restored_target_video.is_file():
                    shutil.copy2(restored_video, restored_target_video)
                record = {
                    "rgb_path": rgb_path,
                    "source_video": str(src_video),
                    "sampled_video": str(sampled_video),
                    "sam_output_video": str(out_video),
                    "restored_video": str(restored_video),
                    "restored_manifest": str(restored_manifest),
                    "sampled_frame_indices": manifest_sampled_indices,
                    "status": "skipped_existing",
                }
                summary_records.append(record)
                skipped_records.append(record)
                flush_logs(logs_dir, summary_records, skipped_records, error_records)
                continue

            written_indices = build_sampled_video(
                src_video=src_video,
                frame_indices=frame_indices,
                dst_video=sampled_video,
                sample_long_side=args.sample_long_side,
            )
            run_one_video(
                python_bin=args.python_bin,
                sam_script=args.sam_script,
                sampled_video=sampled_video,
                prompt_yaml=prompt_yaml,
                run_dir=run_dir,
                args=args,
            )

            if not out_video.is_file() or not summary_json.is_file():
                raise RuntimeError(f"Missing SAM outputs for {src_video}")
            restored_info = rebuild_full_video_with_sam_frames(
                src_video=src_video,
                sam_video=out_video,
                sampled_frame_indices=written_indices,
                output_video=restored_video,
            )
            restored_manifest.write_text(
                json.dumps(
                    {
                        "rgb_path": rgb_path,
                        "source_video": str(src_video),
                        "sampled_video": str(sampled_video),
                        "sam_output_video": str(out_video),
                        "sampled_frame_indices": written_indices,
                        **restored_info,
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            shutil.copy2(
                out_video, videos_only_dir / f"{run_name}_all_classes_combined_mask.mp4"
            )
            shutil.copy2(
                restored_video,
                restored_videos_only_dir
                / f"{run_name}_all_classes_combined_mask_restored_full.mp4",
            )
            summary_records.append(
                {
                    "rgb_path": rgb_path,
                    "source_video": str(src_video),
                    "sampled_video": str(sampled_video),
                    "sam_output_video": str(out_video),
                    "restored_video": str(restored_video),
                    "restored_manifest": str(restored_manifest),
                    "sampled_frame_indices": written_indices,
                    "sampled_frames_written": len(written_indices),
                    "run_dir": str(run_dir),
                    "restored_replaced_positions": int(
                        restored_info["replaced_positions"]
                    ),
                    "status": "done",
                }
            )
            flush_logs(logs_dir, summary_records, skipped_records, error_records)
        except Exception as exc:
            error_record = {
                "rgb_path": rgb_path,
                "source_video": str(src_video),
                "sampled_video": str(sampled_video),
                "run_dir": str(run_dir),
                "status": "error",
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            summary_records.append(error_record)
            error_records.append(error_record)
            flush_logs(logs_dir, summary_records, skipped_records, error_records)

    print(f"[done] processed {len(summary_records)} videos")
    print(f"[done] output root: {output_root}")


if __name__ == "__main__":
    main()
