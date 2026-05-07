#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SAM_SCRIPT = Path(
    "/Users/isang-won/Desktop/CVPR/Experiment/Segmantation/run_sam3_1_objects_to_single_mask_video.py"
)
DEFAULT_PYTHON_BIN = "/workspace/yuyeon/Experiments/sam3_1_env/bin/python"
DEFAULT_SUMMARY_JSON = Path(
    "/root/Desktop/workspace/yuyeon/Experiments/17. SSSSS/object_eval_full_merged/final_summary.json"
)
DEFAULT_VIDEO_ROOT = Path("/root/Desktop/workspace/yuyeon/raw/accident/videos")
DEFAULT_METADATA_CSV = Path(
    "/root/Desktop/workspace/yuyeon/raw/accident/test_metadata.csv"
)
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "runs" / "sam_test_sampled_traffic3"

TARGET_FPS = 2.0
MIN_FRAMES = 8
MAX_FRAMES = 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SAM on only test frames used by baseline.py"
    )
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--metadata-csv", type=Path, default=DEFAULT_METADATA_CSV)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--python-bin", type=str, default=DEFAULT_PYTHON_BIN)
    parser.add_argument("--sam-script", type=Path, default=DEFAULT_SAM_SCRIPT)
    parser.add_argument("--test-max-frames", type=int, default=MAX_FRAMES)
    parser.add_argument("--sample-long-side", type=int, default=1008)
    parser.add_argument("--max-num-objects", type=int, default=48)
    parser.add_argument("--multiplex-count", type=int, default=16)
    parser.add_argument("--postprocess-batch-size", type=int, default=4)
    parser.add_argument("--batched-grounding-batch-size", type=int, default=4)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--cuda-device", type=str, default="")
    parser.add_argument("--max-videos", type=int, default=0, help="0 means all")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    return parser.parse_args()


def ensure_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


def append_failed_video(path: Path, filename: str) -> None:
    if not filename:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(f"{filename}\n")


def load_entries(summary_json: Path) -> List[Dict[str, Any]]:
    data = json.loads(summary_json.read_text(encoding="utf-8"))
    videos = data.get("videos_analyzed", [])
    if not isinstance(videos, list):
        raise ValueError("final_summary.json missing videos_analyzed list")
    return videos


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


def load_metadata(csv_path: Path) -> Dict[str, Dict[str, Any]]:
    if not csv_path.is_file():
        return {}
    meta: Dict[str, Dict[str, Any]] = {}
    with csv_path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            key = str(row.get("path", "")).strip()
            if not key:
                continue
            meta[key] = {
                "duration": to_float_or(row.get("duration"), 10.0),
                "no_frames": to_int_or(row.get("no_frames"), 0),
                "height": to_int_or(row.get("height"), 720),
                "width": to_int_or(row.get("width"), 1280),
            }
    return meta


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


def resolve_meta(
    filename: str, metadata_map: Dict[str, Dict[str, Any]]
) -> Dict[str, Any] | None:
    if filename in metadata_map:
        return metadata_map[filename]
    base_name = Path(filename).name
    if base_name in metadata_map:
        return metadata_map[base_name]
    for key, value in metadata_map.items():
        if Path(key).name == base_name:
            return value
    return None


def write_prompt_yaml(path: Path, prompts: List[str]) -> None:
    names = {idx: prompt for idx, prompt in enumerate(prompts)}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump({"names": names}, fh, sort_keys=False, allow_unicode=True)


def compute_sample_frame_indices(
    video_path: Path,
    duration: float,
    no_frames: int,
    height: int,
    width: int,
    test_max_frames: int,
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

    n_frames = max(MIN_FRAMES, min(test_max_frames, int(duration * fps)))
    return np.linspace(0, total_frames_actual - 1, n_frames, dtype=int)


def build_sampled_video(
    src_video: Path,
    frame_indices: np.ndarray,
    dst_video: Path,
    sample_long_side: int,
) -> int:
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

    written = 0
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
            written += 1
    finally:
        cap.release()
        writer.release()

    if written <= 0:
        raise RuntimeError(f"No frames written for sampled video: {dst_video}")
    return written


def run_one_video(
    python_bin: str,
    sam_script: Path,
    sampled_video: Path,
    prompt_yaml: Path,
    run_dir: Path,
    args: argparse.Namespace,
    env: Dict[str, str],
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
        "test_traffic_v2",
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
    subprocess.run(cmd, check=True, env=env)


def main() -> None:
    args = parse_args()
    if args.num_shards <= 0:
        raise ValueError("num_shards must be >= 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError(f"shard_index must be in [0, {args.num_shards - 1}]")

    ensure_file(args.summary_json, "summary json")
    ensure_file(args.sam_script, "sam script")
    metadata_map = load_metadata(args.metadata_csv)

    run_env = os.environ.copy()
    if args.cuda_device:
        run_env["CUDA_VISIBLE_DEVICES"] = args.cuda_device

    output_root = args.output_root
    sampled_dir = output_root / "sampled_videos"
    runs_dir = output_root / "runs"
    videos_only_dir = output_root / "videos_only"
    prompts_dir = output_root / "prompts"
    logs_dir = output_root / "logs"
    for path in (sampled_dir, runs_dir, videos_only_dir, prompts_dir, logs_dir):
        path.mkdir(parents=True, exist_ok=True)

    failed_jsonl_name = (
        f"failed_cases_shard_{args.shard_index}.jsonl"
        if args.num_shards > 1
        else "failed_cases.jsonl"
    )
    failed_txt_name = (
        f"failed_videos_shard_{args.shard_index}.txt"
        if args.num_shards > 1
        else "failed_videos.txt"
    )
    failed_jsonl_path = logs_dir / failed_jsonl_name
    failed_txt_path = logs_dir / failed_txt_name

    entries = load_entries(args.summary_json)
    total_candidates = len(entries)
    entries = [
        entry
        for idx, entry in enumerate(entries)
        if idx % args.num_shards == args.shard_index
    ]
    if args.max_videos > 0:
        entries = entries[: args.max_videos]

    summary_records: List[Dict[str, Any]] = []

    for entry in entries:
        filename = str(entry.get("filename", "")).strip()
        prompts = [
            str(x).strip() for x in entry.get("unique_objects", []) if str(x).strip()
        ]
        if not filename or not prompts:
            record = {"filename": filename, "status": "invalid_entry"}
            summary_records.append(record)
            append_jsonl(failed_jsonl_path, record)
            append_failed_video(failed_txt_path, filename)
            continue

        src_video = (args.video_root / filename).resolve()
        if not src_video.is_file():
            record = {
                "filename": filename,
                "status": "missing_video",
                "source_video": str(src_video),
            }
            summary_records.append(record)
            append_jsonl(failed_jsonl_path, record)
            append_failed_video(failed_txt_path, filename)
            continue

        run_name = Path(filename).stem
        run_dir = runs_dir / run_name
        out_video = run_dir / "all_classes_combined_mask.mp4"
        summary_json = run_dir / "combined_mask.json"
        sampled_video = sampled_dir / f"{run_name}_sampled_test.mp4"
        prompt_yaml = prompts_dir / f"{run_name}.yaml"

        if args.skip_existing and out_video.is_file() and summary_json.is_file():
            target_video = videos_only_dir / f"{run_name}_all_classes_combined_mask.mp4"
            if not target_video.is_file():
                shutil.copy2(out_video, target_video)
            summary_records.append({"filename": filename, "status": "skipped_existing"})
            continue

        meta = resolve_meta(filename, metadata_map)
        if meta:
            duration = to_float_or(str(meta.get("duration", "")), 10.0)
            no_frames = to_int_or(str(meta.get("no_frames", "")), 0)
            height = to_int_or(str(meta.get("height", "")), 720)
            width = to_int_or(str(meta.get("width", "")), 1280)
        else:
            duration = 10.0
            no_frames = 0
            height = 720
            width = 1280

        frame_indices = compute_sample_frame_indices(
            video_path=src_video,
            duration=duration,
            no_frames=no_frames,
            height=height,
            width=width,
            test_max_frames=args.test_max_frames,
        )

        write_prompt_yaml(prompt_yaml, prompts)
        written = build_sampled_video(
            src_video=src_video,
            frame_indices=frame_indices,
            dst_video=sampled_video,
            sample_long_side=args.sample_long_side,
        )
        try:
            run_one_video(
                python_bin=args.python_bin,
                sam_script=args.sam_script,
                sampled_video=sampled_video,
                prompt_yaml=prompt_yaml,
                run_dir=run_dir,
                args=args,
                env=run_env,
            )
        except subprocess.CalledProcessError as exc:
            record = {
                "filename": filename,
                "source_video": str(src_video),
                "sampled_video": str(sampled_video),
                "sampled_frames_written": int(written),
                "run_dir": str(run_dir),
                "status": "failed_subprocess",
                "returncode": int(exc.returncode),
            }
            summary_records.append(record)
            append_jsonl(failed_jsonl_path, record)
            append_failed_video(failed_txt_path, filename)
            continue

        if not out_video.is_file() or not summary_json.is_file():
            record = {
                "filename": filename,
                "source_video": str(src_video),
                "sampled_video": str(sampled_video),
                "sampled_frames_written": int(written),
                "run_dir": str(run_dir),
                "status": "missing_outputs",
            }
            summary_records.append(record)
            append_jsonl(failed_jsonl_path, record)
            append_failed_video(failed_txt_path, filename)
            continue
        shutil.copy2(
            out_video, videos_only_dir / f"{run_name}_all_classes_combined_mask.mp4"
        )
        summary_records.append(
            {
                "filename": filename,
                "source_video": str(src_video),
                "sampled_video": str(sampled_video),
                "sampled_frames_written": int(written),
                "run_dir": str(run_dir),
                "status": "done",
            }
        )

    summary_name = (
        f"test_sampled_summary_shard_{args.shard_index}.json"
        if args.num_shards > 1
        else "test_sampled_summary.json"
    )
    with (logs_dir / summary_name).open("w", encoding="utf-8") as fh:
        json.dump(summary_records, fh, indent=2, ensure_ascii=False)

    failed_unique = sorted(
        {
            str(item.get("filename", "")).strip()
            for item in summary_records
            if str(item.get("status", "")).startswith("failed")
            or str(item.get("status", "")).startswith("missing")
            or str(item.get("status", "")) == "invalid_entry"
        }
    )
    if failed_unique:
        with failed_txt_path.open("a", encoding="utf-8") as fh:
            fh.write("# unique_failed_videos_current_run\n")
            for name in failed_unique:
                if name:
                    fh.write(f"{name}\n")

    print(
        f"[done] processed {len(summary_records)} videos "
        f"(total_candidates={total_candidates}, shard={args.shard_index}/{args.num_shards})"
    )
    print(f"[done] output root: {output_root}")


if __name__ == "__main__":
    main()
