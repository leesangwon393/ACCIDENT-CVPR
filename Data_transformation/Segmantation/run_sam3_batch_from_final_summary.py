#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SAM batch jobs from final_summary.json")
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--runner-script", type=Path, required=True)
    parser.add_argument("--python-bin", type=str, default="python3")
    parser.add_argument("--num-shards", type=int, default=2)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--cuda-device", type=str, required=True)
    parser.add_argument("--max-num-objects", type=int, default=48)
    parser.add_argument("--multiplex-count", type=int, default=16)
    parser.add_argument("--postprocess-batch-size", type=int, default=2)
    parser.add_argument("--batched-grounding-batch-size", type=int, default=2)
    parser.add_argument(
        "--label-policy",
        choices=("raw", "train_traffic_v2", "test_traffic_v2"),
        default="test_traffic_v2",
    )
    parser.add_argument(
        "--category-grouping",
        choices=("none", "traffic3", "auto"),
        default="auto",
    )
    parser.add_argument("--skip-existing", action="store_true", default=True)
    return parser.parse_args()


def ensure_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def write_prompt_yaml(path: Path, prompts: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["names:"]
    for idx, prompt in enumerate(prompts):
        lines.append(f"  {idx}: {prompt}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_entries(summary_json: Path) -> List[Dict[str, Any]]:
    data = json.loads(summary_json.read_text(encoding="utf-8"))
    videos = data.get("videos_analyzed", [])
    if not isinstance(videos, list):
        raise ValueError("final_summary.json missing videos_analyzed list")
    return videos


def main() -> None:
    args = parse_args()
    ensure_file(args.summary_json, "summary json")
    ensure_file(args.runner_script, "runner script")

    entries = load_entries(args.summary_json)
    output_root = args.output_root
    prompts_root = output_root / "prompts"
    videos_only_root = output_root / "videos_only"
    logs_root = output_root / "logs"
    output_root.mkdir(parents=True, exist_ok=True)
    prompts_root.mkdir(parents=True, exist_ok=True)
    videos_only_root.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)

    status_jsonl = logs_root / f"shard_{args.shard_index}.jsonl"

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    total_in_shard = 0
    done = 0
    skipped = 0
    failed = 0

    for global_idx, entry in enumerate(entries):
        if global_idx % args.num_shards != args.shard_index:
            continue
        total_in_shard += 1

    for global_idx, entry in enumerate(entries):
        if global_idx % args.num_shards != args.shard_index:
            continue

        filename = str(entry.get("filename", "")).strip()
        prompts = entry.get("unique_objects", [])
        if not filename or not isinstance(prompts, list) or len(prompts) == 0:
            failed += 1
            record = {
                "filename": filename,
                "status": "failed",
                "reason": "invalid filename or prompts",
                "global_index": global_idx,
            }
            with status_jsonl.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            continue

        video_path = args.video_root / filename
        run_name = Path(filename).stem
        run_dir = output_root / run_name
        summary_path = run_dir / "combined_mask.json"
        output_video = run_dir / "all_classes_combined_mask.mp4"
        flat_video = videos_only_root / f"{run_name}_all_classes_combined_mask.mp4"
        yaml_path = prompts_root / f"{run_name}.yaml"

        if args.skip_existing and summary_path.is_file() and output_video.is_file():
            if not flat_video.is_file():
                shutil.copy2(output_video, flat_video)
            skipped += 1
            record = {
                "filename": filename,
                "status": "skipped",
                "reason": "existing output",
                "global_index": global_idx,
            }
            with status_jsonl.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            continue

        if not video_path.is_file():
            failed += 1
            record = {
                "filename": filename,
                "status": "failed",
                "reason": f"video not found: {video_path}",
                "global_index": global_idx,
            }
            with status_jsonl.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            continue

        write_prompt_yaml(yaml_path, [str(x).strip() for x in prompts if str(x).strip()])

        cmd = [
            args.python_bin,
            str(args.runner_script),
            "--video-path",
            str(video_path),
            "--class-yaml-path",
            str(yaml_path),
            "--output-dir",
            str(run_dir),
            "--max-num-objects",
            str(args.max_num_objects),
            "--multiplex-count",
            str(args.multiplex_count),
            "--postprocess-batch-size",
            str(args.postprocess_batch_size),
            "--batched-grounding-batch-size",
            str(args.batched_grounding_batch_size),
            "--label-policy",
            str(args.label_policy),
            "--category-grouping",
            str(args.category_grouping),
        ]

        t0 = time.perf_counter()
        proc = subprocess.run(cmd, env=env)
        elapsed = round(time.perf_counter() - t0, 2)

        if proc.returncode == 0 and summary_path.is_file() and output_video.is_file():
            shutil.copy2(output_video, flat_video)
            done += 1
            record = {
                "filename": filename,
                "status": "done",
                "elapsed_sec": elapsed,
                "global_index": global_idx,
            }
        else:
            failed += 1
            record = {
                "filename": filename,
                "status": "failed",
                "elapsed_sec": elapsed,
                "returncode": proc.returncode,
                "global_index": global_idx,
            }

        with status_jsonl.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    final = {
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "cuda_device": args.cuda_device,
        "max_num_objects": args.max_num_objects,
        "multiplex_count": args.multiplex_count,
        "postprocess_batch_size": args.postprocess_batch_size,
        "batched_grounding_batch_size": args.batched_grounding_batch_size,
        "label_policy": args.label_policy,
        "category_grouping": args.category_grouping,
        "total_in_shard": total_in_shard,
        "done": done,
        "skipped": skipped,
        "failed": failed,
        "status_jsonl": str(status_jsonl),
    }
    (logs_root / f"shard_{args.shard_index}_summary.json").write_text(
        json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(final, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
