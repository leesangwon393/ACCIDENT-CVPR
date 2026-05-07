import argparse
import importlib.util
from pathlib import Path
from typing import Any, Dict, List, cast

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
MODULE_PATH = SCRIPT_DIR / "baseline_3stage.py"
VIDEO_DIR = REPO_ROOT / "raw" / "accident" / "videos"
METADATA_CSV = REPO_ROOT / "raw" / "accident" / "test_metadata.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "baseline_3stage_full"
DEFAULT_CHECKPOINT_EVERY = 10


def load_baseline_module() -> Any:
    spec = importlib.util.spec_from_file_location("baseline_3stage", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_all_video_paths() -> List[Path]:
    paths = sorted(VIDEO_DIR.glob("*.mp4"))
    if not paths:
        raise FileNotFoundError(f"No .mp4 files found in: {VIDEO_DIR}")
    return paths


def get_shard_paths(paths: List[Path], rank: int, world_size: int) -> List[Path]:
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"rank must be in [0, {world_size}), got {rank}")
    return paths[rank::world_size]


def build_metadata_lookup(module: Any) -> Dict[str, Dict[str, Any]]:
    return module.load_metadata(str(METADATA_CSV))


def build_submission(df: pd.DataFrame) -> pd.DataFrame:
    submission = cast(pd.DataFrame, df.loc[:, ["path", "accident_time", "center_x", "center_y", "type"]].copy())
    submission["accident_time"] = submission["accident_time"].astype(float).round(2)
    submission["center_x"] = submission["center_x"].astype(float).round(3)
    submission["center_y"] = submission["center_y"].astype(float).round(3)
    return submission


def ensure_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def normalize_prediction_path(value: Any) -> str:
    text = str(value).strip()
    if not text:
        return text
    if text.startswith("videos/"):
        return text
    return f"videos/{Path(text).name}"


def save_rank_outputs(df: pd.DataFrame, output_dir: Path, rank: int) -> None:
    ensure_output_dir(output_dir)
    debug_path = output_dir / f"debug_results_rank{rank}.csv"
    submission_path = output_dir / f"submission_rank{rank}.csv"
    df.to_csv(debug_path, index=False)
    build_submission(df).to_csv(submission_path, index=False)
    print(f"[INFO] rank {rank} debug saved -> {debug_path}")
    print(f"[INFO] rank {rank} submission saved -> {submission_path}")


def run_shard(rank: int, world_size: int, output_dir: Path, checkpoint_every: int) -> None:
    module = load_baseline_module()
    metadata = build_metadata_lookup(module)
    all_paths = get_all_video_paths()
    shard_paths = get_shard_paths(all_paths, rank, world_size)
    shard_metas = [metadata.get(f"videos/{path.name}") for path in shard_paths]

    print(f"[INFO] Loaded metadata for {len(metadata)} videos")
    print(f"[INFO] Rank {rank}/{world_size} processing {len(shard_paths)} videos")
    if shard_paths:
        print(f"[INFO] Rank {rank} first video: {shard_paths[0].name}")
        print(f"[INFO] Rank {rank} last video: {shard_paths[-1].name}")

    model, processor = module.load_model()

    records: List[Dict[str, Any]] = []
    total = len(shard_paths)
    for index, (video_path, meta) in enumerate(zip(shard_paths, shard_metas), start=1):
        video_name = video_path.name
        if meta:
            fps = module.compute_video_fps(meta["duration"], meta["no_frames"], meta["height"], meta["width"])
            est_frames = int(meta["duration"] * fps)
            print(
                f"[rank {rank} {index}/{total}] {video_name} | "
                f"{meta['duration']:.1f}s {meta['width']}x{meta['height']} -> stage1 fps={fps} (~{est_frames}f)"
            )
        else:
            print(f"[rank {rank} {index}/{total}] {video_name} | no metadata")

        record = module.infer_video(model, processor, str(video_path), meta=meta)
        print(
            f"  -> {record['accident_time']:.2f}s | "
            f"({record['center_x']:.3f}, {record['center_y']:.3f}) | "
            f"{record['type']} | conf={record['confidence']:.2f} | {record['method']}"
        )
        records.append(record)

        if checkpoint_every > 0 and index % checkpoint_every == 0:
            save_rank_outputs(pd.DataFrame(records), output_dir, rank)
            print(f"[INFO] Rank {rank} checkpoint saved at {index}/{total} videos")

    save_rank_outputs(pd.DataFrame(records), output_dir, rank)
    print(f"[INFO] Rank {rank} completed {total} videos")


def merge_outputs(world_size: int, output_dir: Path) -> None:
    ensure_output_dir(output_dir)
    all_paths = get_all_video_paths()
    expected_relative_paths = [f"videos/{path.name}" for path in all_paths]
    ordering = {path: index for index, path in enumerate(expected_relative_paths)}

    frames = []
    for rank in range(world_size):
        shard_path = output_dir / f"debug_results_rank{rank}.csv"
        if not shard_path.exists():
            raise FileNotFoundError(f"Missing shard output: {shard_path}")
        frame = pd.read_csv(shard_path)
        frame["path"] = frame["path"].map(normalize_prediction_path)
        frames.append(frame)

    merged = pd.concat(frames, ignore_index=True)
    if len(merged) != len(expected_relative_paths):
        raise ValueError(f"Merged row count {len(merged)} does not match expected {len(expected_relative_paths)}")
    if merged["path"].duplicated().any():
        duplicates = merged.loc[merged["path"].duplicated(), "path"].tolist()
        raise ValueError(f"Duplicate paths detected: {duplicates[:10]}")

    merged["__order"] = merged["path"].map(ordering)
    missing_mask = pd.isna(merged["__order"])
    if int(missing_mask.sum()) > 0:
        missing_paths = merged.loc[missing_mask, "path"].tolist()
        raise ValueError(f"Unexpected paths detected: {missing_paths[:10]}")

    merged = merged.sort_values("__order").drop(columns=["__order"]).reset_index(drop=True)

    missing = sorted(set(expected_relative_paths) - set(merged["path"].tolist()))
    if missing:
        raise ValueError(f"Missing paths detected: {missing[:10]}")

    debug_path = output_dir / "debug_results_full.csv"
    submission_path = output_dir / "submission_full.csv"
    merged.to_csv(debug_path, index=False)
    build_submission(merged).to_csv(submission_path, index=False)
    print(f"[INFO] merged debug saved -> {debug_path}")
    print(f"[INFO] merged submission saved -> {submission_path}")
    print(f"[INFO] merged rows -> {len(merged)}")


def print_preflight(world_size: int) -> None:
    all_paths = get_all_video_paths()
    print(f"[INFO] Total videos: {len(all_paths)}")
    for rank in range(world_size):
        shard_paths = get_shard_paths(all_paths, rank, world_size)
        first_name = shard_paths[0].name if shard_paths else "-"
        last_name = shard_paths[-1].name if shard_paths else "-"
        print(
            f"[INFO] Rank {rank}: count={len(shard_paths)} "
            f"first={first_name} last={last_name}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Shard-aware full inference runner for baseline_3stage.py")
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint-every", type=int, default=DEFAULT_CHECKPOINT_EVERY)
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.preflight:
        print_preflight(args.world_size)
        return

    if args.merge_only:
        merge_outputs(args.world_size, args.output_dir)
        return

    if args.rank is None:
        raise ValueError("--rank is required unless --merge-only or --preflight is used")

    run_shard(args.rank, args.world_size, args.output_dir, args.checkpoint_every)


if __name__ == "__main__":
    main()
