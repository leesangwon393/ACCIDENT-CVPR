#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import random
import shutil
import subprocess
from pathlib import Path

import pandas as pd


def clamp(value: int, lower: int, upper: int) -> int:
    return max(lower, min(value, upper))


def compute_crop(
    width: int, height: int, center_x: float, center_y: float
) -> tuple[int, int, int, int]:
    crop_width = max(2, int(round(width * 0.5)))
    crop_height = max(2, int(round(height * 0.5)))

    crop_width = min(crop_width, width)
    crop_height = min(crop_height, height)

    center_px = center_x * width
    center_py = center_y * height

    left = int(round(center_px - crop_width / 2))
    top = int(round(center_py - crop_height / 2))

    left = clamp(left, 0, width - crop_width)
    top = clamp(top, 0, height - crop_height)
    return left, top, crop_width, crop_height


def compute_gt_contained_crop(
    width: int, height: int, x1: float, y1: float, x2: float, y2: float
) -> tuple[int, int, int, int]:
    crop_width = max(2, int(round(width * 0.5)))
    crop_height = max(2, int(round(height * 0.5)))

    crop_width = min(crop_width, width)
    crop_height = min(crop_height, height)

    x1_px = x1 * width
    y1_px = y1 * height
    x2_px = x2 * width
    y2_px = y2 * height

    if (x2_px - x1_px) > crop_width or (y2_px - y1_px) > crop_height:
        raise ValueError("GT bbox is larger than the requested crop window")

    min_left = max(0, int(round(x2_px - crop_width)))
    max_left = min(int(round(x1_px)), width - crop_width)
    min_top = max(0, int(round(y2_px - crop_height)))
    max_top = min(int(round(y1_px)), height - crop_height)

    if min_left > max_left or min_top > max_top:
        raise ValueError("Unable to place 25% crop that fully contains GT bbox")

    left = min_left
    top = min_top
    return left, top, crop_width, crop_height


def compute_random_contained_crop(
    width: int,
    height: int,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    rng: random.Random,
) -> tuple[int, int, int, int]:
    crop_width = max(2, int(round(width * 0.5)))
    crop_height = max(2, int(round(height * 0.5)))

    crop_width = min(crop_width, width)
    crop_height = min(crop_height, height)

    x1_px = x1 * width
    y1_px = y1 * height
    x2_px = x2 * width
    y2_px = y2 * height

    if (x2_px - x1_px) > crop_width or (y2_px - y1_px) > crop_height:
        raise ValueError("GT bbox is larger than the requested crop window")

    min_left = max(0, int(round(x2_px - crop_width)))
    max_left = min(int(round(x1_px)), width - crop_width)
    min_top = max(0, int(round(y2_px - crop_height)))
    max_top = min(int(round(y1_px)), height - crop_height)

    if min_left > max_left or min_top > max_top:
        raise ValueError("Unable to place 25% crop that fully contains GT bbox")

    left = rng.randint(min_left, max_left)
    top = rng.randint(min_top, max_top)
    return left, top, crop_width, crop_height


def ffmpeg_crop(
    src: Path, dst: Path, left: int, top: int, crop_width: int, crop_height: int
) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    command = [
        shutil.which("ffmpeg") or "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-vf",
        f"crop={crop_width}:{crop_height}:{left}:{top}",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-c:a",
        "copy",
        str(dst),
    ]
    subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def update_row(
    row: pd.Series,
    left: int,
    top: int,
    crop_width: int,
    crop_height: int,
    output_rel_path: str,
) -> dict:
    x1 = float(row["x1"]) * float(row["width"])
    y1 = float(row["y1"]) * float(row["height"])
    x2 = float(row["x2"]) * float(row["width"])
    y2 = float(row["y2"]) * float(row["height"])

    new_x1 = clamp(int(round(x1 - left)), 0, crop_width)
    new_y1 = clamp(int(round(y1 - top)), 0, crop_height)
    new_x2 = clamp(int(round(x2 - left)), 0, crop_width)
    new_y2 = clamp(int(round(y2 - top)), 0, crop_height)

    if new_x2 < new_x1:
        new_x1, new_x2 = new_x2, new_x1
    if new_y2 < new_y1:
        new_y1, new_y2 = new_y2, new_y1

    center_px = float(row["center_x"]) * float(row["width"])
    center_py = float(row["center_y"]) * float(row["height"])
    new_center_x = min(max((center_px - left) / crop_width, 0.0), 1.0)
    new_center_y = min(max((center_py - top) / crop_height, 0.0), 1.0)

    updated = row.to_dict()
    updated["rgb_path"] = output_rel_path
    updated["height"] = crop_height
    updated["width"] = crop_width
    updated["center_x"] = new_center_x
    updated["center_y"] = new_center_y
    updated["x1"] = new_x1 / crop_width
    updated["y1"] = new_y1 / crop_height
    updated["x2"] = new_x2 / crop_width
    updated["y2"] = new_y2 / crop_height
    updated["crop_left"] = left
    updated["crop_top"] = top
    updated["crop_width"] = crop_width
    updated["crop_height"] = crop_height
    return updated


def split_weather_day_time(raw_weather: str) -> tuple[str, str]:
    normalized = raw_weather.strip().lower()
    if normalized == "night":
        return "normal", "night"
    if normalized == "sunset":
        return "normal", "sunset"
    if normalized == "clear":
        return "normal", "day"
    return normalized, "day"


def build_test_metadata_like(updated_rows: list[dict]) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for row in updated_rows:
        weather, day_time = split_weather_day_time(str(row["weather"]))
        records.append(
            {
                "path": row["rgb_path"],
                "region": row["map"],
                "scene_layout": row["type"],
                "weather": weather,
                "day_time": day_time,
                "quality": "Synthetic",
                "no_frames": int(row["no_frames"]),
                "duration": float(row["duration"]),
                "height": int(row["height"]),
                "width": int(row["width"]),
            }
        )
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create 25%%-area cropped accident video dataset."
    )
    parser.add_argument("--input-dir", required=True, help="Original dataset directory")
    parser.add_argument(
        "--output-dir", required=True, help="Output cropped dataset directory"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Optional row limit for dry runs"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing cropped videos"
    )
    parser.add_argument(
        "--mode",
        choices=["centered", "gt-contained", "random-contained"],
        default="centered",
        help="Crop around accident center, deterministically contain GT bbox, or randomly place a crop while containing GT bbox",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Seed used for random crop placement in random-contained mode",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    labels_path = input_dir / "labels.csv"

    if not labels_path.exists():
        raise FileNotFoundError(f"labels.csv not found: {labels_path}")

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found in PATH")

    df = pd.read_csv(labels_path)
    if args.limit is not None:
        df = df.head(args.limit).copy()

    output_dir.mkdir(parents=True, exist_ok=True)
    if (input_dir / "annotation_classes.yaml").exists():
        shutil.copy2(
            input_dir / "annotation_classes.yaml",
            output_dir / "annotation_classes.yaml",
        )

    updated_rows: list[dict] = []
    failures: list[dict] = []
    rng = random.Random(args.random_seed)

    total = len(df)
    for index, (_, row_series) in enumerate(df.iterrows(), start=1):
        rgb_path = str(row_series["rgb_path"])
        source_rel = rgb_path.replace("videos/", "retestvideos/", 1)
        source_path = input_dir / source_rel
        output_rel = source_rel
        output_path = output_dir / output_rel

        width = int(row_series["width"])
        height = int(row_series["height"])
        if args.mode == "centered":
            left, top, crop_width, crop_height = compute_crop(
                width=width,
                height=height,
                center_x=float(row_series["center_x"]),
                center_y=float(row_series["center_y"]),
            )
        elif args.mode == "gt-contained":
            left, top, crop_width, crop_height = compute_gt_contained_crop(
                width=width,
                height=height,
                x1=float(row_series["x1"]),
                y1=float(row_series["y1"]),
                x2=float(row_series["x2"]),
                y2=float(row_series["y2"]),
            )
        else:
            left, top, crop_width, crop_height = compute_random_contained_crop(
                width=width,
                height=height,
                x1=float(row_series["x1"]),
                y1=float(row_series["y1"]),
                x2=float(row_series["x2"]),
                y2=float(row_series["y2"]),
                rng=rng,
            )

        try:
            if not source_path.exists():
                raise FileNotFoundError(f"missing source video: {source_path}")

            if args.overwrite or not output_path.exists():
                ffmpeg_crop(
                    source_path, output_path, left, top, crop_width, crop_height
                )

            updated_rows.append(
                update_row(row_series, left, top, crop_width, crop_height, output_rel)
            )
        except Exception as exc:  # noqa: BLE001
            failures.append({"rgb_path": rgb_path, "error": str(exc)})

        if index % 100 == 0 or index == total:
            print(
                f"processed {index}/{total} | ok={len(updated_rows)} | failed={len(failures)}"
            )

    if args.mode == "centered":
        labels_output_name = "labels_crop25.csv"
    elif args.mode == "gt-contained":
        labels_output_name = "labels_crop25_gt_contained.csv"
    else:
        labels_output_name = "labels_crop25_random_contained.csv"
    labels_output = output_dir / labels_output_name
    pd.DataFrame(updated_rows).to_csv(labels_output, index=False)

    if args.mode == "centered":
        metadata_output_name = "test_metadata_crop25.csv"
    elif args.mode == "gt-contained":
        metadata_output_name = "test_metadata_crop25_gt_contained.csv"
    else:
        metadata_output_name = "test_metadata_crop25_random_contained.csv"
    build_test_metadata_like(updated_rows).to_csv(
        output_dir / metadata_output_name, index=False
    )

    if failures:
        with (output_dir / "crop_failures.csv").open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(file, fieldnames=["rgb_path", "error"])
            writer.writeheader()
            writer.writerows(failures)

    summary_path = output_dir / "crop_summary.txt"
    summary_path.write_text(
        "\n".join(
            [
                f"input_dir={input_dir}",
                f"output_dir={output_dir}",
                f"requested_area_ratio=0.25",
                f"mode={args.mode}",
                f"random_seed={args.random_seed}",
                f"crop_width_ratio=0.5",
                f"crop_height_ratio=0.5",
                f"total_rows={total}",
                f"successful_rows={len(updated_rows)}",
                f"failed_rows={len(failures)}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
