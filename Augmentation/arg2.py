from __future__ import annotations

import csv
import importlib
import importlib.util
import os
import shutil
import time
from pathlib import Path
from typing import Protocol, TypedDict, cast


class FrameLike(Protocol):
    shape: tuple[int, int] | tuple[int, int, int]

    def copy(self) -> FrameLike: ...


class KernelLike(Protocol):
    def __setitem__(
        self, key: tuple[int, slice] | tuple[slice, int], value: float
    ) -> None: ...

    def __itruediv__(self, other: float) -> KernelLike: ...

    def sum(self) -> float: ...


class VideoCaptureLike(Protocol):
    def isOpened(self) -> bool: ...

    def get(self, prop_id: int) -> float: ...

    def read(self) -> tuple[bool, FrameLike | None]: ...

    def release(self) -> None: ...


class VideoWriterLike(Protocol):
    def isOpened(self) -> bool: ...

    def write(self, frame: FrameLike) -> None: ...

    def release(self) -> None: ...


class Cv2Module(Protocol):
    INTER_AREA: int
    INTER_LINEAR: int
    IMWRITE_JPEG_QUALITY: int
    IMREAD_COLOR: int
    CAP_PROP_FPS: int
    CAP_PROP_FRAME_WIDTH: int
    CAP_PROP_FRAME_HEIGHT: int

    def resize(
        self, frame: FrameLike, dsize: tuple[int, int], interpolation: int
    ) -> FrameLike: ...

    def GaussianBlur(
        self, frame: FrameLike, ksize: tuple[int, int], sigmaX: int
    ) -> FrameLike: ...

    def filter2D(
        self, frame: FrameLike, ddepth: int, kernel: KernelLike
    ) -> FrameLike: ...

    def imencode(
        self, ext: str, img: FrameLike, params: list[int]
    ) -> tuple[bool, object]: ...

    def imdecode(self, buf: object, flags: int) -> FrameLike | None: ...

    def VideoCapture(self, filename: str) -> VideoCaptureLike: ...

    def VideoWriter(
        self, filename: str, fourcc: int, fps: float, frameSize: tuple[int, int]
    ) -> VideoWriterLike: ...

    def VideoWriter_fourcc(self, *args: str) -> int: ...


class NumpyModule(Protocol):
    float32: object

    def zeros(self, shape: tuple[int, int], dtype: object) -> KernelLike: ...

    def fill_diagonal(self, a: KernelLike, val: float) -> None: ...

    def fliplr(self, m: KernelLike) -> KernelLike: ...


cv2: Cv2Module | None = cast(
    Cv2Module | None,
    importlib.import_module("cv2") if importlib.util.find_spec("cv2") else None,
)
np: NumpyModule | None = cast(
    NumpyModule | None,
    importlib.import_module("numpy") if importlib.util.find_spec("numpy") else None,
)


BASE_DIR = Path(__file__).resolve().parent
SRC_ROOT = BASE_DIR / "sim_dataset"
DST_ROOT = Path(
    os.environ.get("ARGMENT_OUTPUT_DIR", str(BASE_DIR / "sim_dataset_aug_fixed2"))
)
SRC_VIDEO_DIR = SRC_ROOT / "videos"
OUT_ROOT = DST_ROOT / "arg2"
SRC_LABELS_CSV = SRC_ROOT / "labels.csv"
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
OUTPUT_FOURCC = "mp4v"


class AugConfig(TypedDict):
    resize_scale: float
    blur_type: str
    blur_kernel: int
    jpeg_quality: int


CONFIG: AugConfig = {
    "resize_scale": 0.4,
    "blur_type": "motion",
    "blur_kernel": 9,
    "jpeg_quality": 28,
}


def log(message: str) -> None:
    print(message, flush=True)


def format_duration(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return (
        f"{hours:d}:{minutes:02d}:{secs:02d}"
        if hours > 0
        else f"{minutes:02d}:{secs:02d}"
    )


def has_video_dependencies() -> bool:
    return cv2 is not None and np is not None


def require_cv2() -> Cv2Module:
    assert cv2 is not None
    return cv2


def require_np() -> NumpyModule:
    assert np is not None
    return np


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def list_video_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTS
    )


def normalize_rel_path(value: str) -> str:
    return value.replace("\\", "/")


def build_augmented_rel_path(original_rel_path: str, tag: str) -> str:
    path = Path(normalize_rel_path(original_rel_path))
    filename = f"{path.stem}_{tag}{path.suffix}"
    return (
        filename
        if str(path.parent) in {"", "."}
        else (path.parent / filename).as_posix()
    )


def downsample_then_upsample(frame: FrameLike, scale: float) -> FrameLike:
    cv2m = require_cv2()
    height = frame.shape[0]
    width = frame.shape[1]
    small_w = max(1, int(round(width * scale)))
    small_h = max(1, int(round(height * scale)))
    small = cv2m.resize(frame, (small_w, small_h), interpolation=cv2m.INTER_AREA)
    return cv2m.resize(small, (width, height), interpolation=cv2m.INTER_LINEAR)


def apply_gaussian_blur(frame: FrameLike, ksize: int) -> FrameLike:
    cv2m = require_cv2()
    if ksize % 2 == 0:
        ksize += 1
    return cv2m.GaussianBlur(frame, (ksize, ksize), 0)


def apply_motion_blur(frame: FrameLike, ksize: int) -> FrameLike:
    cv2m = require_cv2()
    npm = require_np()
    kernel = npm.zeros((ksize, ksize), dtype=npm.float32)
    kernel[ksize // 2, :] = 1.0
    kernel /= kernel.sum()
    return cv2m.filter2D(frame, -1, kernel)


def apply_jpeg_compression(frame: FrameLike, quality: int) -> FrameLike:
    cv2m = require_cv2()
    quality = max(5, min(100, int(quality)))
    ok, encoded = cv2m.imencode(
        ".jpg", frame, [int(cv2m.IMWRITE_JPEG_QUALITY), quality]
    )
    if not ok:
        return frame
    decoded = cv2m.imdecode(encoded, cv2m.IMREAD_COLOR)
    return decoded if decoded is not None else frame


def augment_frame(frame: FrameLike, config: AugConfig) -> FrameLike:
    out = frame.copy()
    out = downsample_then_upsample(out, float(config["resize_scale"]))
    out = (
        apply_motion_blur(out, int(config["blur_kernel"]))
        if config["blur_type"] == "motion"
        else apply_gaussian_blur(out, int(config["blur_kernel"]))
    )
    return apply_jpeg_compression(out, int(config["jpeg_quality"]))


def process_video(
    input_path: Path, output_path: Path, config: AugConfig
) -> dict[str, object]:
    cv2m = require_cv2()
    cap = cv2m.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    fps = cap.get(cv2m.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    width = int(cap.get(cv2m.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2m.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError(f"Invalid frame size: {input_path}")

    ensure_dir(output_path.parent)
    writer = cv2m.VideoWriter(
        str(output_path), cv2m.VideoWriter_fourcc(*OUTPUT_FOURCC), fps, (width, height)
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot create writer: {output_path}")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        writer.write(augment_frame(frame, config))
        frame_count += 1

    cap.release()
    writer.release()
    return {"frame_count": frame_count}


def detect_video_path_column(fieldnames: list[str]) -> str | None:
    candidates = [
        "rgb_path",
        "path",
        "video_path",
        "video",
        "filepath",
        "file_path",
        "filename",
    ]
    lowered = {name.lower(): name for name in fieldnames}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    return None


def read_csv_rows(csv_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with open(csv_path, "r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return list(reader.fieldnames or []), [dict(row) for row in reader]


def write_csv_rows(
    csv_path: Path, fieldnames: list[str], rows: list[dict[str, str]]
) -> None:
    ensure_dir(csv_path.parent)
    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_video_order_map(root: Path) -> dict[str, int]:
    return {
        (Path("videos") / video_path.relative_to(root)).as_posix(): index
        for index, video_path in enumerate(list_video_files(root))
    }


def write_labels(dst_csv: Path, source_order_map: dict[str, int], tag: str) -> None:
    if not SRC_LABELS_CSV.exists():
        return

    fieldnames, rows = read_csv_rows(SRC_LABELS_CSV)
    video_col = detect_video_path_column(fieldnames)
    if video_col is None:
        log(
            f"[WARN] labels.csv에서 video path 컬럼을 찾지 못했습니다: {SRC_LABELS_CSV}"
        )
        return

    ordered_rows: list[tuple[int, dict[str, str]]] = []
    for row in rows:
        original_path = normalize_rel_path(row.get(video_col, ""))
        if original_path not in source_order_map:
            continue
        new_row = dict(row)
        new_row[video_col] = build_augmented_rel_path(original_path, tag)
        ordered_rows.append((source_order_map[original_path], new_row))

    ordered_rows.sort(key=lambda item: item[0])
    write_csv_rows(dst_csv, fieldnames, [row for _, row in ordered_rows])


def process_variant() -> None:
    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)

    video_output_dir = OUT_ROOT / "videos"
    ensure_dir(video_output_dir)

    video_files = list_video_files(SRC_VIDEO_DIR)
    if not video_files:
        log(f"[WARN] 증강할 영상이 없음: {SRC_VIDEO_DIR}")
        return

    source_order_map = build_video_order_map(SRC_VIDEO_DIR)
    started_at = time.perf_counter()
    total_videos = len(video_files)

    for idx, src_video in enumerate(video_files, start=1):
        rel_from_root = src_video.relative_to(SRC_VIDEO_DIR)
        dst_video = (
            video_output_dir
            / rel_from_root.parent
            / f"{src_video.stem}_arg2{src_video.suffix}"
        )

        elapsed = time.perf_counter() - started_at
        processed_count = idx - 1
        average_seconds = elapsed / processed_count if processed_count else 0.0
        remaining_videos = total_videos - processed_count
        eta_seconds = average_seconds * remaining_videos
        progress = (processed_count / total_videos) * 100

        log(
            f"\n[{idx}/{total_videos}] {rel_from_root.as_posix()} | progress={progress:.1f}% | elapsed={format_duration(elapsed)} | eta={format_duration(eta_seconds)}"
        )

        try:
            stats = process_video(src_video, dst_video, CONFIG)
        except Exception as error:
            log(f"  [ERROR] arg2: {error}")
            continue

        log(
            f"  -> saved | scale={CONFIG['resize_scale']} | blur={CONFIG['blur_type']}({CONFIG['blur_kernel']}) | jpeg_q={CONFIG['jpeg_quality']} | frames={stats['frame_count']}"
        )

    write_labels(OUT_ROOT / "labels.csv", source_order_map, "arg2")
    log(f"\nDone arg2. Output: {OUT_ROOT}")


def main() -> None:
    if not SRC_VIDEO_DIR.exists():
        raise FileNotFoundError(f"영상 폴더 없음: {SRC_VIDEO_DIR}")
    if not has_video_dependencies():
        raise RuntimeError("cv2/numpy가 없어 증강을 수행할 수 없습니다.")
    process_variant()


if __name__ == "__main__":
    main()
