#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Dict, Iterable


ZERO_VEHICLE_ENV = "SAM3_VEHICLE_ZERO_JSON"
NO_POINTS_ERROR_MESSAGE = "No points are provided; please add points first"
VEHICLE_PROMPT = "vehicle"
LOWER_VEHICLE_THRESHOLD = 0.35
RETRY_FRAME_FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)


class BaseModuleHandle:
    def __init__(self, module: ModuleType) -> None:
        self._module = module

    def get_run_prompt(self) -> Callable[..., Dict[str, Any]]:
        return self._module.run_prompt

    def set_run_prompt(self, value: Callable[..., Dict[str, Any]]) -> None:
        setattr(self._module, "run_prompt", value)

    def main(self) -> None:
        self._module.main()

    def set_grad_enabled(self, enabled: bool) -> None:
        self._module.torch.set_grad_enabled(enabled)


def load_base_module() -> BaseModuleHandle:
    module_path = Path(__file__).with_name("run_sam3_1_objects_to_single_mask_video.py")
    spec = importlib.util.spec_from_file_location("sam3_single_mask_vehicle_recovery_base", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load base script from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return BaseModuleHandle(module)


def load_zero_vehicle_config() -> dict[str, dict[str, Any]]:
    config_path = os.environ.get(ZERO_VEHICLE_ENV, "").strip()
    if not config_path:
        raise RuntimeError(f"{ZERO_VEHICLE_ENV} is required")
    path = Path(config_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    result: dict[str, dict[str, Any]] = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        run_name = str(item.get("run", "")).strip()
        if run_name:
            result[run_name] = item
    if not result:
        raise RuntimeError(f"No target runs found in {path}")
    return result


def build_retry_frames(frame_count: int, base_frame_index: int) -> list[int]:
    frames: list[int] = []
    max_index = max(frame_count - 1, 0)
    seed = min(max(base_frame_index, 0), max_index)
    frames.append(seed)
    for fraction in RETRY_FRAME_FRACTIONS:
        candidate = int(round(max_index * fraction))
        if candidate not in frames:
            frames.append(candidate)
    return frames


base = load_base_module()
original_run_prompt = base.get_run_prompt()
ZERO_VEHICLE_CONFIG = load_zero_vehicle_config()


def iter_prompt_stream(predictor: Any, session_id: str, propagation_direction: str, output_prob_thresh: float) -> Iterable[Dict[str, Any]]:
    return base._module.prompt_stream(
        predictor=predictor,
        session_id=session_id,
        propagation_direction=propagation_direction,
        output_prob_thresh=output_prob_thresh,
    )


def collect_attempt(
    predictor: Any,
    session_id: str,
    prompt: str,
    args: Any,
    render_buffer: Any,
    color_map: Dict[str, list[int]],
    prompt_category_map: Dict[str, str],
    frames_with_union_mask: Any,
    prompt_frame_index: int,
    output_prob_thresh: float,
    propagation_direction: str,
) -> Dict[str, Any]:
    attempt_start = time.perf_counter()
    predictor.handle_request({"type": "reset_session", "session_id": session_id})

    add_prompt_response = predictor.handle_request(
        {
            "type": "add_prompt",
            "session_id": session_id,
            "frame_index": prompt_frame_index,
            "text": prompt,
            "output_prob_thresh": output_prob_thresh,
        }
    )
    add_outputs = add_prompt_response.get("outputs", {})
    add_masks = base._module.normalize_masks(add_outputs.get("out_binary_masks", base._module.np.zeros((0, 0, 0))))
    objects_on_add_prompt_frame = int(add_masks.shape[0])
    unique_object_ids: set[int] = set(
        int(value)
        for value in base._module.np.asarray(add_outputs.get("out_obj_ids", []), dtype=base._module.np.int64).tolist()
    )

    processed_frame_indices: set[int] = set()
    frames_with_mask = 0
    max_objects_in_frame = objects_on_add_prompt_frame

    if objects_on_add_prompt_frame > 0:
        for item in iter_prompt_stream(
            predictor=predictor,
            session_id=session_id,
            propagation_direction=propagation_direction,
            output_prob_thresh=output_prob_thresh,
        ):
            frame_index = int(item["frame_index"])
            if frame_index in processed_frame_indices:
                continue

            outputs = item.get("outputs", {})
            masks = base._module.normalize_masks(outputs.get("out_binary_masks", base._module.np.zeros((0, 0, 0))))
            obj_ids = base._module.np.asarray(outputs.get("out_obj_ids", []), dtype=base._module.np.int64)
            unique_object_ids.update(int(value) for value in obj_ids.tolist())
            object_count = int(masks.shape[0])
            max_objects_in_frame = max(max_objects_in_frame, object_count)
            processed_frame_indices.add(frame_index)

            if object_count == 0:
                continue

            prompt_union_mask = base._module.np.any(masks, axis=0)
            if not prompt_union_mask.any():
                continue

            frames_with_mask += 1
            frames_with_union_mask[frame_index] = True
            base._module.blend_mask_into_frame(
                frame=render_buffer[frame_index],
                mask=prompt_union_mask,
                color_rgb=color_map[prompt],
                alpha=args.alpha,
            )

    runtime_sec = round(time.perf_counter() - attempt_start, 2)
    return {
        "prompt": prompt,
        "category": prompt_category_map.get(prompt, prompt),
        "objects_on_add_prompt_frame": objects_on_add_prompt_frame,
        "frames_with_mask": frames_with_mask,
        "max_objects_in_frame": max_objects_in_frame,
        "unique_object_ids": sorted(unique_object_ids),
        "total_unique_objects": len(unique_object_ids),
        "runtime_sec": runtime_sec,
        "attempt_prompt_frame_index": prompt_frame_index,
        "output_prob_thresh_used": output_prob_thresh,
        "propagation_direction_used": propagation_direction,
    }


def run_prompt_vehicle_recovery(
    predictor: Any,
    session_id: str,
    prompt: str,
    args: Any,
    render_buffer: Any,
    color_map: Dict[str, list[int]],
    prompt_category_map: Dict[str, str],
    frames_with_union_mask: Any,
) -> Dict[str, Any]:
    run_name = Path(args.output_dir).name
    target_info = ZERO_VEHICLE_CONFIG.get(run_name)
    if prompt != VEHICLE_PROMPT or target_info is None:
        stats = original_run_prompt(
            predictor=predictor,
            session_id=session_id,
            prompt=prompt,
            args=args,
            render_buffer=render_buffer,
            color_map=color_map,
            prompt_category_map=prompt_category_map,
            frames_with_union_mask=frames_with_union_mask,
        )
        stats["status"] = stats.get("status") or "done"
        stats["error"] = stats.get("error")
        return stats

    vehicle_status = str(target_info.get("status") or "").strip()
    threshold_override = LOWER_VEHICLE_THRESHOLD if vehicle_status != "skipped_no_points" else float(args.output_prob_thresh)
    candidate_frames = build_retry_frames(int(render_buffer.shape[0]), int(args.prompt_frame_index))
    attempt_errors: list[str] = []

    for attempt_index, prompt_frame_index in enumerate(candidate_frames, start=1):
        propagation_direction = args.propagation_direction if prompt_frame_index == int(args.prompt_frame_index) else "both"
        try:
            stats = collect_attempt(
                predictor=predictor,
                session_id=session_id,
                prompt=prompt,
                args=args,
                render_buffer=render_buffer,
                color_map=color_map,
                prompt_category_map=prompt_category_map,
                frames_with_union_mask=frames_with_union_mask,
                prompt_frame_index=prompt_frame_index,
                output_prob_thresh=threshold_override,
                propagation_direction=propagation_direction,
            )
        except RuntimeError as exc:
            if NO_POINTS_ERROR_MESSAGE not in str(exc):
                raise
            attempt_errors.append(f"frame {prompt_frame_index}: {exc}")
            continue

        stats["attempted_prompt_frame_indices"] = candidate_frames[:attempt_index]
        if stats["frames_with_mask"] > 0:
            stats["status"] = "done" if attempt_index == 1 and threshold_override == float(args.output_prob_thresh) else "done_after_retry"
            stats["error"] = None
            return stats
        attempt_errors.append(
            f"frame {prompt_frame_index}: zero mask (objects_on_add_prompt_frame={stats['objects_on_add_prompt_frame']}, total_unique_objects={stats['total_unique_objects']})"
        )

    return {
        "prompt": prompt,
        "category": prompt_category_map.get(prompt, prompt),
        "objects_on_add_prompt_frame": 0,
        "frames_with_mask": 0,
        "max_objects_in_frame": 0,
        "unique_object_ids": [],
        "total_unique_objects": 0,
        "runtime_sec": 0.0,
        "attempted_prompt_frame_indices": candidate_frames,
        "output_prob_thresh_used": threshold_override,
        "status": "skipped_no_points" if vehicle_status == "skipped_no_points" else "retry_exhausted",
        "error": "; ".join(attempt_errors) if attempt_errors else "vehicle retry exhausted without detections",
    }


base.set_run_prompt(run_prompt_vehicle_recovery)


def main() -> None:
    base.main()


if __name__ == "__main__":
    base.set_grad_enabled(False)
    main()
