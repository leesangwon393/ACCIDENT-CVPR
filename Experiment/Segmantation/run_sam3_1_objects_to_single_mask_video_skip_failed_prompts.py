#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Dict


NO_POINTS_ERROR_MESSAGE = "No points are provided; please add points first"


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
    spec = importlib.util.spec_from_file_location("sam3_single_mask_base", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load base script from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return BaseModuleHandle(module)


base = load_base_module()
original_run_prompt = base.get_run_prompt()


def run_prompt_fail_soft(
    predictor: Any,
    session_id: str,
    prompt: str,
    args: Any,
    render_buffer: Any,
    color_map: Dict[str, list[int]],
    prompt_category_map: Dict[str, str],
    frames_with_union_mask: Any,
) -> Dict[str, Any]:
    start = time.perf_counter()
    try:
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
        stats["status"] = "done"
        stats["error"] = None
        return stats
    except RuntimeError as exc:
        if NO_POINTS_ERROR_MESSAGE not in str(exc):
            raise
        runtime_sec = round(time.perf_counter() - start, 2)
        print(f"[warn] skipping prompt '{prompt}': {exc}")
        return {
            "prompt": prompt,
            "category": prompt_category_map.get(prompt, prompt),
            "objects_on_add_prompt_frame": 0,
            "frames_with_mask": 0,
            "max_objects_in_frame": 0,
            "unique_object_ids": [],
            "total_unique_objects": 0,
            "runtime_sec": runtime_sec,
            "status": "skipped_no_points",
            "error": str(exc),
        }


base.set_run_prompt(run_prompt_fail_soft)


def main() -> None:
    base.main()


if __name__ == "__main__":
    base.set_grad_enabled(False)
    main()
