#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, List


ZERO_VEHICLE_ENV = "SAM3_VEHICLE_ZERO_JSON"


class BaseModuleHandle:
    def __init__(self, module: ModuleType) -> None:
        self._module = module

    def get_load_entries(self) -> Callable[[Path], List[dict[str, Any]]]:
        return self._module.load_entries

    def set_load_entries(self, value: Callable[[Path], List[dict[str, Any]]]) -> None:
        setattr(self._module, "load_entries", value)

    def main(self) -> None:
        self._module.main()


def load_base_module() -> BaseModuleHandle:
    module_path = Path(__file__).with_name("run_sam3_test_sampled_frames.py")
    spec = importlib.util.spec_from_file_location("sam3_test_sampled_frames_base", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load base runner from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return BaseModuleHandle(module)


def load_target_runs() -> set[str]:
    config_path = os.environ.get(ZERO_VEHICLE_ENV, "").strip()
    if not config_path:
        raise RuntimeError(f"{ZERO_VEHICLE_ENV} is required")
    path = Path(config_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    target_runs: set[str] = set()
    for item in data:
        if not isinstance(item, dict):
            continue
        run_name = str(item.get("run", "")).strip()
        if run_name:
            target_runs.add(run_name)
    if not target_runs:
        raise RuntimeError(f"No target runs found in {path}")
    return target_runs


base = load_base_module()
original_load_entries = base.get_load_entries()
TARGET_RUNS = load_target_runs()


def load_entries_vehicle_zero_only(summary_json: Path) -> List[dict[str, Any]]:
    entries = original_load_entries(summary_json)
    filtered = [entry for entry in entries if Path(str(entry.get("filename", ""))).stem in TARGET_RUNS]
    print(f"[info] vehicle recovery filtered entries: {len(filtered)} / {len(entries)}")
    return filtered


base.set_load_entries(load_entries_vehicle_zero_only)


def main() -> None:
    base.main()


if __name__ == "__main__":
    main()
