#!/usr/bin/env python3

from pathlib import Path
import runpy


if __name__ == "__main__":
    runpy.run_path(
        str(
            Path(
                "/workspace/yuyeon/run_sam3_1_objects_to_single_mask_video_collision_obstacle_recovery.py"
            )
        ),
        run_name="__main__",
    )
