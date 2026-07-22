#!/usr/bin/env python3
"""
Remove steps where action.delta_ee_pos is all zeros,
re-index steps starting from step_00000,
then recalculate delta_ee_pos and delta_joint_pos from consecutive observations.

  step[i].action.delta_ee_pos    = step[i+1].observations.ee_pos    - step[i].observations.ee_pos
  step[i].action.delta_joint_pos = step[i+1].observations.joint_pos - step[i].observations.joint_pos
  last step: both set to [0, 0, 0, 0, 0, 0]

Input:  /workspace/ghsun/real/data/nj/pick_package_new
Output: /workspace/ghsun/real/data/nj/pick_package_new_remove_zero
"""

import json
import os
import glob
import shutil

from config import CURRENT_DATASET_NAME
DATASET_NAME = CURRENT_DATASET_NAME

SRC_DIR = f"/home/zheyu/aarosh/datasets/{DATASET_NAME}"
DST_DIR = f"/home/zheyu/aarosh/datasets/{DATASET_NAME}_remove_zero"


def main():
    episodes = sorted(glob.glob(os.path.join(SRC_DIR, "episode_*")))
    print(f"Found {len(episodes)} episode(s)")

    for ep_src in episodes:
        ep_name = os.path.basename(ep_src)
        ep_dst = os.path.join(DST_DIR, ep_name)
        os.makedirs(ep_dst, exist_ok=True)

        steps = sorted(glob.glob(os.path.join(ep_src, "step_*")))
        total = len(steps)

        # === Pass 1: filter out zero-action steps, collect data ===
        kept_data = []
        kept_src_dirs = []
        removed = 0

        for step_dir in steps:
            data_path = os.path.join(step_dir, "data.json")
            with open(data_path) as f:
                data = json.load(f)

            delta = data["action"]["delta_ee_pos"]
            delta_gripper = data["action"]["delta_gripper"]
            if all(v == 0.0 for v in delta) and abs(delta_gripper) < 3:
                removed += 1
                continue

            kept_data.append(data)
            kept_src_dirs.append(step_dir)

        kept = len(kept_data)
        print(f"{ep_name}: {total} -> {kept} steps (removed {removed} zero-action steps)")

        # === Pass 2: recalculate deltas and write output ===
        for i in range(kept):
            new_step_dir = os.path.join(ep_dst, f"step_{i:05d}")
            os.makedirs(new_step_dir, exist_ok=True)

            data = kept_data[i]
            data["meta"]["step"] = i

            if i < kept - 1:
                # delta = next_obs - current_obs
                cur_ee = data["observations"]["ee_pos"]
                nxt_ee = kept_data[i + 1]["observations"]["ee_pos"]
                data["action"]["delta_ee_pos"] = [
                    nxt_ee[j] - cur_ee[j] for j in range(len(cur_ee))
                ]

                cur_jp = data["observations"]["joint_pos"]
                nxt_jp = kept_data[i + 1]["observations"]["joint_pos"]
                data["action"]["delta_joint_pos"] = [
                    nxt_jp[j] - cur_jp[j] for j in range(len(cur_jp))
                ]
            else:
                # Last step: zeros
                data["action"]["delta_ee_pos"] = [0.0] * len(data["action"]["delta_ee_pos"])
                data["action"]["delta_joint_pos"] = [0.0] * len(data["action"]["delta_joint_pos"])

            with open(os.path.join(new_step_dir, "data.json"), "w") as f:
                json.dump(data, f, indent=4)

            # Copy images
            for cam in ["cam_0.jpg", "cam_1.jpg"]:
                src_img = os.path.join(kept_src_dirs[i], cam)
                if os.path.exists(src_img):
                    shutil.copy2(src_img, os.path.join(new_step_dir, cam))

        print(f"  Recalculated delta_ee_pos & delta_joint_pos for {kept} steps")

    print(f"\nOutput saved to: {DST_DIR}")


if __name__ == "__main__":
    main()