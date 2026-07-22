"""
Script for converting xarm dataset to LeRobot format.
Optimized: Episode-Level Parallelism + Single Pass Statistics.

Usage:
python convert_our_xarm_data_to_lerobot.py --data_dir /workspace/ghsun/real/data/xarm

The resulting dataset will get saved to the $HF_LEROBOT_HOME directory.

Features:
- State: joint_pos (6 dims) - converted from degrees to radians
- Actions: delta_ee_pos (6 dims) + gripper_action (1 dim) = 7 dims
  - delta_ee_pos: [delta_x, delta_y, delta_z, delta_rx, delta_ry, delta_rz]
    - Position deltas (xyz): converted from mm to cm
    - Rotation deltas (rxryrz): converted from degrees to radians
  - gripper_action: binarized using delta_gripper edge detection
    - delta_gripper < -10 (closing) -> 1.0 (close)
    - delta_gripper > 10 (opening) -> -1.0 (open)
- Normalization: Computes mean/std from entire dataset and saves to stats.json
  (Training will apply normalization automatically using these stats)
"""

import json
import shutil
import concurrent.futures
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Union

import numpy as np
from PIL import Image
import tyro

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# REPO_NAME = "local/pick_packages_cm_rad_bool_15_nj_0221_resize_main"  # Name of the output dataset
# TASK_NAME = "take out the white package and place it on the conveyor belt with the label facing up"



# REPO_NAME = "xarm850-pickplace-multiobj-v1-tacgen"
REPO_NAME = "pickplace-tacgen"

def get_task_name(episode_idx: int) -> str:
    return 'put the cube into the bowl'

IMAGE_SIZE = 224  # Target image size (224x224)


# --- 统计累加器类 (用于一次遍历计算 Mean/Std) ---
@dataclass
class StatsAccumulator:
    count: int = 0
    sum_x: np.ndarray = None     # 累加和
    sum_sq_x: np.ndarray = None  # 平方和
    min_x: np.ndarray = None     # 最小值
    max_x: np.ndarray = None     # 最大值

    def update(self, data: np.ndarray):
        """Update stats with a single frame of data."""
        data_f64 = data.astype(np.float64)

        if self.count == 0:
            self.sum_x = np.zeros_like(data_f64)
            self.sum_sq_x = np.zeros_like(data_f64)
            self.min_x = data_f64.copy()
            self.max_x = data_f64.copy()

        self.count += 1
        self.sum_x += data_f64
        self.sum_sq_x += (data_f64 ** 2)
        self.min_x = np.minimum(self.min_x, data_f64)
        self.max_x = np.maximum(self.max_x, data_f64)

    def merge(self, other: 'StatsAccumulator'):
        """Merge another accumulator into this one."""
        if other.count == 0:
            return
        if self.count == 0:
            self.count = other.count
            self.sum_x = other.sum_x
            self.sum_sq_x = other.sum_sq_x
            self.min_x = other.min_x
            self.max_x = other.max_x
        else:
            self.count += other.count
            self.sum_x += other.sum_x
            self.sum_sq_x += other.sum_sq_x
            self.min_x = np.minimum(self.min_x, other.min_x)
            self.max_x = np.maximum(self.max_x, other.max_x)


def crop_and_resize(
    image_input: Union[Path, str],
    crop_size: int = 1080,
    target_size: int = 224,
    crop_mode: str = "center"
) -> np.ndarray:
    """
    读取、裁剪并缩放图片。
    接收 Path 对象以减少进程间序列化开销。
    """
    with Image.open(image_input) as image:
        image.load()
        width, height = image.size  # 1920, 1080

        if crop_mode == "center":
            left = (width - crop_size) // 2
            top = 0
            right = left + crop_size
            bottom = crop_size
            image_cropped = image.crop((left, top, right, bottom))
            image_resized = image_cropped.resize((target_size, target_size), Image.LANCZOS)
            return np.array(image_resized)

        elif crop_mode == "right":
            right = width - 80
            left = right - crop_size
            top = 0
            bottom = crop_size
            image_cropped = image.crop((left, top, right, bottom))
            image_resized = image_cropped.resize((target_size, target_size), Image.LANCZOS)
            return np.array(image_resized)

        elif crop_mode == "left_320":
            left = 320
            right = left + crop_size
            top = 0
            bottom = crop_size
            image_cropped = image.crop((left, top, right, bottom))
            image_resized = image_cropped.resize((target_size, target_size), Image.LANCZOS)
            return np.array(image_resized)

        elif crop_mode == "right_3/4":
            left = width // 4
            top = 0
            right = width
            bottom = height
            image_cropped = image.crop((left, top, right, bottom))
            image_resized = image_cropped.resize((target_size, target_size), Image.LANCZOS)
            return np.array(image_resized)

        elif crop_mode == "left_540":
            left = 540
            right = left + crop_size
            top = 0
            bottom = crop_size
            image_cropped = image.crop((left, top, right, bottom))
            image_resized = image_cropped.resize((target_size, target_size), Image.LANCZOS)
            return np.array(image_resized)

        elif crop_mode == "resize_only":
            image_resized = image.resize((target_size, target_size), Image.LANCZOS)
            return np.array(image_resized)

        else:
            raise ValueError(f"Invalid crop_mode: {crop_mode}")


# --- 子进程处理函数：处理单个 Episode ---
def process_episode(args: Tuple[Path, str]) -> Optional[Tuple[List[Dict], StatsAccumulator, StatsAccumulator]]:
    """
    处理一个 Episode 中的所有 Steps。
    返回：(帧数据列表, 状态统计, 动作统计)
    """
    episode_dir, task_name = args
    step_dirs = sorted([d for d in episode_dir.iterdir() if d.is_dir() and d.name.startswith("step_")])

    if not step_dirs:
        return None

    frames = []
    local_state_stats = StatsAccumulator()
    local_action_stats = StatsAccumulator()
    gripper_state = -1.0  # start as open (gripper_pos ~850)

    for step_dir in step_dirs:
        cam_0_path = step_dir / "cam_0.jpg"  # Wrist Camera
        cam_1_path = step_dir / "cam_1.jpg"  # Main Camera
        data_json_path = step_dir / "data.json"

        if not (cam_0_path.exists() and cam_1_path.exists() and data_json_path.exists()):
            print(f"Warning: Missing files in {step_dir}, skipping...")
            continue

        try:
            # 1. 图像处理: cam_0 用 right crop, cam_1 用 resize_only
            wrist_image_array = crop_and_resize(
                cam_0_path, crop_size=1080, target_size=IMAGE_SIZE, crop_mode="right"
            )
            main_image_array = crop_and_resize(
                cam_1_path, crop_size=1080, target_size=IMAGE_SIZE, crop_mode="left_540"
            )

            # 2. 读取 JSON 数据
            with open(data_json_path, 'r') as f:
                data = json.load(f)

            obs = data["observations"]
            act = data["action"]

            # 3. 处理 State: joint_pos (6 dims) -> radians
            state = np.array(obs["joint_pos"], dtype=np.float32) * np.pi / 180.0

            # 4. 处理 Actions: delta_ee_pos (6 dims) + gripper_action
            delta_ee = np.array(act["delta_ee_pos"], dtype=np.float32)
            delta_ee[0:3] = delta_ee[0:3] / 10.0            # mm -> cm
            delta_ee[3:6] = delta_ee[3:6] * np.pi / 180.0   # deg -> rad

            # Process gripper: use delta_gripper edge detection
            # delta_gripper < -10 -> start closing -> 1.0 (close)
            # delta_gripper > 10  -> start opening -> -1.0 (open)
            # otherwise keep previous state
            delta_g = act["delta_gripper"]
            if delta_g < -10:
                gripper_state = 1.0
            elif delta_g > 10:
                gripper_state = -1.0
            gripper_action = gripper_state

            actions = np.concatenate([delta_ee, [gripper_action]], dtype=np.float32)

            # 5. 组装 Frame 数据
            frame_data = {
                "image": main_image_array,
                "wrist_image": wrist_image_array,
                "state": state,
                "actions": actions,
                "task": task_name,
            }
            frames.append(frame_data)

            # 6. 更新统计
            local_state_stats.update(state)
            local_action_stats.update(actions)

        except Exception as e:
            print(f"Error processing {step_dir}: {e}")
            continue

    if not frames:
        return None

    return frames, local_state_stats, local_action_stats


# --- 主函数 ---
def main(data_dir: str = "/workspace/ghsun/real/0517_lab_xarm_data/setting1/lab_xarm_0517_0518_merged", *, push_to_hub: bool = False):
    # 1. 清理并初始化输出目录
    output_path = HF_LEROBOT_HOME / REPO_NAME
    if output_path.exists():
        shutil.rmtree(output_path)

    print(f"Processing data from: {data_dir}")
    print(f"Output to: {output_path}")

    # 2. 创建 LeRobotDataset
    # image_writer_processes=0 避免与处理进程冲突
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="xarm",
        fps=10,
        features={
            "image": {
                "dtype": "image",
                "shape": (IMAGE_SIZE, IMAGE_SIZE, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (IMAGE_SIZE, IMAGE_SIZE, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (6,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["actions"],
            },
        },
        image_writer_threads=5,
        image_writer_processes=0,
    )

    # 3. 扫描 Episodes
    data_path = Path(data_dir)
    episode_dirs = sorted([d for d in data_path.iterdir() if d.is_dir() and d.name.startswith("episode_")])
    print(f"Found {len(episode_dirs)} episodes. Starting parallel processing...")

    # 全局统计累加器
    global_state_stats = StatsAccumulator()
    global_action_stats = StatsAccumulator()

    # 4. 并行处理 (Episode Level)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # map 保证返回结果的顺序与 episode_dirs 一致
        episode_args = [(d, get_task_name(i)) for i, d in enumerate(episode_dirs)]
        results = executor.map(process_episode, episode_args)

        for i, result in enumerate(results):
            if result is None:
                print(f"Skipping empty/failed episode index {i}")
                continue

            frames, ep_state_stats, ep_action_stats = result

            # 写入当前 Episode 的所有帧
            for frame in frames:
                dataset.add_frame(frame)

            dataset.save_episode()

            # 合并统计数据
            global_state_stats.merge(ep_state_stats)
            global_action_stats.merge(ep_action_stats)

            print(f"Saved Episode {i} ({len(frames)} frames)")

    # 5. 计算并保存最终统计信息
    print("\nComputing final statistics...")

    def finalize_stats(acc: StatsAccumulator) -> Dict[str, List[float]]:
        if acc.count == 0:
            return {
                "mean": [0.0]*6, "std": [1.0]*6,
                "min": [0.0]*6, "max": [0.0]*6
            }

        mean = acc.sum_x / acc.count
        variance = (acc.sum_sq_x / acc.count) - (mean ** 2)
        variance = np.maximum(variance, 0)  # 修正浮点误差导致的负数
        std = np.sqrt(variance)

        return {
            "mean": mean.tolist(),
            "std": std.tolist(),
            "min": acc.min_x.tolist(),
            "max": acc.max_x.tolist(),
        }

    final_stats_data = {
        "state": finalize_stats(global_state_stats),
        "actions": finalize_stats(global_action_stats),
        "description": {
            "state": "joint_pos (6 dims) - converted from degrees to radians",
            "actions": "delta_ee_pos (6 dims) + gripper_action (1 dim)",
            "delta_ee_pos": "[delta_x, delta_y, delta_z, delta_rx, delta_ry, delta_rz] - xyz in cm, rotation in radians",
            "gripper_action": "Binarized: delta_gripper < -10 -> 1.0 (close), > 10 -> -1.0 (open)",
            "note": "Training will apply normalization automatically using these stats.",
            "normalization": "Training applies: (x - mean) / std",
            "denormalization": "Inference applies: x_original = x_normalized * std + mean"
        }
    }

    stats_file = output_path.parent / f"{REPO_NAME.replace('/', '_')}_normalization_stats.json"
    stats_file.parent.mkdir(parents=True, exist_ok=True)

    with open(stats_file, 'w') as f:
        json.dump(final_stats_data, f, indent=2)

    print(f"✓ Saved normalization stats to: {stats_file}")
    print(f"Conversion complete. Dataset saved to: {output_path}")

    if push_to_hub:
        dataset.push_to_hub(
            tags=["xarm", "pick_packages"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)