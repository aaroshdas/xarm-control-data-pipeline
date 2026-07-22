import json
import os
import glob

from config import CURRENT_DATASET_NAME
DATASET_NAME = CURRENT_DATASET_NAME
DATA_DIR = f"/home/zheyu/aarosh/datasets/{DATASET_NAME}"

def normalize_angle(delta):
    """Wrap angle delta to [-180, 180]."""
    return (delta + 180) % 360 - 180

# Collect and sort episode directories
episode_dirs = sorted(glob.glob(os.path.join(DATA_DIR, "episode_*")))
print(f"Found {len(episode_dirs)} episodes")

fixed_count = 0
total_steps = 0
for episode_dir in episode_dirs:
    episode_name = os.path.basename(episode_dir)
    step_dirs = sorted(glob.glob(os.path.join(episode_dir, "step_*")))
    total_steps += len(step_dirs)
    episode_fixed = 0

    for step_dir in step_dirs:
        data_path = os.path.join(step_dir, "data.json")
        with open(data_path, "r") as f:
            data = json.load(f)

        delta = data["action"]["delta_ee_pos"]
        new_delta = list(delta)
        changed = False

        # Fix last 3 dimensions (indices 3, 4, 5) — angular components
        for i in range(3, 6):
            wrapped = normalize_angle(delta[i])
            if abs(wrapped - delta[i]) > 1e-6:
                changed = True
                new_delta[i] = wrapped

        if changed:
            step_name = os.path.basename(step_dir)
            print(f"  {episode_name}/{step_name}: {[f'{d:.4f}' for d in delta[3:]]} -> {[f'{d:.4f}' for d in new_delta[3:]]}")
            data["action"]["delta_ee_pos"] = new_delta
            with open(data_path, "w") as f:
                json.dump(data, f, indent=4)
            episode_fixed += 1

    if episode_fixed > 0:
        print(f"  [{episode_name}] fixed {episode_fixed}/{len(step_dirs)} steps")
    fixed_count += episode_fixed

print(f"\nTotal: fixed {fixed_count}/{total_steps} steps across {len(episode_dirs)} episodes")