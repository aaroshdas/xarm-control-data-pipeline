"""collector_aarosh.py — personal copy of vr/collector_new.py.

Only change from the original: base_dir defaults to a folder under
~/aarosh instead of the shared lab folder, so recordings don't land in
someone else's dataset directory. Everything else (schema, camera serials,
10Hz loop, delta computation) is identical — raw2lerobot etc. need no changes.
"""
import os, time, json, cv2, threading, queue

import numpy as np

from xarm.wrapper import XArmAPI

import pyrealsense2 as rs

from config import CURRENT_DATASET_NAME
DATASET_NAME = CURRENT_DATASET_NAME

DEFAULT_BASE_DIR = os.path.expanduser(f'~/aarosh/datasets/{DATASET_NAME}')
CAMERA_SERIALS = ['845112070404', '215222078407']  # same physical cameras as the lab pipeline
ROBOT_IP = "192.168.1.230"


class TeleopVLACollector(threading.Thread):
    def __init__(self, arm, serials, hz=10, base_dir=DEFAULT_BASE_DIR):
        super().__init__()
        self.arm = arm
        self.hz = hz
        self.interval = 1.0 / hz
        self.running = True
        self.save_queue = queue.Queue(maxsize=500)
        self.base_dir = base_dir

        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
            self.ep_idx = 0
        else:
            existing_eps = [d for d in os.listdir(self.base_dir) if d.startswith('episode_')]
            if not existing_eps:
                self.ep_idx = 0
            else:
                try:
                    indices = [int(d.split('_')[1]) for d in existing_eps if d.split('_')[1].isdigit()]
                    self.ep_idx = max(indices) + 1 if indices else 0
                except Exception:
                    self.ep_idx = 0

        self.latest_imgs = [None] * len(serials)
        self.pipelines = []
        for sn in serials:
            p = rs.pipeline()
            cfg = rs.config()
            cfg.enable_device(sn)
            cfg.enable_stream(rs.stream.color, 1920, 1080, rs.format.yuyv, 30)
            p.start(cfg)
            self.pipelines.append(p)

        threading.Thread(target=self._saver_worker, daemon=True).start()
        for i in range(len(self.pipelines)):
            threading.Thread(target=self._camera_worker, args=(i,), daemon=True).start()

    def _camera_worker(self, i):
        while self.running:
            try:
                frames = self.pipelines[i].wait_for_frames(timeout_ms=1000)
                f = frames.get_color_frame()
                if f:
                    raw = np.asanyarray(f.get_data())
                    img = cv2.cvtColor(raw.view(np.uint8).reshape(1080, 1920, 2), cv2.COLOR_YUV2BGR_YUYV) if raw.dtype == np.uint16 else cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
                    self.latest_imgs[i] = img
            except Exception:
                continue

    def _saver_worker(self):
        while self.running or not self.save_queue.empty():
            try:
                task = self.save_queue.get(timeout=1)
                path, imgs, data = task
                os.makedirs(path, exist_ok=True)
                for i, img in enumerate(imgs):
                    cv2.imwrite(os.path.join(path, f"cam_{i}.jpg"), img)
                with open(os.path.join(path, "data.json"), "w") as f:
                    json.dump(data, f, indent=4)
            except Exception:
                continue

    def run(self):
        while self.running:
            ep_path = os.path.join(self.base_dir, f"episode_{self.ep_idx:03d}")
            print(f"recording started: {ep_path}")
            step = 0
            history = []
            gripper_move_count = 0
            last_gripper_pos = None

            while self.running:
                t_start = time.perf_counter()

                _, pos = self.arm.get_position()
                _, joints_raw = self.arm.get_servo_angle()
                _, gripper_pos = self.arm.get_gripper_position()

                joints = joints_raw[:6]

                if last_gripper_pos is not None:
                    if abs(gripper_pos - last_gripper_pos) > 10.0:
                        gripper_move_count += 1
                last_gripper_pos = gripper_pos

                if all(img is not None for img in self.latest_imgs):
                    history.append({
                        "pos": pos,
                        "joints": joints,
                        "gripper": gripper_pos,
                        "imgs": list(self.latest_imgs)
                    })

                    if step > 0:
                        prev, curr = history[step - 1], history[step]
                        delta_pos = (np.array(curr["pos"]) - np.array(prev["pos"])).tolist()
                        delta_joints = (np.array(curr["joints"]) - np.array(prev["joints"])).tolist()
                        delta_gripper = curr["gripper"] - prev["gripper"]

                        vla_data = {
                            "observations": {
                                "ee_pos": prev["pos"],
                                "joint_pos": prev["joints"],
                                "gripper_pos": prev["gripper"]
                            },
                            "action": {
                                "delta_ee_pos": delta_pos,
                                "delta_joint_pos": delta_joints,
                                "delta_gripper": delta_gripper
                            },
                            "meta": {
                                "step": step - 1,
                                "total_gripper_moves": gripper_move_count
                            }
                        }
                        self.save_queue.put((os.path.join(ep_path, f"step_{step - 1:05d}"), prev["imgs"], vla_data))

                    step += 1

                time.sleep(max(0.01, self.interval - (time.perf_counter() - t_start)))

            self.ep_idx += 1


if __name__ == "__main__":
    arm = XArmAPI(ROBOT_IP)
    collector = TeleopVLACollector(arm, CAMERA_SERIALS)
    collector.daemon = True
    collector.start()

    print(f"ready. saving to: {collector.base_dir}  starting at episode {collector.ep_idx:03d}")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        collector.running = False
        print("\nsaving remaining data...")
        while not collector.save_queue.empty():
            time.sleep(0.1)
        print(f"done. saved to: {collector.base_dir}")
        time.sleep(0.5)
