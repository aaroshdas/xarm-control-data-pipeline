"""eval_client.py — let a trained pi0.5 policy drive the real xArm 850.

This is the "body" side of the brain/body split (see RUNBOOK). The "brain"
is openpi's serve_policy.py running on the GPU server, which holds ONE
checkpoint and answers with 7-dim actions. This script, running on the Thor
machine that owns the arm, does the loop:

    1. read the two cameras + the arm's joint angles
    2. send that observation to the policy server (over the network)
    3. get back an action chunk (10 future steps)
    4. execute those steps on the arm as end-effector position moves
    5. repeat, and let you mark each trial success / fail

IMPORTANT — the action format (must match setting1_raw2lerobot.py exactly):
    action[0:3] = delta end-effector XYZ in CENTIMETERS  -> *10 = mm
    action[3:6] = delta end-effector rotation in RADIANS  -> *180/pi = deg
    action[6]   = gripper:  > 0 -> close,  < 0 -> open
The state we send is the 6 joint angles in RADIANS (SDK gives degrees).
Base image  = cam_1 (215222078407), cropped 'left_540' then resized to 224.
Wrist image = cam_0 (845112070404), cropped 'right'     then resized to 224.

SAFETY — this MOVES A REAL ARM under model control:
  * KEEP A HAND ON THE E-STOP, especially the first runs.
  * Every per-step delta is CLAMPED (MAX_DXYZ_MM / MAX_DROT_DEG) so a bad
    prediction can't command a huge jump.
  * Optional workspace box (WS_MIN/WS_MAX) refuses targets outside a safe
    region — set these to your table before trusting it.
  * Start with SPEED low. Ctrl-C sends a stop and disconnects.

Run (on Thor, in the venv that has xarm SDK + pyrealsense2 + openpi-client):
    python3 eval_client.py --host <GPU_SERVER_IP> --prompt "put the red cube into the plastic cup"
"""
import argparse
import math
import time

import numpy as np
from PIL import Image

from xarm.wrapper import XArmAPI
import pyrealsense2 as rs

# openpi's lightweight client (install: pip install -e packages/openpi-client
# from the openpi repo, or `pip install openpi-client`). Only needs numpy /
# msgpack / websockets — no JAX, nothing heavy.
from openpi_client import websocket_client_policy


# ---------------- config you may need to change ----------------
ROBOT_IP = "192.168.1.230"
CAM_BASE_SERIAL = "215222078407"   # cam_1 in training -> "image" (base view)
CAM_WRIST_SERIAL = "845112070404"  # cam_0 in training -> "wrist_image"

CONTROL_HZ = 10                    # MUST match the training fps (10 Hz)
EXEC_HORIZON = 8                   # steps of the 10-step chunk to run before re-inferring
SPEED = 80                         # mm/s cap for set_position (start low, raise later)
MVACC = 1000                       # mm/s^2 accel cap

# per-step safety clamps — a single 10Hz action can't exceed these
MAX_DXYZ_MM = 30.0                 # max position move per step (mm)
MAX_DROT_DEG = 8.0                 # max rotation per step (deg)

# optional workspace box in mm [x,y,z]; set to your table then flip WS_ENABLE
WS_ENABLE = False
WS_MIN = [150.0, -400.0, 100.0]
WS_MAX = [650.0,  400.0, 500.0]

GRIPPER_OPEN = 850
GRIPPER_CLOSE = 0


# ---------------- cameras ----------------
class Cameras:
    """Grab the newest 1920x1080 color frame from each RealSense, like the collector."""

    def __init__(self, serials):
        self.pipes = []
        for sn in serials:
            p = rs.pipeline()
            cfg = rs.config()
            cfg.enable_device(sn)
            cfg.enable_stream(rs.stream.color, 1920, 1080, rs.format.yuyv, 30)
            p.start(cfg)
            self.pipes.append(p)

    def latest_bgr(self, i):
        # wait_for_frames returns the current frame; we call it right before infer
        frames = self.pipes[i].wait_for_frames(timeout_ms=1000)
        f = frames.get_color_frame()
        raw = np.asanyarray(f.get_data())
        import cv2  # local import so the file loads even if cv2 missing until run
        return cv2.cvtColor(raw.view(np.uint8).reshape(1080, 1920, 2), cv2.COLOR_YUV2BGR_YUYV)

    def close(self):
        for p in self.pipes:
            try:
                p.stop()
            except Exception:
                pass


def crop_resize(bgr, mode, crop=1080, target=224):
    """Replicate setting1_raw2lerobot.crop_and_resize EXACTLY, on an in-memory frame.

    Training loaded the saved .jpg with PIL (RGB) then cropped+LANCZOS-resized.
    So: BGR -> RGB, same crop box, same LANCZOS resize, return uint8 HWC RGB.
    """
    rgb = bgr[:, :, ::-1]  # BGR -> RGB
    img = Image.fromarray(rgb)
    w, h = img.size  # 1920, 1080
    if mode == "left_540":
        left = 540
    elif mode == "right":
        left = (w - 80) - crop     # right edge at w-80, width `crop`
    else:
        raise ValueError(mode)
    box = (left, 0, left + crop, crop)
    out = img.crop(box).resize((target, target), Image.LANCZOS)
    return np.asarray(out, dtype=np.uint8)


# ---------------- arm ----------------
def setup_arm():
    arm = XArmAPI(ROBOT_IP)
    arm.clean_error()
    arm.clean_warn()
    arm.motion_enable(enable=True)
    arm.set_mode(0)        # 0 = position mode (set_position moves to absolute pose)
    arm.set_state(0)
    arm.set_gripper_enable(True)
    arm.set_gripper_mode(0)
    arm.set_gripper_speed(3000)
    time.sleep(0.5)
    _, (err, warn) = arm.get_err_warn_code()
    print(f"arm: mode={arm.mode} state={arm.state} error={err} warn={warn}")
    if err:
        print(f"ERROR {err} — clear it in UFactory Studio, close Studio, rerun.")
        arm.disconnect()
        raise SystemExit(1)
    print("robot ready (position mode).")
    return arm


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True, help="GPU server IP/hostname running serve_policy")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--prompt", required=True, help="task text, e.g. 'put the red cube into the plastic cup'")
    ap.add_argument("--max-steps", type=int, default=300, help="hard cap on control steps per trial")
    ap.add_argument("--dry-run", action="store_true",
                    help="perceive + infer + PRINT the action chunk, but NEVER move the arm (diagnostic)")
    args = ap.parse_args()

    cams = Cameras([CAM_BASE_SERIAL, CAM_WRIST_SERIAL])
    arm = setup_arm()
    client = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    print(f"connected to policy server at {args.host}:{args.port}")
    print(f"prompt: {args.prompt!r}")
    if args.dry_run:
        print(">>> DRY RUN: reading cameras/state and printing actions ONLY. The arm will NOT move.")
        input(">>> Press ENTER to start the dry run (Ctrl-C to stop)...")
    else:
        input(">>> hand on the E-STOP. Press ENTER to start the trial (Ctrl-C to abort)...")

    dt = 1.0 / CONTROL_HZ
    gripper_closed = False
    step_count = 0

    try:
        while step_count < args.max_steps:
            # ---- build the observation (post-repack keys XarmInputs reads) ----
            base = crop_resize(cams.latest_bgr(0), "left_540")
            wrist = crop_resize(cams.latest_bgr(1), "right")
            _, joints_deg = arm.get_servo_angle()          # degrees
            state = np.array(joints_deg[:6], dtype=np.float32) * math.pi / 180.0

            obs = {
                "observation/image": base,
                "observation/wrist_image": wrist,
                "observation/state": state,
                "prompt": args.prompt,
            }

            # ---- ask the policy ----
            result = client.infer(obs)
            actions = np.asarray(result["actions"])         # [H, 7], real units (cm/rad)

            # Show the raw chunk the policy returned so we can judge sanity.
            # Columns are the model's real-unit deltas: xyz(cm), rot(rad), gripper.
            print(f"\n[infer @ step {step_count}] action chunk (cm, cm, cm, rad, rad, rad, grip):")
            for i in range(min(EXEC_HORIZON, actions.shape[0])):
                a = actions[i]
                print(f"   {i}: "
                      f"[{a[0]:+.3f} {a[1]:+.3f} {a[2]:+.3f} | "
                      f"{a[3]:+.3f} {a[4]:+.3f} {a[5]:+.3f} | {a[6]:+.2f}]")

            if args.dry_run:
                # look only — never touch the arm
                step_count += min(EXEC_HORIZON, actions.shape[0])
                time.sleep(0.5)
                continue

            # ---- execute the first EXEC_HORIZON steps as position moves ----
            # anchor on the arm's ACTUAL current pose each chunk (corrects drift),
            # then accumulate the per-step deltas onto a running target.
            _, cur = arm.get_position()                     # [x,y,z (mm), roll,pitch,yaw (deg)]
            target = list(cur)

            for i in range(min(EXEC_HORIZON, actions.shape[0])):
                a = actions[i]
                dxyz = [clamp(float(a[j]) * 10.0, -MAX_DXYZ_MM, MAX_DXYZ_MM) for j in range(3)]
                drot = [clamp(float(a[3 + j]) * 180.0 / math.pi, -MAX_DROT_DEG, MAX_DROT_DEG) for j in range(3)]

                for j in range(3):
                    target[j] += dxyz[j]
                    target[3 + j] += drot[j]

                if WS_ENABLE:
                    for j in range(3):
                        if not (WS_MIN[j] <= target[j] <= WS_MAX[j]):
                            print(f"\n[safety] target axis {j}={target[j]:.0f} out of workspace box — stopping.")
                            raise KeyboardInterrupt

                # wait=True => one move completes before the next (NO buffered queue
                # that could keep running after Ctrl-C). Slower but safe & non-glitchy.
                arm.set_position(*target, speed=SPEED, mvacc=MVACC, wait=True, is_radian=False)

                # gripper: edge-triggered on the sign of action[6]
                want_close = a[6] > 0
                if want_close != gripper_closed:
                    gripper_closed = want_close
                    arm.set_gripper_position(GRIPPER_CLOSE if gripper_closed else GRIPPER_OPEN, wait=False)

                step_count += 1

        print("\nreached max-steps.")

    except KeyboardInterrupt:
        print("\naborted.")
    finally:
        # HARD STOP: set_state(4) halts motion immediately and clears any queued
        # moves (set_state(0) does NOT — that was the runaway bug).
        try:
            arm.set_state(4)
            time.sleep(0.2)
        except Exception:
            pass
        cams.close()
        if not args.dry_run:
            res = input("\ntrial result — type 's' for SUCCESS, anything else for FAIL: ").strip().lower()
            print("SUCCESS" if res == "s" else "FAIL")
        arm.disconnect()


if __name__ == "__main__":
    main()
