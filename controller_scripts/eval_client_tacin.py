"""eval_client_tacin.py — eval client for the TACTILE-INPUT (oracle) policies.

Identical to eval_client.py EXCEPT it feeds the per-object 128-d TacGen latent
into the policy as `tacgen_input` every step. Use this ONLY for the
tacin models (pi05_xarm_tacin / _shuffled / _random), which were trained to
expect that input. For vision / aux-tactile policies use eval_client.py instead.

The latent comes from object_latents.npz (keys: 'objects' [names], 'latents'
[N x 128]); you pass --object <name> and it looks up that object's latent. This
is the "oracle" step: you tell it which object is present (the real-robot analog
of the paper's contact-local RGB). The SAME latent is fed for the whole trial.

Setup on Thor:
  * copy object_latents.npz next to this script (from the server:
    /export/wy891/home/_tacgen_probe_tmp/object_latents.npz)
  * the tacin policy must be SERVED from the dev env (tactile_input_model), since
    only that code understands the tactile-input token. Same serve_policy command,
    just sourced from setup_env_newmodel.sh and pointed at the dev checkpoints.

Run:
  python3 eval_client_tacin.py --host localhost --prompt "put the cube into the bowl" \
      --object sponge_stack --policy tacin
"""
import argparse
import csv
import math
import os
import time
from datetime import datetime

import numpy as np
from PIL import Image

from xarm.wrapper import XArmAPI
import pyrealsense2 as rs

from openpi_client import websocket_client_policy


# ---------------- config you may need to change ----------------
ROBOT_IP = "192.168.1.230"
CAM_BASE_SERIAL = "215222078407"   # cam_1 in training -> "image" (base view)
CAM_WRIST_SERIAL = "845112070404"  # cam_0 in training -> "wrist_image"

CONTROL_HZ = 10
EXEC_HORIZON = 8
SPEED = 80
MVACC = 1000

MAX_DXYZ_MM = 30.0
MAX_DROT_DEG = 8.0

WS_ENABLE = False
WS_MIN = [150.0, -400.0, 100.0]
WS_MAX = [650.0,  400.0, 500.0]

GRIPPER_OPEN = 850
GRIPPER_CLOSE = 0


# ---------------- cameras ----------------
class Cameras:
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
        frames = self.pipes[i].wait_for_frames(timeout_ms=1000)
        f = frames.get_color_frame()
        raw = np.asanyarray(f.get_data())
        import cv2
        return cv2.cvtColor(raw.view(np.uint8).reshape(1080, 1920, 2), cv2.COLOR_YUV2BGR_YUYV)

    def close(self):
        for p in self.pipes:
            try:
                p.stop()
            except Exception:
                pass


def crop_resize(bgr, mode, crop=1080, target=224):
    rgb = bgr[:, :, ::-1]
    img = Image.fromarray(rgb)
    w, h = img.size
    if mode == "left_540":
        left = 540
    elif mode == "right":
        left = (w - 80) - crop
    else:
        raise ValueError(mode)
    box = (left, 0, left + crop, crop)
    out = img.crop(box).resize((target, target), Image.LANCZOS)
    return np.asarray(out, dtype=np.uint8)


def load_object_latent(latents_file, object_name):
    """Return the 128-d float32 latent for object_name from object_latents.npz."""
    if not os.path.exists(latents_file):
        raise SystemExit(f"latents file not found: {latents_file} "
                         f"(copy it from the server: /export/wy891/home/_tacgen_probe_tmp/object_latents.npz)")
    d = np.load(latents_file, allow_pickle=True)
    names = [str(x) for x in d["objects"]]
    latents = d["latents"]
    if object_name not in names:
        raise SystemExit(f"object {object_name!r} not in latents file. Available: {names}")
    lat = np.asarray(latents[names.index(object_name)], dtype=np.float32)
    assert lat.shape == (128,), f"expected 128-d latent, got {lat.shape}"
    return lat


# ---------------- arm ----------------
def setup_arm():
    arm = XArmAPI(ROBOT_IP)
    arm.clean_error()
    arm.clean_warn()
    arm.motion_enable(enable=True)
    arm.set_mode(0)
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
    ap.add_argument("--host", required=True, help="GPU server IP/hostname (use localhost via the SSH tunnel)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--prompt", required=True, help="task text, e.g. 'put the cube into the bowl'")
    ap.add_argument("--object", required=True,
                    help="object present. With --latent-npy this is just a CSV label; otherwise it MUST "
                         "match a name in the latents file (e.g. sponge_stack, wooden_block)")
    ap.add_argument("--latents-file", default="object_latents.npz", help="per-object TacGen latents (.npz)")
    ap.add_argument("--latent-npy", default=None,
                    help="path to a single 128-d .npy latent (for a NEW/unseen object; see compute_latent.py). "
                         "Overrides the --latents-file lookup.")
    ap.add_argument("--max-steps", type=int, default=650)
    ap.add_argument("--dry-run", action="store_true",
                    help="perceive + infer + PRINT the action chunk, but NEVER move the arm (diagnostic)")
    ap.add_argument("--policy", default="tacin", help="policy label for the results CSV")
    ap.add_argument("--log", default="eval_results.csv")
    args = ap.parse_args()

    if args.latent_npy:
        tac_latent = np.asarray(np.load(args.latent_npy), dtype=np.float32).reshape(-1)
        if tac_latent.shape != (128,):
            raise SystemExit(f"{args.latent_npy}: expected a 128-d latent, got {tac_latent.shape}")
        print(f"loaded tactile latent from {args.latent_npy} for NEW object {args.object!r} "
              f"(norm={np.linalg.norm(tac_latent):.3f})")
    else:
        tac_latent = load_object_latent(args.latents_file, args.object)  # (128,) float32
        print(f"loaded tactile latent for object {args.object!r} (norm={np.linalg.norm(tac_latent):.3f})")

    cams = Cameras([CAM_BASE_SERIAL, CAM_WRIST_SERIAL])
    arm = setup_arm()
    client = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    print(f"connected to policy server at {args.host}:{args.port}")
    print(f"prompt: {args.prompt!r}")
    if args.dry_run:
        print(">>> DRY RUN: reading cameras/state + feeding tactile latent, printing actions ONLY. Arm will NOT move.")
        input(">>> Press ENTER to start the dry run (Ctrl-C to stop)...")
    else:
        input(">>> hand on the E-STOP. Press ENTER to start the trial (Ctrl-C to abort)...")

    dt = 1.0 / CONTROL_HZ
    gripper_closed = False
    step_count = 0

    try:
        while step_count < args.max_steps:
            base = crop_resize(cams.latest_bgr(0), "left_540")
            wrist = crop_resize(cams.latest_bgr(1), "right")
            _, joints_deg = arm.get_servo_angle()
            state = np.array(joints_deg[:6], dtype=np.float32) * math.pi / 180.0

            obs = {
                "observation/image": base,
                "observation/wrist_image": wrist,
                "observation/state": state,
                # NOTE: bare key "tacgen_input" (NOT "observation/tacgen_input") — that's the
                # key XarmInputs reads, matching the training repack. The ONLY difference from
                # eval_client.py.
                "tacgen_input": tac_latent,
                "prompt": args.prompt,
            }

            result = client.infer(obs)
            actions = np.asarray(result["actions"])

            if args.dry_run:
                print(f"\n[infer @ step {step_count}] action chunk (cm, cm, cm, rad, rad, rad, grip):")
                for i in range(min(EXEC_HORIZON, actions.shape[0])):
                    a = actions[i]
                    print(f"   {i}: "
                          f"[{a[0]:+.3f} {a[1]:+.3f} {a[2]:+.3f} | "
                          f"{a[3]:+.3f} {a[4]:+.3f} {a[5]:+.3f} | {a[6]:+.2f}]")
                step_count += min(EXEC_HORIZON, actions.shape[0])
                time.sleep(0.5)
                continue

            _, cur = arm.get_position()
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

                arm.set_position(*target, speed=SPEED, mvacc=MVACC, wait=True, is_radian=False)

                want_close = a[6] > 0
                if want_close != gripper_closed:
                    gripper_closed = want_close
                    arm.set_gripper_position(GRIPPER_CLOSE if gripper_closed else GRIPPER_OPEN, wait=False)

                print(f"step {step_count:4d}  d_mm=[{dxyz[0]:+5.1f},{dxyz[1]:+5.1f},{dxyz[2]:+5.1f}] "
                      f"grip={'C' if gripper_closed else 'O'}   ", end="\r")
                step_count += 1

        print("\nreached max-steps.")

    except KeyboardInterrupt:
        print("\naborted.")
    finally:
        try:
            arm.set_state(4)
            time.sleep(0.2)
        except Exception:
            pass
        cams.close()
        if not args.dry_run:
            res = input("\ntrial result — type 's' for SUCCESS, anything else for FAIL: ").strip().lower()
            success = res == "s"
            print("SUCCESS" if success else "FAIL")
            new_file = not os.path.exists(args.log)
            with open(args.log, "a", newline="") as f:
                w = csv.writer(f)
                if new_file:
                    w.writerow(["timestamp", "policy", "object", "prompt", "result", "steps"])
                w.writerow([datetime.now().isoformat(timespec="seconds"), args.policy,
                            args.object, args.prompt, "success" if success else "fail", step_count])
            print(f"logged -> {os.path.abspath(args.log)}")
        arm.disconnect()


if __name__ == "__main__":
    main()
