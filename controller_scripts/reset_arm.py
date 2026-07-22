"""reset_arm.py — save a "home" pose once, return to it any time.

Two commands:

    python3 reset_arm.py save    # remember where the arm is RIGHT NOW as home
    python3 reset_arm.py go      # move the arm back to that saved home pose

Typical use: hand-guide (or drive) the arm to a good starting pose for your
task, run `save` once. Then between episodes, run `go` to snap back to the
exact same start every time — so all your demos begin from the same place.

The pose is stored in reset_pose.json next to this script (joint angles +
gripper). Uses the xArm SDK directly (no ROS2 needed), same as the collector.

IMPORTANT: stop bridge_aarosh.py before running `go` — only one program
should command the arm's motion at a time. `save` is read-only and safe
to run any time.
"""
import argparse
import json
import os
import sys

from xarm.wrapper import XArmAPI

ROBOT_IP = "192.168.1.230"
POSE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reset_pose.json")
MOVE_SPEED = 30            # deg/s — deliberately slow/safe for an automatic move
GRIPPER_OPEN = 850


def connect():
    arm = XArmAPI(ROBOT_IP)
    arm.clean_error()
    arm.motion_enable(enable=True)
    return arm


def do_save():
    arm = connect()
    code, angles = arm.get_servo_angle()
    if code != 0:
        sys.exit(f"could not read joint angles (code {code})")
    joints = angles[:6]
    gripper = None
    try:
        gcode, gpos = arm.get_gripper_position()
        if gcode == 0:
            gripper = gpos
    except Exception:
        pass
    with open(POSE_PATH, "w") as f:
        json.dump({"joints": joints, "gripper": gripper}, f, indent=2)
    print(f"saved home pose to {POSE_PATH}")
    print(f"  joints:  {[round(a, 2) for a in joints]}")
    print(f"  gripper: {gripper}")
    arm.disconnect()


def do_go():
    if not os.path.exists(POSE_PATH):
        sys.exit(f"no saved pose at {POSE_PATH} — run `python3 reset_arm.py save` first")
    with open(POSE_PATH) as f:
        pose = json.load(f)

    arm = connect()
    arm.set_mode(0)        # position mode (needed for set_servo_angle)
    arm.set_state(0)

    print(f"moving to saved home pose (speed {MOVE_SPEED} deg/s)...")
    arm.set_servo_angle(angle=pose["joints"], speed=MOVE_SPEED, wait=True)

    if pose.get("gripper") is not None:
        try:
            arm.set_gripper_enable(True)
            arm.set_gripper_mode(0)
            arm.set_gripper_position(pose["gripper"], wait=True)
        except Exception as e:
            print(f"  (gripper move skipped: {e})")

    print("done — arm is at home pose.")
    arm.disconnect()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["save", "go"], help="save current pose, or go to saved pose")
    args = ap.parse_args()
    if args.command == "save":
        do_save()
    else:
        do_go()


if __name__ == "__main__":
    main()
