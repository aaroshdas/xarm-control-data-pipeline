"""xbox_control.py — drive the xArm with an Xbox One S controller instead of the VR headset.

bridge_new.py doesn't know or care where its UDP packets come from — it just
unpacks floats off port 5005 and turns them into arm velocity commands. The
VR headset is just one possible sender. This script is another one: it reads
the Xbox controller and sends the same packet shape, so bridge_new.py and
collector_new.py both keep working completely unchanged.

FIRST TIME / IF CONTROLS ARE WRONG: run  python3 calibrate.py  once. It
walks you through each control, auto-detects the right button/axis numbers,
and writes xbox_mapping.json. This script loads that file automatically.

Run this ON THOR (same machine as bridge_new.py), controller connected:

    python3 xbox_control.py            # normal teleop
    python3 xbox_control.py --test     # print interpreted values, no UDP sent
    python3 xbox_control.py --raw      # dump raw axis/button indices (debug)

Safety: motion only goes out while the dead-man button (RB by default) is
held down — let go and the arm gets a zero-velocity packet and stops.
"""
import argparse
import json
import os
import socket
import struct
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")  # Thor is headless; pygame still needs a video driver to init joysticks
import pygame

HOST_DEFAULT = "127.0.0.1"  # bridge_new.py binds 0.0.0.0:5005 on Thor itself
PORT = 5005
SEND_HZ = 30
DEADZONE = 0.15

MAPPING_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "xbox_mapping.json")

# Fallback mapping if xbox_mapping.json doesn't exist yet. These are guesses
# and are exactly what breaks between controllers — run calibrate.py to
# replace them with values measured from YOUR controller.
DEFAULT_MAPPING = {
    "deadman_button": 5,
    "gripper_button": 0,
    "axes": {
        "lx": {"index": 0, "rest": 0.0, "active": 1.0},
        "ly": {"index": 1, "rest": 0.0, "active": -1.0},   # active negative = up gives +
        "rx": {"index": 3, "rest": 0.0, "active": 1.0},
        "ry": {"index": 4, "rest": 0.0, "active": -1.0},
        "trig_up": {"index": 5, "rest": -1.0, "active": 1.0},
        "trig_down": {"index": 2, "rest": -1.0, "active": 1.0},
    },
}


def load_mapping():
    if os.path.exists(MAPPING_PATH):
        with open(MAPPING_PATH) as f:
            print(f"using calibrated mapping: {MAPPING_PATH}")
            return json.load(f)
    print("no xbox_mapping.json found — using built-in guesses. "
          "Run `python3 calibrate.py` if controls are wrong.")
    return DEFAULT_MAPPING


def norm_axis(js, cal):
    """Map a raw axis value to a calibrated range: rest -> 0, active -> +1.
    Sticks come out ~-1..+1 (direction baked in); triggers ~0..1."""
    raw = js.get_axis(cal["index"])
    denom = cal["active"] - cal["rest"]
    if abs(denom) < 1e-6:
        return 0.0
    return (raw - cal["rest"]) / denom


def deadzone(v):
    return 0.0 if abs(v) < DEADZONE else v


def run_raw(js):
    naxes, nbtn = js.get_numaxes(), js.get_numbuttons()
    print(f"axes={naxes} buttons={nbtn}")
    print("wiggle each control; note which index moves. ctrl-C to quit.\n")
    try:
        while True:
            pygame.event.pump()
            axvals = " ".join(f"a{i}={js.get_axis(i):+.2f}" for i in range(naxes))
            btn_down = [i for i in range(nbtn) if js.get_button(i)]
            print(f"\r{axvals}  buttons_down={btn_down}        ", end="")
            time.sleep(1.0 / SEND_HZ)
    except KeyboardInterrupt:
        print("\nstopped")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=HOST_DEFAULT)
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--test", action="store_true", help="print interpreted values, don't send UDP")
    ap.add_argument("--raw", action="store_true", help="dump every axis + pressed button by index")
    args = ap.parse_args()

    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        raise SystemExit("no controller found — check USB/Bluetooth connection")
    js = pygame.joystick.Joystick(0)
    js.init()
    print(f"connected: {js.get_name()}  axes={js.get_numaxes()} buttons={js.get_numbuttons()}")

    if args.raw:
        run_raw(js)
        return

    m = load_mapping()
    ax = m["axes"]
    btn_deadman = m["deadman_button"]
    btn_gripper = m["gripper_button"]

    sock = None
    if not args.test:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        print(f"sending to {args.host}:{args.port}")

    gripper_closed = False
    last_grip_btn = 0

    print("hold the dead-man button (RB) to move, A to toggle gripper, ctrl-C to quit")
    try:
        while True:
            pygame.event.pump()

            vx = deadzone(norm_axis(js, ax["lx"]))
            vy = deadzone(norm_axis(js, ax["ly"]))
            rx = deadzone(norm_axis(js, ax["rx"]))
            ry = deadzone(norm_axis(js, ax["ry"]))
            up = norm_axis(js, ax["trig_up"])
            down = norm_axis(js, ax["trig_down"])
            vz = up - down                     # RT up, LT down
            vrx, vry, vrz = ry, rx, 0.0        # wrist tilt from right stick

            grip_btn = js.get_button(btn_gripper)
            deadman = js.get_button(btn_deadman)

            if grip_btn and not last_grip_btn:
                gripper_closed = not gripper_closed
            last_grip_btn = grip_btn

            if not deadman:
                vx = vy = vz = vrx = vry = vrz = 0.0

            gripper = 1.0 if gripper_closed else 0.0

            if args.test:
                print(f"\rmove=({vx:+.2f},{vy:+.2f},{vz:+.2f}) rot=({vrx:+.2f},{vry:+.2f}) "
                      f"deadman={int(deadman)} grip={gripper:.0f}      ", end="")
            else:
                pkt = struct.pack("<fffffff", vx, vy, vz, vrx, vry, vrz, gripper)
                sock.sendto(pkt, (args.host, args.port))

            time.sleep(1.0 / SEND_HZ)
    except KeyboardInterrupt:
        print("\nstopped")
        if sock:
            sock.sendto(struct.pack("<fffffff", 0, 0, 0, 0, 0, 0, -1.0), (args.host, args.port))


if __name__ == "__main__":
    main()
