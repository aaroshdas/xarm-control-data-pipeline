"""calibrate.py — guided controller mapping for xbox_control.py.

Different controllers / drivers / USB-vs-Bluetooth number their buttons and
axes differently, which is why hardcoded indices break. This walks you through
each control once, auto-detects which button or axis you moved (and which
direction), and saves it to `xbox_mapping.json` next to this script.
xbox_control.py loads that file automatically — no code editing needed.

Run it ON THOR with the controller connected:

    python3 calibrate.py

Follow the prompts. Takes about a minute. Re-run any time the mapping feels
off or you switch to a different controller / connection.
"""
import json
import os
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import pygame

MAPPING_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "xbox_mapping.json")
SETTLE_PUMPS = 15   # event pumps after Enter, to flush the held-state into pygame


def read_state(js):
    """Return (axes, buttons) after flushing the event queue so held inputs register."""
    for _ in range(SETTLE_PUMPS):
        pygame.event.pump()
        time.sleep(0.01)
    axes = [js.get_axis(i) for i in range(js.get_numaxes())]
    buttons = [js.get_button(i) for i in range(js.get_numbuttons())]
    return axes, buttons


def capture_button(js, prompt):
    input(f"  {prompt}, keep holding it, then press Enter... ")
    _, buttons = read_state(js)
    down = [i for i, b in enumerate(buttons) if b]
    if len(down) == 0:
        print("  ⚠️  no button detected — try again, make sure you're holding it as you press Enter.")
        return capture_button(js, prompt)
    if len(down) > 1:
        print(f"  ⚠️  multiple buttons down {down} — release everything else and try again.")
        return capture_button(js, prompt)
    print(f"  ✅ got button {down[0]}")
    return down[0]


def capture_axis(js, baseline, prompt):
    """Detect which axis moved most from baseline; record its rest + active values."""
    input(f"  {prompt}, hold it there, then press Enter... ")
    axes, _ = read_state(js)
    deltas = [abs(axes[i] - baseline[i]) for i in range(len(axes))]
    idx = max(range(len(deltas)), key=lambda i: deltas[i])
    if deltas[idx] < 0.3:
        print(f"  ⚠️  no clear axis movement (biggest change only {deltas[idx]:.2f}) — try again, push it fully.")
        return capture_axis(js, baseline, prompt)
    print(f"  ✅ got axis {idx}  (rest={baseline[idx]:+.2f}  active={axes[idx]:+.2f})")
    return {"index": idx, "rest": round(baseline[idx], 3), "active": round(axes[idx], 3)}


def main():
    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        raise SystemExit("no controller found — check USB/Bluetooth connection")
    js = pygame.joystick.Joystick(0)
    js.init()
    print(f"connected: {js.get_name()}  axes={js.get_numaxes()} buttons={js.get_numbuttons()}\n")

    input("First, DON'T touch any sticks/triggers. Press Enter to record the resting state... ")
    baseline, _ = read_state(js)
    print(f"  baseline axes: {[f'{v:+.2f}' for v in baseline]}\n")

    print("Now each control, one at a time:\n")
    mapping = {
        "deadman_button": capture_button(js, "Hold the RIGHT BUMPER (RB) — this is the dead-man 'allow motion' button"),
        "gripper_button": capture_button(js, "Press the A button — toggles the gripper"),
        "axes": {},
    }
    mapping["axes"]["lx"] = capture_axis(js, baseline, "Push the LEFT stick fully RIGHT")
    mapping["axes"]["ly"] = capture_axis(js, baseline, "Push the LEFT stick fully UP (forward)")
    mapping["axes"]["rx"] = capture_axis(js, baseline, "Push the RIGHT stick fully RIGHT")
    mapping["axes"]["ry"] = capture_axis(js, baseline, "Push the RIGHT stick fully UP")
    mapping["axes"]["trig_up"] = capture_axis(js, baseline, "Squeeze the RIGHT TRIGGER (RT) fully — arm goes UP")
    mapping["axes"]["trig_down"] = capture_axis(js, baseline, "Squeeze the LEFT TRIGGER (LT) fully — arm goes DOWN")

    with open(MAPPING_PATH, "w") as f:
        json.dump(mapping, f, indent=2)
    print(f"\n✅ saved mapping to {MAPPING_PATH}")
    print("xbox_control.py will now use it automatically. Run:  python3 xbox_control.py --test")


if __name__ == "__main__":
    main()
