"""bridge_sdk.py — SDK-only bridge. No ROS2, no driver launch needed.

Same job as bridge_aarosh.py (receive UDP velocity packets from
xbox_control.py and drive the arm), but commands the arm DIRECTLY through
the xArm Python SDK instead of a ROS2 velocity service. That service needs
the separate xarm_api driver node running; this doesn't. One SDK connection
owns everything — the same connection that already moves the gripper.

Nothing else in the pipeline changes: collector_aarosh.py reads arm state
through its own SDK connection, so the recorded data format is identical no
matter how the arm was moved.

SAFETY (three independent layers so the arm can't run away):
  1. Dead-man: xbox_control sends zero velocity whenever RB is released.
  2. Firmware watchdog: each velocity command is valid for VEL_DURATION s;
     the arm auto-stops if no fresh command arrives (needs firmware >=1.8.0).
  3. Software watchdog: if no UDP packet arrives for STALE_STOP s (e.g.
     xbox_control crashes), the bridge itself commands a stop.
Also: conservative default speed, and KEEP A HAND NEAR THE E-STOP the first
few runs. Raise MOVE_GAIN once the motion feels right.

Run (plain terminal, just needs the xArm SDK in the venv — NO ROS2 source):
    python3 bridge_sdk.py
"""
import socket
import struct
import time

from xarm.wrapper import XArmAPI

ROBOT_IP = "192.168.1.230"
HOST = "0.0.0.0"
PORT = 5005

MOVE_GAIN = 50.0      # mm/s at full stick — start low; the lab bridge used 500
ROTATION_GAIN = 0.3    # rad/s at full stick
VEL_DURATION = 0.3     # s — firmware watchdog: arm stops if no fresh cmd in this window
STALE_STOP = 0.3       # s — software watchdog: force-stop if no packet for this long
SEND_HZ = 60           # steady rate we stream velocity to the arm (smoothness)


def main():
    arm = XArmAPI(ROBOT_IP)
    arm.clean_error()
    arm.clean_warn()
    arm.motion_enable(enable=True)
    arm.set_mode(5)      # cartesian velocity control
    arm.set_state(0)
    arm.set_gripper_enable(True)
    arm.set_gripper_mode(0)
    arm.set_gripper_speed(3000)
    time.sleep(0.5)

    _, (err, warn) = arm.get_err_warn_code()
    print(f"arm: mode={arm.mode} state={arm.state} error={err} warn={warn}")
    if err:
        print(f"⚠️  arm ERROR {err} — clear it in UFactory Studio, CLOSE Studio, then rerun.")
        arm.disconnect()
        return
    print("robot ready (SDK velocity mode).")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((HOST, PORT))
    sock.setblocking(False)
    print(f"listening on {HOST}:{PORT}")

    gripper_is_closed = False
    last_trigger_val = 0.0
    last_rx = time.time()
    last_speeds = [0.0] * 6           # most recent commanded velocity
    send_interval = 1.0 / SEND_HZ     # stream to the arm at this steady rate
    next_send = time.time()

    try:
        while True:
            # Drain the socket to the NEWEST packet (throw away any backlog so
            # we never lag behind stale commands — a big source of stutter).
            newest = None
            while True:
                try:
                    newest, _ = sock.recvfrom(1024)
                except BlockingIOError:
                    break

            if newest is not None:
                length = len(newest)
                vx, vy, vz, vrx, vry, vrz, gripper = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0
                ok = True
                if length == 24:
                    vx, vy, vz, vrx, vry, vrz = struct.unpack('<ffffff', newest)
                elif length == 28:
                    vx, vy, vz, vrx, vry, vrz, gripper = struct.unpack('<fffffff', newest)
                else:
                    ok = False
                if ok:
                    last_speeds = [vx * MOVE_GAIN, vy * MOVE_GAIN, vz * MOVE_GAIN,
                                   vrx * ROTATION_GAIN, vry * ROTATION_GAIN, vrz * ROTATION_GAIN]
                    last_rx = time.time()
                    if gripper != -1.0:
                        if gripper > 0.5 and last_trigger_val <= 0.5:
                            gripper_is_closed = not gripper_is_closed
                            arm.set_gripper_position(0 if gripper_is_closed else 850, wait=False)
                        last_trigger_val = gripper

            # Stream the latest command to the arm at a STEADY rate, decoupled
            # from packet arrival. If the stream went stale, stream zeros
            # (software watchdog) so the arm can't run away.
            now = time.time()
            if now >= next_send:
                next_send = now + send_interval
                stale = (now - last_rx) > STALE_STOP
                speeds = [0.0] * 6 if stale else last_speeds
                arm.vc_set_cartesian_velocity(speeds, is_radian=True, duration=VEL_DURATION)

                moving = "STALE" if stale else ("YES" if any(abs(s) > 1e-6 for s in speeds) else "no ")
                print(f"XYZ:[{speeds[0]:>5.0f},{speeds[1]:>5.0f},{speeds[2]:>5.0f}] "
                      f"mode:{arm.mode} state:{arm.state} moving:{moving}   ", end='\r')

            time.sleep(0.002)

    except KeyboardInterrupt:
        print("\nstopping...")
        arm.vc_set_cartesian_velocity([0, 0, 0, 0, 0, 0], is_radian=True)
        time.sleep(0.2)
        arm.disconnect()


if __name__ == '__main__':
    main()