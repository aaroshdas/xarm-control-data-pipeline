"""bridge_aarosh.py — personal copy of vr/bridge_new.py.

Identical logic to the shared bridge (same UDP packet formats, same ROS2
velocity-control calls) — this is just your own copy so you can tune
MOVE_GAIN/ROTATION_GAIN for controller feel without touching the shared
lab file. xbox_control.py talks to this the same way it would talk to the
original.
"""
import rclpy
from rclpy.node import Node
from xarm_msgs.srv import MoveVelocity, SetInt16, SetInt16ById
import socket
import struct
import time
import select

ROBOT_IP = "192.168.1.230"
HOST = "0.0.0.0"
PORT = 5005

MOVE_GAIN = 500.0      # mm/s
ROTATION_GAIN = 0.3    # rad/s

try:
    from xarm.wrapper import XArmAPI
except ImportError:
    XArmAPI = None


def main():
    arm_sdk = None
    if XArmAPI:
        try:
            arm_sdk = XArmAPI(ROBOT_IP)
            arm_sdk.clean_error()
            arm_sdk.set_gripper_enable(True)
            arm_sdk.set_gripper_mode(0)
            arm_sdk.set_gripper_speed(3000)
            arm_sdk.set_gripper_force(1000)
            print("gripper SDK connected")
        except Exception:
            print("gripper SDK connect skipped")

    rclpy.init()
    node = rclpy.create_node('bridge_aarosh')

    velo_cli = node.create_client(MoveVelocity, '/ufactory/vc_set_cartesian_velocity')
    clean_err_cli = node.create_client(SetInt16, '/ufactory/clean_error')
    motion_en_cli = node.create_client(SetInt16ById, '/ufactory/motion_enable')
    set_mode_cli = node.create_client(SetInt16, '/ufactory/set_mode')
    set_state_cli = node.create_client(SetInt16, '/ufactory/set_state')

    def call_srv(cli, req):
        if cli.wait_for_service(timeout_sec=1.0):
            return cli.call_async(req)
        return None

    print("initializing real robot...")
    call_srv(clean_err_cli, SetInt16.Request(data=0))
    time.sleep(0.5)
    call_srv(motion_en_cli, SetInt16ById.Request(id=8, data=1))
    time.sleep(0.5)

    call_srv(set_mode_cli, SetInt16.Request(data=5))  # mode 5 = velocity control
    call_srv(set_state_cli, SetInt16.Request(data=0))
    time.sleep(0.5)

    # The ROS2 calls above are fire-and-forget (call_async, never confirmed),
    # so the arm can end up NOT actually enabled — which looks like "bridge
    # sends velocity but arm ignores it". Re-do enable synchronously via the
    # SDK, which blocks until each step is applied. Mode/state/enable are
    # global on the controller, so this reinforces the ROS2 velocity path.
    if arm_sdk:
        arm_sdk.clean_error()
        arm_sdk.clean_warn()
        code_en = arm_sdk.motion_enable(enable=True)
        code_md = arm_sdk.set_mode(5)     # 5 = cartesian velocity control
        code_st = arm_sdk.set_state(0)    # 0 = ready/running
        time.sleep(0.5)
        _, (err, warn) = arm_sdk.get_err_warn_code()
        _, state = arm_sdk.get_state()
        print(f"  SDK enable: motion_enable={code_en} set_mode={code_md} set_state={code_st}")
        print(f"  arm state={state} (0=ready, expected)   error={err} warn={warn}")
        if err:
            print(f"  ⚠️  arm has ERROR code {err} — clear it in UFactory Studio, CLOSE Studio, then rerun the bridge.")

    print("robot ready: mode 5 (teleop)\n")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((HOST, PORT))
    sock.setblocking(False)
    print(f"listening on {HOST}:{PORT}")

    gripper_is_closed = False
    last_trigger_val = 0.0

    # Live mode/state monitor. Reading these from the SDK every packet (30Hz)
    # is too chatty, so refresh ~1x/sec and show the last value. If `mode`
    # isn't 5 during teleop, the arm silently ignores velocity commands —
    # that's the thing we're hunting.
    loop_i = 0
    cur_mode, cur_state = -1, -1

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.001)
            try:
                data, addr = sock.recvfrom(1024)
                length = len(data)

                vx, vy, vz, vrx, vry, vrz, gripper = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0

                if length == 24:
                    vx, vy, vz, vrx, vry, vrz = struct.unpack('<ffffff', data)
                elif length == 28:
                    vx, vy, vz, vrx, vry, vrz, gripper = struct.unpack('<fffffff', data)
                else:
                    continue

                svx, svy, svz = vx * MOVE_GAIN, vy * MOVE_GAIN, vz * MOVE_GAIN
                svrx, svry, svrz = vrx * ROTATION_GAIN, vry * ROTATION_GAIN, vrz * ROTATION_GAIN

                loop_i += 1
                if arm_sdk and loop_i % 30 == 0:
                    cur_mode = arm_sdk.mode
                    _, cur_state = arm_sdk.get_state()

                moving = "YES" if any(abs(v) > 1e-6 for v in (svx, svy, svz, svrx, svry, svrz)) else "no "
                print(f"XYZ:[{svx:>4.0f},{svy:>4.0f},{svz:>4.0f}] Grip:{gripper:+.0f} "
                      f"mode:{cur_mode} state:{cur_state} sending_motion:{moving}", end='\r')

                req = MoveVelocity.Request()
                req.speeds = [float(svx), float(svy), float(svz), float(svrx), float(svry), float(svrz)]
                req.is_tool_coord = False
                velo_cli.call_async(req)

                if arm_sdk and gripper != -1.0:
                    if gripper > 0.5 and last_trigger_val <= 0.5:
                        gripper_is_closed = not gripper_is_closed
                        target_pos = 0 if gripper_is_closed else 850
                        arm_sdk.set_gripper_position(target_pos, wait=False)
                    last_trigger_val = gripper

            except BlockingIOError:
                pass
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nstopping...")
        if arm_sdk:
            arm_sdk.disconnect()
        rclpy.shutdown()


if __name__ == '__main__':
    main()