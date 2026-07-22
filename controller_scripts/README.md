# Xbox teleop pipeline — setup + run


## ** For now genuinely just ignore all the enviornment stuff, claude made this doc and doesn't understand the Thor setup
Use openpi_xarm venv (pygame is already installed)

Three separate scripts, three separate terminals, running at the same time:

| Script | Job | Needs ROS2? |
|---|---|---|
| `bridge_aarosh.py` | turns controller numbers into arm motion | yes |
| `collector_aarosh.py` | watches the arm + 2 cameras, saves your dataset | no (uses the xArm SDK + RealSense directly) |
| `xbox_control.py` | reads the Xbox controller, sends numbers to the bridge | no |

They don't import each other — they talk over network sockets (UDP) and by
both watching the same physical robot. That's why folder location doesn't
matter; only the Python environment each one runs in matters.

## `rclpy` isn't a pip package

Instead: find out what environment already successfully runs
`bridge_new.py` today (ask whoever set it up, or just check `which python3`
and `conda env list` on Thor), and run `bridge_aarosh.py` there too, after
sourcing ROS2 the same way it's normally sourced (usually something like
`source /opt/ros/<distro>/setup.bash` plus a workspace `install/setup.bash`
— check the lab doc or ask whoever's env this is for the exact line).

`collector_aarosh.py` and `xbox_control.py` have no ROS2 dependency at all —
they're safe to set up fresh.

## Packages needed (on top of the existing ROS2 setup above)

```bash
# xbox_control.py
pip install pygame
```

Notes:
- **** Honestly just have cluade do this to not mess anything up. "can you give me the commands to
  install pygame on this virtual enviornment and make sure that its safe and won't mess anything else
  on the enviornment up


## Running it (3 terminals on Thor)

**Terminal 1 — calibarate before doing anything else**:
```bash
source /home/zheyu/code/openpi_xarm/.venv/bin/activate 
python3 ./controller_scripts/calibrate.py 
```
**Terminal 1 — bridge** (SDK VERSION):
```bash
source /home/zheyu/code/openpi_xarm/.venv/bin/activate 
python3 ./controller_scripts/bridge_sdk.py
```
**ALTERNATIVE Terminal 1 — bridge** (needs ROS2 sourced, in every terminal activate the venv first):
```bash
source /home/zheyu/code/openpi_xarm/.venv/bin/activate 
source ~/ros2_ws/install/setup.bash
python3 ./controller_scripts/bridge_aarosh.py
```
Wait for `robot ready: mode 5 (teleop)` before moving on.

**Terminal 2 — collector** (conda env `xbox_teleop`, or wherever pyrealsense2 lives):
```bash
python3 ./controller_scripts/collector_aarosh.py
```
Wait for `ready. saving to: ...` — this is also where you confirm your
dataset is landing in *your* folder (`~/aarosh/datasets/xbox_pickplace` by
default), not the shared one.

**Terminal 3 — controller**:
```bash
conda activate xbox_teleop
python3 ./controller_scripts/xbox_control.py --test     # confirm the stick/button mapping looks right first
python3 ./controller_scripts/xbox_control.py            # then go live
```

Ctrl-C any of the three to stop; the collector saves whatever's left in its
queue before exiting.
