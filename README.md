# Xbox teleop

Just the commands, in order. (Environment/venv/ROS setup is assumed already done.)

Two scripts command the arm — the **bridge** and **reset_arm** — so never run
them at the same time. Reset while the bridge is stopped.

Also rememeber to NEVER delete raw data

### Ensure BOTH config.py files in the **/controller_scripts** and **/data_processing** folder are updated and aligned
.env files not working on Thor setup for now



1. Activate venv in every terminal
```bash
source /home/zheyu/code/openpi_xarm/.venv/bin/activate
```

---

## PART A — collect one episode

**1. Reset the arm to the home pose** (bridge NOT running yet)
```bash
python3 ./controller_scripts/reset_arm.py go
```
> First time only, set the home pose: drive/position the arm where you want
> every episode to start, then run `python3 reset_arm.py save` once. After
> that, `go` returns to it.

**2. Start the bridge** — Terminal 1 (leave running)
```bash
python3 ./controller_scripts/bridge_sdk.py
```
Wait for `robot ready (SDK velocity mode)`.

**3. Start the controller** — Terminal 2 (leave running)
```bash
python3 ./controller_scripts/xbox_control.py
```
Hold RB to move, A toggles the gripper.

**4. Start the collector** — Terminal 3
```bash
python3 ./controller_scripts/collector_aarosh.py
```
Wait for `ready. saving to: ...`.

**5. Do the demo**, then **stop the collector** (Ctrl-C in Terminal 3) a beat
after the object settles in the bowl. That saves the episode.

---

## PART B — collect the next episode

The collector auto-numbers (000, 001, 002…) and never overwrites, so:

1. Ctrl-C the collector (if still running).
2. Reset to home — either drive back with the controller, OR for an exact
   reset: Ctrl-C the bridge, run `python3 reset_arm.py go`, restart the bridge.
3. Move the cube to a new spot (keep the bowl fixed).
4. `python3 collector_aarosh.py` again → records the next episode.

Repeat until you have enough demos.

---

## PART C — turn the raw data into a HuggingFace dataset

Raw data lives in `~/aarosh/datasets/xbox_pickplace` (episode_NNN/step_NNNNN/…).
Run these on the machine that has the data-process scripts.

**1. (optional) Remove zero-action steps** — edit the two paths at the top of
`remove_zero_actions.py` *JUST MAKE SURE `config.py` IN PREPROCESSING IS UPDATED:
```python
SRC_DIR = config.py
DST_DIR = config.py
```
```bash
python3 ./data_processing/remove_zero_actions.py
```

**2. Fix delta-EE (CRITICAL — never skip)** — edit the one path at the top of
`fix_delta_ee_pos.py` to point at the dataset from step 1 (or the raw dataset
if you skipped step 1). It edits the files IN PLACE:
```python
DATA_DIR = config.py
```
```bash
python3 ./data_processing/fix_delta_ee_pos.py
```

**3. Convert to LeRobot (save to a local folder — no auto-upload)** — edit
`REPO_NAME` near the top of `setting1_raw2lerobot.py` to your dataset name:
```python
REPO_NAME = "CASE-Lab/xarm850-pickplace-multiobj-v1"
```
Run it WITHOUT `--push-to-hub`. The `HF_LEROBOT_HOME=...` in front just tells
it which folder to write the finished dataset into (pick any empty folder):
```bash
HF_LEROBOT_HOME=/home/zheyu/aarosh/lerobot_out \
python3 ./data_processing/setting1_raw2lerobot.py \
    --data-dir /home/zheyu/aarosh/datasets/NAME_remove_zero
```
This resizes images to 224×224 and encodes to LeRobot format. When it finishes
it prints `Dataset saved to: ...` — that folder is your finished dataset
(here: `/home/zheyu/aarosh/lerobot_out/CASE-Lab/xarm850-pickplace-multiobj-v1`).

**4. Upload manually.** On the HuggingFace website, create the dataset repo
under the CASE-Lab org (private), then drag-and-drop the contents of that
saved folder into it. That's the dataset training will use.

> If you skip the `HF_LEROBOT_HOME=...` prefix, it still saves locally — just
> to the default `~/.cache/huggingface/lerobot/<REPO_NAME>`. Either way, read
> the `Dataset saved to:` line it prints to find the folder.

---

## One-line summary of the loop

```
reset_arm.py go  →  bridge_sdk.py  →  xbox_control.py  →  collector_aarosh.py
   →  (do demo, Ctrl-C collector)  →  repeat
   →  remove_zero_actions.py  →  fix_delta_ee_pos.py  →  setting1_raw2lerobot.py (saves local folder)
   →  drag-and-drop that folder to HuggingFace
```