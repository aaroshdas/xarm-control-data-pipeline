# Policy Evaluation Runbook

Run a trained policy on the real xArm and record success/fail. Two machines:

- **GPU server** (`ecehpavw1202c.umd.edu`) = the "brain": runs `serve_policy.py`, holds ONE checkpoint, answers with actions.
- **Thor** (in the lab, owns the arm) = the "body": runs `eval_client.py`, reads cameras + arm, executes actions.

They talk over the UMD network on **port 8000**. To test a different policy you only restart the server — the Thor command never changes.

---

## 0. One-time per session

**On the GPU server**, open a tmux session (skip the AFS config that throws "Permission denied"):
```bash
tmux -f /dev/null new -s policy_testing
bash
source /export/wy891/aarosh/openpi/setup_env.sh
```
(The `source` redirects caches off AFS onto /export — do it every session or the server can die when the AFS token expires.)

---

## 1. Start the policy server (GPU server)

Pick ONE policy. The four checkpoints live under `/export/wy891/aarosh/saved_checkpoints/`
(adjust the exact folder names to what you saved):

| Policy            | `--policy.config`         | `--policy.dir` (example)                                     |
|-------------------|---------------------------|-------------------------------------------------------------|
| Vision baseline   | `pi05_xarm_vision`        | `/export/wy891/aarosh/saved_checkpoints/vision/vision_20k`         |
| Tactile-trained   | `pi05_xarm_tactile`       | `/export/wy891/aarosh/saved_checkpoints/tactile/tactile_20k`        |
| Control: shuffled | `pi05_xarm_ctrl_shuffled` | `/export/wy891/aarosh/saved_checkpoints/shuffled/shuffled_20k`  |
| Control: random   | `pi05_xarm_ctrl_random`   | `/export/wy891/aarosh/saved_checkpoints/random/random_20k`    |

```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_xarm_vision \
  --policy.dir=/export/wy891/aarosh/saved_checkpoints/vision/vision_20k
```
Wait until it prints that it's serving on `0.0.0.0:8000`. Detach with **Ctrl-b, d**.

> If it errors about **missing normalization stats**, STOP — do not run the arm.
> The checkpoint's `assets/` must contain the patched `norm_stats.json` (the one
> with dims 3/4/5 floored). Ping Claude with the exact error before continuing.

### 1b. Open the SSH tunnel (Thor) — REQUIRED

The server firewall only allows SSH (port 22); port 8000 is blocked, so Thor
cannot connect to `ecehpavw1202c:8000` directly (it times out). Tunnel 8000
through SSH instead. In a **separate terminal on Thor, left open** the whole session:
```bash
ssh -N -L 8000:localhost:8000 wy891@ecehpavw1202c.umd.edu
```
(`-N` = just forward, no shell. Enter the wy891 password.)

**Sanity check from Thor** that the tunnel + server are up:
```bash
nc -vz localhost 8000    # should say "succeeded" (NOT time out)
```
> A blank/hanging `nc` that times out = the tunnel isn't up (or the server isn't
> running). Direct `nc -vz ecehpavw1202c.umd.edu 8000` will ALWAYS time out —
> that's the blocked firewall port; use the tunnel + `localhost`.

---

## 2. Reset the arm (Thor)

Put the arm in its standard start pose before every trial so runs are comparable:
```bash
python3 reset_arm.py
```

---

## 3. Run a trial (Thor)

```bash
python3 eval_client.py \
  --host localhost \
  --prompt "put the cube into the bowl" \
  --policy vision \
  --object sponge
```
- `--host localhost` — you connect through the SSH tunnel (step 1b), not the server directly.
- `--policy` — MUST match the checkpoint currently being served (`vision` / `tactile` /
  `ctrl_shuffled` / `ctrl_random`). This labels the row in the results file.
- `--object` — which object you placed (e.g. `sponge`, `rock`). Optional but recommended.

**Diagnostics:** add `--dry-run` to print the policy's action numbers WITHOUT moving the
arm (no e-stop needed). Add `--max-steps 40` to cap a run's length while testing.

**Results are saved.** Each finished trial appends a row to `eval_results.csv` (in the
folder you run from): `timestamp, policy, object, prompt, result, steps`. Success rates
per policy come straight from this file — no hand-tallying.
- **Keep a hand on the E-STOP.** It waits for ENTER before moving.
- It runs until the task is done, you Ctrl-C, or it hits `--max-steps` (default 300 = 30 s).
- At the end it asks: type **`s`** for success, anything else for fail.

**Prompt:** always use exactly `"put the cube into the bowl"`. That is the ONLY
task string in the training dataset (verified in `meta/tasks.jsonl`), so it's what
the model expects for every object and every trial. There is no per-object prompt —
the policy identifies the object from the cameras, not the words. Use the same
prompt across all four policies so the comparison stays clean.

---

## 4. Repeat & swap policies

- **Same policy, next trial:** reset (step 2) → run (step 3). Do N trials per object.
- **Different policy:** go to the server tmux, **Ctrl-C** the server, relaunch step 1 with the new config/dir. Thor command is unchanged.

Keep trials matched across the 4 policies: same objects, same start positions, same
number of trials each, so success rates are comparable. Record every result.

---

## Tuning knobs (top of `eval_client.py`)
| Knob           | Meaning                                              | Start | Try if… |
|----------------|------------------------------------------------------|-------|---------|
| `SPEED`        | mm/s cap on arm moves                                | 80    | raise if too slow once safe |
| `EXEC_HORIZON` | steps of the 10-step chunk run before re-asking      | 8     | lower (e.g. 4) if it over/undershoots — more reactive |
| `MAX_DXYZ_MM`  | max position move per step (safety clamp)            | 30    | leave unless motions are truncated |
| `MAX_DROT_DEG` | max rotation per step (safety clamp)                 | 8     | leave unless rotations are truncated |
| `WS_ENABLE`    | refuse targets outside `WS_MIN/WS_MAX` box           | False | set box to your table, then True |

## Evaluating the tactile-INPUT (tacin) models — DIFFERENT from vision

The `tacin` policies were trained to receive the per-object TacGen latent as an INPUT,
so they need (a) a different client that feeds the latent, and (b) the server run from
the DEV code (only it understands the tactile-input token).

**One-time on Thor:** copy the extra client + the latents file:
```bash
scp wy891@ecehpavw1202c.umd.edu:/export/wy891/home/_tacgen_probe_tmp/object_latents.npz .
# and copy eval_client_tacin.py next to eval_client.py
```

**Serve the tacin policy — from the DEV env (not the frozen one):**
```bash
tmux -f /dev/null new -s tacin_serve ; bash
source /export/wy891/aarosh/tactile_input_model/setup_env_newmodel.sh
ls checkpoints/pi05_xarm_tacin/tacin_v1/            # find the step folder (e.g. 20000)
CUDA_VISIBLE_DEVICES=<free_gpu> $PY scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_xarm_tacin \
  --policy.dir=/export/wy891/aarosh/tactile_input_model/checkpoints/pi05_xarm_tacin/tacin_v1/20000
```
(For the control, use `--policy.config=pi05_xarm_tacin_shuffled` + its checkpoint dir.)
Same SSH tunnel (step 1b) and `--host localhost` as always.

**Run a trial with the tacin client** (feeds the latent for the object present):
```bash
python3 eval_client_tacin.py --host localhost --prompt "put the cube into the bowl" \
  --object sponge_stack --policy tacin
```
`--object` MUST be one of: cork_block, foam_block, lego_block, metal_block,
sponge_stack, stuffed_cube, styrofoam_block, wooden_block — and match the object on
the table (this is the "oracle" step). Everything else (reset, success/fail, CSV) is
the same as the vision client.

> Which client for which policy: **vision** and the old aux-tactile/control policies →
> `eval_client.py`. **tacin / tacin_shuffled / tacin_random** → `eval_client_tacin.py`.

## Safety checklist (every run)
1. Hand on E-STOP.
2. Arm reset to start pose.
3. Workspace clear of people/laptops.
4. First run of any new policy: watch the very first motion, be ready to Ctrl-C.

---

## Troubleshooting

**`PyTrees have different structure ... {'tacgen_head'}` when loading the VISION checkpoint.**
Already fixed (2026-08-12). The vision policy was trained *before* the `tacgen_head`
layer existed, so it has one fewer layer than the tactile runs. The model now builds
`tacgen_head` only when the config flag `use_tacgen_head=True` — which is set only on
the 3 tactile/control configs, not vision. If this error comes back it means the server
code got reverted; re-apply `patch_tacgen_head.py` (in `/export/wy891/aarosh/`).

**Server loads but the arm does nothing / jerks.**
Lower `EXEC_HORIZON` (more reactive) and/or `SPEED` at the top of `eval_client.py`.
Confirm the base/wrist cameras aren't swapped — base = cam_1 (`215222078407`),
wrist = cam_0 (`845112070404`).

**Eval client "timed out in connect" / `nc` to the server:8000 times out.**
The server firewall blocks port 8000 (only SSH/22 is open) — this is expected, not a
bug. You MUST use the SSH tunnel (step 1b) and connect with `--host localhost`.
If `nc -vz localhost 8000` still times out, the tunnel terminal died (reopen it) or the
policy server isn't running (check step 1).
