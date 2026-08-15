# RUN — Vision policy (command by command)

Runs the **vision** baseline on the real xArm. Two machines:
- **GPU server** `ecehpavw1202c.umd.edu` (account `wy891`) — runs the policy ("brain").
- **Thor** (in the lab, owns the arm) — reads cameras + drives the arm ("body").

They talk over port 8000 via an SSH tunnel. Do the steps in order.

---

## A. GPU SERVER — start the policy

1. SSH in from your laptop (must be on UMD network / eduroam):
```bash
ssh wy891@ecehpavw1202c.umd.edu
```
2. Open a tmux session (the `-f /dev/null` avoids the AFS config warning):
```bash
tmux -f /dev/null new -s vision_serve
```
3. Switch to bash and load the FROZEN env (vision uses the frozen repo):
```bash
bash
source /export/wy891/aarosh/openpi/setup_env.sh
```
4. Pick a free GPU:
```bash
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
```
5. Start the server (replace `<gpu>` with a free index):
```bash
CUDA_VISIBLE_DEVICES=<gpu> uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_xarm_vision \
  --policy.dir=/export/wy891/aarosh/saved_checkpoints/vision/vision_20k
```
6. Wait until it prints that it is serving on `0.0.0.0:8000`. Detach: **Ctrl-b, then d**.

---

## B. THOR — open the tunnel

The server firewall blocks port 8000, so tunnel it through SSH (port 22).

1. New terminal on Thor — leave it open the whole time:
```bash
ssh -N -L 8000:localhost:8000 -o PubkeyAuthentication=no -o PreferredAuthentications=keyboard-interactive,password wy891@ecehpavw1202c.umd.edu
```
Enter the `wy891` password. The terminal goes blank and hangs = tunnel is up.

2. In ANOTHER Thor terminal, confirm it works:
```bash
nc -vz localhost 8000
```
Should say **succeeded**.

---

## C. THOR — run trials

1. Reset the arm to its start pose:
```bash
python3 reset_arm.py
```
2. (Optional but recommended first time) dry run — reads cameras + prints actions, arm does NOT move:
```bash
python3 eval_client.py --host localhost --prompt "put the cube into the bowl" --dry-run
```
3. Real trial — **hand on the E-STOP**:
```bash
python3 eval_client.py --host localhost --prompt "put the cube into the bowl" --policy vision --object sponge_stack
```
- It waits for ENTER before moving.
- When the object is in the bowl → **Ctrl-C**, then type **`s`** for success (anything else = fail).
- If it's failing, let it hit `--max-steps` (default 300) or Ctrl-C at your fixed cutoff, then mark fail.
- Each trial appends a row to `eval_results.csv`.

4. Next trial: reset (C1) → run (C3). Change `--object` to match the object on the table.

---

## D. Notes
- Prompt is ALWAYS `"put the cube into the bowl"` (the only task string the model was trained on).
- `--object` is just a label for the CSV here (vision doesn't use it as input). Use one of:
  cork_block, foam_block, lego_block, metal_block, sponge_stack, stuffed_cube, styrofoam_block, wooden_block.
- Keep object positions/lighting identical across policies so the comparison is fair.
- To stop the server later: re-attach (`tmux attach -t vision_serve`) and Ctrl-C.
- If `nc` times out: the tunnel died (reopen B1) or the server isn't running (redo A).
