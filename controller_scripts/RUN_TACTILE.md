# RUN — Tactile-input policy (tacin + shuffled control), command by command

Runs the **tactile-INPUT** models on the real xArm. These are different from vision in
two ways: (1) served from the **dev** code (`tactile_input_model`), and (2) the client
feeds the per-object TacGen latent as input (the "oracle" step). Two machines:
- **GPU server** `ecehpavw1202c.umd.edu` (account `wy891`) — runs the policy.
- **Thor** (owns the arm) — reads cameras + drives the arm + feeds the latent.

---

## 0. THOR — one-time setup (only needed once)

Copy the extra client + the per-object latents onto Thor (next to your other scripts):
```bash
scp wy891@ecehpavw1202c.umd.edu:/export/wy891/home/_tacgen_probe_tmp/object_latents.npz .
```
Make sure `eval_client_tacin.py` is on Thor in the same folder as `object_latents.npz`.

---

## A. GPU SERVER — start the policy (from the DEV env)

1. SSH in:
```bash
ssh wy891@ecehpavw1202c.umd.edu
```
2. tmux session:
```bash
tmux -f /dev/null new -s tacin_serve
```
3. bash + load the DEV env (this is what makes it use the tactile-input code/configs):
```bash
bash
source /export/wy891/aarosh/tactile_input_model/setup_env_newmodel.sh
```
It should print `openpi resolves to: .../tactile_input_model/openpi_src/openpi` — that
confirms you're on the dev copy.
4. Pick a free GPU:
```bash
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
```
5. Start the server. **For the real tactile model:**
```bash
CUDA_VISIBLE_DEVICES=1 $PY scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_xarm_tacin \
  --policy.dir=/export/wy891/aarosh/tactile_input_model/checkpoints/pi05_xarm_tacin/tacin_v1/19999
```
**For the shuffled control instead**, use:
```bash
CUDA_VISIBLE_DEVICES=1 $PY scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_xarm_tacin_shuffled \
  --policy.dir=/export/wy891/aarosh/tactile_input_model/checkpoints/pi05_xarm_tacin_shuffled/tacin_shuffled_v1/19999
```
**For the NULL (matched no-tactile) baseline**, use:
```bash
CUDA_VISIBLE_DEVICES=1 $PY scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_xarm_tacin_null \
  --policy.dir=/export/wy891/aarosh/tactile_input_model/checkpoints/pi05_xarm_tacin_null/tacin_null_v1/19999
```
> The null model feeds itself an all-zeros token, so it IGNORES any latent you send.
> Run it with the plain **`eval_client.py`** (like vision — NO `--object`/latent needed),
> labeled `--policy null`. It's still served from the DEV env, though.
6. Wait for `serving on 0.0.0.0:8000`. Detach: **Ctrl-b, then d**.

---

## B. THOR — open the tunnel

Same as always (server firewall blocks 8000; tunnel through SSH).

1. New Thor terminal, leave open:
```bash
ssh -N -L 8000:localhost:8000 -o PubkeyAuthentication=no -o PreferredAuthentications=keyboard-interactive,password wy891@ecehpavw1202c.umd.edu
```
Enter the `wy891` password. Blank/hanging = tunnel up.

2. Another Thor terminal — confirm:
```bash
nc -vz localhost 8000
```
Should say **succeeded**.

---

## C. THOR — run trials (uses eval_client_tacin.py + --object)

1. Reset the arm:
```bash
python3 reset_arm.py
```
2. (Recommended first) dry run — feeds the latent, prints actions, arm does NOT move:
```bash
python3 eval_client_tacin.py --host localhost --prompt "put the cube into the bowl" --object sponge_stack --dry-run
```
3. Real trial — **hand on the E-STOP**:
```bash
python3 eval_client_tacin.py --host localhost --prompt "put the cube into the bowl" --object sponge_stack --policy tacin
```
- `--object` MUST match the object physically on the table AND be one of:
  cork_block, foam_block, lego_block, metal_block, sponge_stack, stuffed_cube, styrofoam_block, wooden_block.
  (This is the oracle step — it tells the model which object's tactile latent to use.)
- If serving the shuffled control, set `--policy shuffled` so the CSV is labeled right.
- When the object is in the bowl → **Ctrl-C**, then **`s`** for success (else fail).
- Each trial appends a row to `eval_results.csv`.

4. Next trial: reset (C1) → run (C3), changing `--object` to the new object.

---

## D. Switching between real tactile and shuffled control
Only the SERVER changes: re-attach the server tmux (`tmux attach -t tacin_serve`), Ctrl-C,
and relaunch step A5 with the other `--policy.config` + `--policy.dir`. Thor side is
unchanged (just update `--policy` label). Keep objects/positions identical to the vision
run so all three are comparable.

## E. Notes / troubleshooting
- Prompt is ALWAYS `"put the cube into the bowl"`.
- Must serve from the DEV env (step A3). If you serve tacin from the frozen env it fails
  (no `pi05_xarm_tacin` config there).
- If the client errors that `object_latents.npz` is missing → redo step 0.
- If `nc` times out → tunnel died (redo B1) or server not running (redo A).
- Final checkpoint step is **19999** (not 20000) — that's normal.

---

## F. Testing a NEW object the model never trained on

The 8 trained objects have latents baked into `object_latents.npz`. For a brand-new
object you make its latent the SAME way (close-up photo -> DINOv2 -> proj_v -> 128-d),
then feed that latent directly. This is a genuine GENERALIZATION test — the model never
saw this object's latent in training, so treat the result as "does tactile help on an
unseen object," not as an in-distribution number.

Everything up to the trial is done **on your laptop** (the photos are already there).

**0. One-time laptop setup** — just install the packages:
```bash
pip install torch torchvision pillow numpy
```
`best.pt` and `compute_latent.py` are already in this folder (`xarm_scripts/xbox/`),
so once the pip install is done you're ready.

**1. Take 1–3 close-up photos** of the new object (fill the frame, like the training
close-ups — NOT a far scene shot). Put them next to the script, e.g. `duck1.jpg duck2.jpg`.

**2. Compute the latent locally** (run from this folder, where `best.pt` lives):
```bash
python3 ./latents/compute_latent.py ./latents/best.pt ./latents/rubber_duck.npy ./latents/duck1.jpg ./latents/duck2.jpg ./latents/duck3.jpg
```
It prints the saved shape and (if >1 photo) an intra-cosine — want > 0.5 (your photos
agree). First run downloads DINOv2 automatically. Output: `rubber_duck.npy` (128-d).

**3. Move `rubber_duck.npy` to Thor** (USB drive, next to `eval_client_tacin.py`).

**4. Serve the tacin model** exactly as in section A (nothing new there).

**5. Run the trial feeding the new latent** with `--latent-npy` (overrides the npz lookup;
`--object` is now just a CSV label):
```bash
python3 eval_client_tacin.py --host localhost --prompt "put the cube into the bowl" \
  --object rubber_duck --latent-npy rubber_duck.npy --policy tacin
```
Dry-run first (`--dry-run`) to confirm the actions look sane before moving the arm.

> Fair comparison: to claim tactile helped on the new object, also run **vision**
> (RUN_VISION.md) on the SAME new object and positions. Vision needs no latent.
