"""merge_raw.py — combine several per-object raw folders into ONE, renumbering.

Each object folder has its own episode_000, episode_001, ... which collide.
This copies them all into a single destination folder with continuous
numbering, so the rest of the pipeline (remove_zero -> fix_delta_ee ->
raw2lerobot) can run once over the combined set.

Example:
    python3 merge_raw.py \
        --out ~/aarosh/datasets/all_objects \
        ~/aarosh/datasets/styrofoam_block \
        ~/aarosh/datasets/wooden_block

    # styrofoam episodes -> episode_000..017, wooden -> episode_018..035

Order matters: sources are appended in the order you list them. By default it
COPIES (originals untouched — delete them yourself once you've verified). Pass
--move to move instead (frees space immediately, empties the sources).

A manifest.txt is written into --out recording which source each episode came
from, so you can always tell which episodes are which object.
"""
import argparse
import shutil
from pathlib import Path


def episodes_in(folder: Path):
    return sorted([d for d in folder.iterdir() if d.is_dir() and d.name.startswith("episode_")])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="destination combined folder")
    ap.add_argument("sources", nargs="+", help="per-object raw folders, in the order to append them")
    ap.add_argument("--move", action="store_true", help="move instead of copy (frees space, empties sources)")
    args = ap.parse_args()

    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    # continue numbering after anything already in --out
    existing = episodes_in(out)
    next_idx = (max(int(d.name.split("_")[1]) for d in existing) + 1) if existing else 0

    manifest = []
    for src in args.sources:
        src = Path(src).expanduser()
        eps = episodes_in(src)
        if not eps:
            print(f"WARNING: no episode_* folders in {src}, skipping")
            continue
        print(f"{src.name}: {len(eps)} episodes -> episode_{next_idx:03d}..{next_idx + len(eps) - 1:03d}")
        for ep in eps:
            dst = out / f"episode_{next_idx:03d}"
            if dst.exists():
                raise SystemExit(f"{dst} already exists — refusing to overwrite. Use a fresh --out.")
            if args.move:
                shutil.move(str(ep), str(dst))
            else:
                shutil.copytree(ep, dst)
            manifest.append(f"episode_{next_idx:03d}\t{src.name}\t(was {ep.name})")
            next_idx += 1

    (out / "manifest.txt").write_text("\n".join(manifest) + "\n")
    print(f"\nDone. {next_idx} total episodes in {out}")
    print(f"manifest: {out / 'manifest.txt'}")
    if not args.move:
        print("Originals left in place — delete them once you've verified the merge.")


if __name__ == "__main__":
    main()
