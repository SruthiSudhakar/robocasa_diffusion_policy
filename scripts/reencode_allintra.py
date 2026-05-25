#!/usr/bin/env python
"""Re-encode a lerobot dataset soup's videos to all-intra (GOP=1).

Why: the training DataLoader decodes a random single frame per camera per
sample. The default opencv backend's CAP_PROP_POS_FRAMES seek costs ~50 ms/frame
and is the training bottleneck. The source clips have ~1-2 keyframes total
(GOP ~= whole clip), so any seek-based decoder pays a huge forward-decode.

Re-encoding so every frame is a keyframe (`-g 1`) lets the pyav reader in
lerobot_dataset.py seek+decode a single frame (~6 ms/frame, ~9x faster). Frame
count and fps are preserved, so the parquet timestamp->frame mapping is
unchanged; only h264 requantization differs (negligible, ~0.9% pixel).

Layout: for each ".../<task>/<date>/lerobot" dir we create a sibling
".../lerobot_allintra" that symlinks the small unchanged dirs (meta/, data/,
extras/, *.json ...) and re-encodes only videos/*.mp4. Originals are untouched.

Point training at the result by setting in the task's dataset config:
    lerobot_dir_suffix: _allintra

Usage:
    python scripts/reencode_allintra.py --soup target_atomic_seen --only-first   # validate one
    python scripts/reencode_allintra.py --soup target_atomic_seen                 # all
"""
import argparse
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

SUFFIX = "_allintra"


def resolve_soup_dirs(soup_name):
    from robocasa.utils.dataset_registry import DATASET_SOUP_REGISTRY
    from robocasa.macros import DATASET_BASE_PATH
    dirs = []
    for entry in DATASET_SOUP_REGISTRY[soup_name]:
        p = entry["path"]
        if not os.path.isabs(p) and DATASET_BASE_PATH:
            p = os.path.join(DATASET_BASE_PATH, p)
        dirs.append(p.rstrip("/"))
    return dirs


def mirror_tree(src_root, dst_root):
    """Create dst_root mirroring src_root: symlink everything except videos/,
    which is recreated as real dirs so re-encoded files land inside it."""
    os.makedirs(dst_root, exist_ok=True)
    for name in os.listdir(src_root):
        src = os.path.join(src_root, name)
        dst = os.path.join(dst_root, name)
        if name == "videos":
            continue
        if not os.path.lexists(dst):
            os.symlink(os.path.abspath(src), dst)
    # recreate the videos/ subtree as real directories
    src_videos = os.path.join(src_root, "videos")
    pairs = []
    for dirpath, _, files in os.walk(src_videos):
        rel = os.path.relpath(dirpath, src_root)
        os.makedirs(os.path.join(dst_root, rel), exist_ok=True)
        for f in files:
            if f.endswith(".mp4"):
                pairs.append((os.path.join(dirpath, f),
                              os.path.join(dst_root, rel, f)))
    return pairs


def reencode(args):
    src, dst, crf = args
    if os.path.exists(dst):
        return ("skip", dst)
    tmp = dst + ".tmp.mp4"
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error", "-i", src,
        "-c:v", "libx264", "-g", "1", "-keyint_min", "1",
        "-crf", str(crf), "-pix_fmt", "yuv420p",
        "-fps_mode", "passthrough", "-an", tmp,
    ]
    try:
        subprocess.run(cmd, check=True)
        os.replace(tmp, dst)  # atomic
        return ("ok", dst)
    except Exception as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        return ("err", f"{dst}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--soup", required=True)
    ap.add_argument("--only-first", action="store_true",
                    help="process only the first dataset dir (for validation)")
    ap.add_argument("--workers", type=int, default=min(64, os.cpu_count() or 8))
    ap.add_argument("--crf", type=int, default=18)
    args = ap.parse_args()

    src_dirs = resolve_soup_dirs(args.soup)
    if args.only_first:
        src_dirs = src_dirs[:1]

    all_pairs = []
    for src_root in src_dirs:
        dst_root = src_root + SUFFIX
        pairs = mirror_tree(src_root, dst_root)
        all_pairs.extend((s, d, args.crf) for s, d in pairs)
        print(f"{src_root} -> {dst_root}  ({len(pairs)} videos)")

    print(f"\nRe-encoding {len(all_pairs)} videos with {args.workers} workers...")
    ok = skip = err = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(reencode, p) for p in all_pairs]
        for i, fut in enumerate(as_completed(futs), 1):
            status, info = fut.result()
            if status == "ok":
                ok += 1
            elif status == "skip":
                skip += 1
            else:
                err += 1
                print("ERROR:", info, file=sys.stderr)
            if i % 500 == 0 or i == len(futs):
                print(f"  {i}/{len(futs)}  ok={ok} skip={skip} err={err}")
    print(f"\nDone. ok={ok} skip={skip} err={err}")
    if err:
        sys.exit(1)


if __name__ == "__main__":
    main()
