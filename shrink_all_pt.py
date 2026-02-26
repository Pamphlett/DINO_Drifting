#!/usr/bin/env python3
import argparse
import os
import torch
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


def shrink_payload(obj):
    changed = False

    if torch.is_tensor(obj):
        return obj.clone().contiguous(), True

    if isinstance(obj, dict):
        new = dict(obj)
        for k, v in new.items():
            if torch.is_tensor(v):
                new[k] = v.clone().contiguous()
                changed = True
        return new, changed

    return obj, False


def process_file(path):
    try:
        old_size = os.path.getsize(path)
        obj = torch.load(path, map_location="cpu")
    except Exception as exc:
        return False, f"[skip] load failed: {path} ({exc})"

    new_obj, changed = shrink_payload(obj)
    if not changed:
        return False, None

    tmp_path = path + ".tmp"
    torch.save(new_obj, tmp_path)
    os.replace(tmp_path, path)
    new_size = os.path.getsize(path)
    return True, f"[ok] {path}  {old_size/1024/1024:.1f}MB -> {new_size/1024/1024:.1f}MB"


def iter_pt_files(root):
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name.endswith(".pt"):
                yield os.path.join(dirpath, name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="root dir to scan recursively")
    ap.add_argument("--workers", type=int, default=4, help="number of worker threads")
    ap.add_argument("--log", action="store_true", help="print per-file results")
    args = ap.parse_args()

    paths = iter_pt_files(args.root)
    changed = 0
    scanned = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for ok, msg in tqdm(
            executor.map(process_file, paths),
            desc="Processing .pt",
            unit="file",
        ):
            scanned += 1
            if ok:
                changed += 1
                if args.log and msg:
                    print(msg)
            elif args.log and msg:
                print(msg)

    print(f"Done. scanned={scanned}, changed={changed}")


if __name__ == "__main__":
    main()
