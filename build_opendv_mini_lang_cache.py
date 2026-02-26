import argparse
import json
import os
import multiprocessing as mp

from tqdm import tqdm

from src.data import _iter_json_array


def _youtuber_formatize(youtuber):
    return youtuber.replace(" ", "_")


def load_mini_folder_set(meta_json, split):
    meta = json.load(open(meta_json, "r"))
    folders = set()
    for item in meta:
        if str(item.get("subset", "")).lower() != "mini":
            continue
        if str(item.get("split", "")).lower() != split:
            continue
        youtuber = item.get("youtuber")
        video_id = item.get("videoid")
        if not youtuber or not video_id:
            continue
        root_name = "full_images" if split == "train" else "val_images"
        folders.add(f"{root_name}/{_youtuber_formatize(youtuber)}/{video_id}")
    return folders


def _collect_entries_worker(args):
    ann_path, folder_set_local = args
    local_entries = []
    for entry in _iter_json_array(ann_path):
        folder = entry.get("folder")
        if not folder or folder not in folder_set_local:
            continue
        local_entries.append({
            "folder": folder,
            "first_frame": entry.get("first_frame"),
            "last_frame": entry.get("last_frame"),
            "cmd": entry.get("cmd"),
            "blip": entry.get("blip"),
        })
    return local_entries


def collect_lang_entries(lang_root, split, folder_set, max_entries=None, num_workers=1):
    if split == "val":
        ann_files = [os.path.join(lang_root, "10hz_YouTube_val.json")]
    else:
        ann_files = [
            os.path.join(lang_root, f"10hz_YouTube_train_split{i}.json")
            for i in range(10)
        ]
    if num_workers <= 1:
        entries = []
        for ann_path in ann_files:
            if not os.path.isfile(ann_path):
                continue
            for entry in _iter_json_array(ann_path):
                folder = entry.get("folder")
                if not folder or folder not in folder_set:
                    continue
                entries.append({
                    "folder": folder,
                    "first_frame": entry.get("first_frame"),
                    "last_frame": entry.get("last_frame"),
                    "cmd": entry.get("cmd"),
                    "blip": entry.get("blip"),
                })
                if max_entries and len(entries) >= max_entries:
                    return entries
        return entries

    ann_paths = [p for p in ann_files if os.path.isfile(p)]
    if not ann_paths:
        return []
    entries = []
    with mp.Pool(processes=num_workers) as pool:
        for local_entries in tqdm(
            pool.imap_unordered(_collect_entries_worker, [(p, folder_set) for p in ann_paths]),
            total=len(ann_paths),
        ):
            entries.extend(local_entries)
            if max_entries and len(entries) >= max_entries:
                return entries[:max_entries]
    return entries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta_json", required=True, help="Path to OpenDV-YouTube meta JSON (from CSV).")
    parser.add_argument("--lang_root", required=True, help="Path to OpenDV-YouTube-Language.")
    parser.add_argument("--split", choices=["train", "val"], required=True)
    parser.add_argument("--output", required=True, help="Output cache JSON.")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--max_entries", type=int, default=None)
    args = parser.parse_args()

    folder_set = load_mini_folder_set(args.meta_json, args.split)
    entries = collect_lang_entries(args.lang_root, args.split, folder_set, max_entries=args.max_entries, num_workers=args.num_workers)
    with open(args.output, "w") as f:
        json.dump(entries, f, indent=2, ensure_ascii=True)
    print(f"Saved {len(entries)} entries to {args.output}")


if __name__ == "__main__":
    main()
