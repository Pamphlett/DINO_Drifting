import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional


def iter_json_array(path: str, chunk_size: int = 1024 * 1024) -> Iterable[Dict]:
    decoder = json.JSONDecoder()
    with open(path, "r") as handle:
        buffer = ""
        pos = 0
        while True:
            if pos >= len(buffer):
                chunk = handle.read(chunk_size)
                if not chunk:
                    return
                buffer += chunk
            while pos < len(buffer) and buffer[pos].isspace():
                pos += 1
            if pos < len(buffer):
                if buffer[pos] != "[":
                    raise ValueError(f"Expected '[' at start of JSON array in {path}")
                pos += 1
                break

        while True:
            while True:
                if pos >= len(buffer):
                    chunk = handle.read(chunk_size)
                    if not chunk:
                        return
                    buffer = buffer[pos:] + chunk
                    pos = 0
                if buffer[pos].isspace() or buffer[pos] == ",":
                    pos += 1
                elif buffer[pos] == "]":
                    return
                else:
                    break

            while True:
                try:
                    obj, next_pos = decoder.raw_decode(buffer, pos)
                    pos = next_pos
                    yield obj
                    break
                except json.JSONDecodeError:
                    chunk = handle.read(chunk_size)
                    if not chunk:
                        raise
                    buffer += chunk


def simplify_entry(entry: Dict) -> Optional[Dict]:
    folder = entry.get("folder")
    first_frame = entry.get("first_frame")
    last_frame = entry.get("last_frame")
    if not folder or not first_frame or not last_frame:
        return None
    return {
        "folder": folder,
        "first_frame": first_frame,
        "last_frame": last_frame,
        "cmd": entry.get("cmd"),
        "blip": entry.get("blip"),
    }


def load_entries(path: str) -> List[Dict]:
    entries: List[Dict] = []
    for raw_entry in iter_json_array(path):
        entry = simplify_entry(raw_entry)
        if entry is not None:
            entries.append(entry)
    return entries


def sample_entries(entries: List[Dict], target: Optional[int], seed: int) -> List[Dict]:
    if target is None or target >= len(entries):
        return list(entries)
    rng = random.Random(seed)
    indices = list(range(len(entries)))
    rng.shuffle(indices)
    chosen = [entries[idx] for idx in indices[:target]]
    chosen.sort(key=lambda item: (item["folder"], item["first_frame"], item["last_frame"]))
    return chosen


def collect_annotation_files(lang_root: str, split: str, train_num_splits: int) -> List[str]:
    root = Path(lang_root)
    if split == "val":
        candidates = [root / "10hz_YouTube_val.json"]
    else:
        candidates = [root / f"10hz_YouTube_train_split{i}.json" for i in range(train_num_splits)]
    ann_files = [str(path) for path in candidates if path.is_file()]
    if not ann_files:
        raise FileNotFoundError(f"No annotation JSON files found for split={split} under {root}")
    return ann_files


def distribute_targets(total_target: Optional[int], file_count: int) -> List[Optional[int]]:
    if total_target is None:
        return [None] * file_count
    base = total_target // file_count
    remainder = total_target % file_count
    targets = []
    for idx in range(file_count):
        targets.append(base + (1 if idx < remainder else 0))
    return targets


def build_subset_cache(
    lang_root: str,
    split: str,
    output: str,
    seed: int,
    target_entries: Optional[int],
    train_num_splits: int,
) -> int:
    ann_files = collect_annotation_files(lang_root, split, train_num_splits)
    per_file_targets = distribute_targets(target_entries, len(ann_files))

    selected_entries: List[Dict] = []
    total_loaded = 0
    for file_idx, ann_path in enumerate(ann_files):
        entries = load_entries(ann_path)
        total_loaded += len(entries)
        file_target = per_file_targets[file_idx]
        sampled = sample_entries(entries, file_target, seed + file_idx)
        selected_entries.extend(sampled)
        print(
            f"[subset-cache] {os.path.basename(ann_path)}: loaded={len(entries)} "
            f"selected={len(sampled)}"
        )

    selected_entries.sort(key=lambda item: (item["folder"], item["first_frame"], item["last_frame"]))
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        json.dump(selected_entries, handle, indent=2, ensure_ascii=True)

    print(
        f"[subset-cache] saved {len(selected_entries)} entries to {output_path} "
        f"(split={split}, total_loaded={total_loaded}, seed={seed})"
    )
    return len(selected_entries)


def main():
    parser = argparse.ArgumentParser(
        description="Build a fixed OpenDV language-cache subset directly from train/val annotation JSON files."
    )
    parser.add_argument("--lang_root", required=True, help="Path to OpenDV-YouTube-Language.")
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--output", required=True, help="Output cache JSON path.")
    parser.add_argument("--seed", type=int, default=123, help="Deterministic sampling seed.")
    parser.add_argument(
        "--target_entries",
        type=int,
        default=200,
        help="Target number of entries to keep. For train this is distributed approximately evenly across splits.",
    )
    parser.add_argument("--train_num_splits", type=int, default=10, help="Number of train split JSON shards.")
    args = parser.parse_args()

    build_subset_cache(
        lang_root=args.lang_root,
        split=args.split,
        output=args.output,
        seed=args.seed,
        target_entries=args.target_entries,
        train_num_splits=args.train_num_splits,
    )


if __name__ == "__main__":
    main()
