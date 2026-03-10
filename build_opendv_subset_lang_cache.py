import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional


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


def default_input_json(lang_root: str, split: str) -> Path:
    return Path(lang_root) / f"mini_{split}_cache.json"


def load_entries(path: Path) -> List[Dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Source cache JSON not found: {path}")
    payload = json.loads(path.read_text())
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list in {path}")
    entries: List[Dict] = []
    for raw_entry in payload:
        if not isinstance(raw_entry, dict):
            continue
        entry = simplify_entry(raw_entry)
        if entry is not None:
            entries.append(entry)
    if not entries:
        raise ValueError(f"No valid entries found in {path}")
    return entries


def sample_entries(entries: List[Dict], target: Optional[int], seed: int) -> List[Dict]:
    if target is None or target >= len(entries):
        chosen = list(entries)
    else:
        rng = random.Random(seed)
        indices = list(range(len(entries)))
        rng.shuffle(indices)
        chosen = [entries[idx] for idx in indices[:target]]
    chosen.sort(key=lambda item: (item["folder"], item["first_frame"], item["last_frame"]))
    return chosen


def build_subset_cache(
    lang_root: str,
    split: str,
    output: str,
    seed: int,
    target_entries: Optional[int],
    input_json: Optional[str] = None,
) -> int:
    source_path = Path(input_json) if input_json else default_input_json(lang_root, split)
    output_path = Path(output)

    entries = load_entries(source_path)
    selected_entries = sample_entries(entries, target_entries, seed)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        json.dump(selected_entries, handle, indent=2, ensure_ascii=True)

    print(
        f"[subset-cache] source={source_path} loaded={len(entries)} selected={len(selected_entries)} "
        f"output={output_path} seed={seed}"
    )
    return len(selected_entries)


def main():
    parser = argparse.ArgumentParser(
        description="Build a deterministic subset cache directly from mini_train_cache.json or mini_val_cache.json."
    )
    parser.add_argument("--lang_root", required=True, help="Path to OpenDV-YouTube-Language.")
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--input_json", type=str, default=None, help="Optional source cache JSON. Defaults to mini_<split>_cache.json under --lang_root.")
    parser.add_argument("--output", required=True, help="Output cache JSON path.")
    parser.add_argument("--seed", type=int, default=123, help="Deterministic sampling seed.")
    parser.add_argument("--target_entries", type=int, default=200, help="Target number of entries to keep.")
    args = parser.parse_args()

    build_subset_cache(
        lang_root=args.lang_root,
        split=args.split,
        output=args.output,
        seed=args.seed,
        target_entries=args.target_entries,
        input_json=args.input_json,
    )


if __name__ == "__main__":
    main()
aaaasssssdfssssssss