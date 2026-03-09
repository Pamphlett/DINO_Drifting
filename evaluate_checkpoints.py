import argparse
import csv
import json
import math
import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import ToPILImage

from save_predicted_dino_features import add_missing_args, denormalize_images
from src.data import CS_VideoData, OpenDV_VideoData, opendv_collate
from src.dino_f import Dino_f


OPENDV_CMD_CAPTION_MAP = {
    0: "Go straight.",
    1: "Pass the intersection.",
    2: "Turn left.",
    3: "Turn right.",
    4: "Change to the left lane.",
    5: "Change to the right lane.",
    6: "Go to the left lane branch.",
    7: "Go to the right lane branch.",
    8: "Pass the crosswalk.",
    9: "Pass the railroad.",
    10: "Merge.",
    11: "Make a U-turn.",
    12: "Stop.",
    13: "Deviate.",
}


def parse_tuple(value):
    if value is None or isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return tuple(map(int, str(value).split(",")))


def parse_list(value):
    if value is None or isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return list(map(int, str(value).split(",")))


def parse_bool(value):
    if value is None or isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def log(message: str) -> None:
    print(message, flush=True)


def load_config(config_path: Optional[str]) -> Dict[str, Any]:
    if not config_path:
        return {}
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    suffix = path.suffix.lower()
    with path.open("r") as handle:
        if suffix in {".yaml", ".yml"}:
            try:
                import yaml
            except ImportError as exc:
                raise ImportError("PyYAML is required to read YAML config files.") from exc
            data = yaml.safe_load(handle)
        else:
            data = json.load(handle)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in config file: {path}")
    return data


def build_parser(config_defaults: Optional[Dict[str, Any]] = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline comparison of multiple DINO-Foresight checkpoints on a fixed evaluation subset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Example:\n"
            "  python evaluate_checkpoints.py \\\n"
            "    --ckpt_dir /path/to/checkpoints \\\n"
            "    --data_root /path/to/data \\\n"
            "    --output_dir /path/to/output \\\n"
            "    --eval_subset_index_file /path/to/output/eval_subset_indices.json \\\n"
            "    --rgb_decoder_path /path/to/rgb_decoder.ckpt\n"
        ),
    )
    parser.set_defaults(**(config_defaults or {}))
    parser.add_argument("--config_path", type=str, default=None, help="Optional JSON/YAML config for this evaluation script.")
    parser.add_argument("--ckpt_dir", type=str, default=None, help="Directory containing checkpoints to discover automatically.")
    parser.add_argument("--ckpt_paths", type=str, nargs="+", default=None, help="Explicit checkpoint paths to compare.")
    parser.add_argument("--max_ckpts_to_compare", type=int, default=5, help="Maximum number of checkpoints to compare after discovery.")
    parser.add_argument("--dataset", type=str, default="opendv", choices=["opendv", "cityscapes"])
    parser.add_argument("--data_root", type=str, default=None, help="Dataset root placeholder. For OpenDV this maps to --opendv_root unless overridden.")
    parser.add_argument("--data_split", type=str, default="val", choices=["train", "val", "test"], help="Dataset split used for offline comparison.")
    parser.add_argument("--opendv_root", type=str, default=None)
    parser.add_argument("--opendv_train_root", type=str, default=None)
    parser.add_argument("--opendv_val_root", type=str, default=None)
    parser.add_argument("--opendv_meta_path", type=str, default=None)
    parser.add_argument("--opendv_lang_root", type=str, default=None)
    parser.add_argument("--opendv_lang_cache_path", type=str, default=None)
    parser.add_argument("--opendv_lang_cache_train", type=str, default=None)
    parser.add_argument("--opendv_lang_cache_val", type=str, default=None)
    parser.add_argument("--opendv_use_lang_annos", type=parse_bool, default=None)
    parser.add_argument("--opendv_use_lang_features", type=parse_bool, default=None)
    parser.add_argument("--opendv_lang_feat_name", type=str, default=None)
    parser.add_argument("--opendv_lang_feat_key", type=str, default=None)
    parser.add_argument("--opendv_lang_mask_key", type=str, default=None)
    parser.add_argument("--opendv_feat_ext", type=str, default=None)
    parser.add_argument("--opendv_filter_folder", type=str, default=None)
    parser.add_argument("--opendv_max_clips", type=int, default=None)
    parser.add_argument("--opendv_video_dir", type=str, default=None)
    parser.add_argument("--eval_subset_size", type=int, default=100)
    parser.add_argument("--eval_subset_seed", type=int, default=123)
    parser.add_argument("--eval_subset_index_file", type=str, default=None, help="JSON file used to save/reuse deterministic evaluation indices.")
    parser.add_argument(
        "--eval_cmd_substrings",
        type=str,
        nargs="+",
        default=None,
        help="Optional OpenDV text filters. Only samples whose cmd/blip matches these substrings remain eligible.",
    )
    parser.add_argument(
        "--eval_text_field",
        type=str,
        default="cmd",
        choices=["cmd", "blip", "either"],
        help="Which text field to use when filtering eval samples by substring.",
    )
    parser.add_argument(
        "--eval_cmd_groups",
        type=str,
        nargs="+",
        default=None,
        help="Optional balanced cmd groups, e.g. straight=straight,go_straight turn=turn,left,right",
    )
    parser.add_argument("--eval_sampling_seed", type=int, default=0, help="Base seed for stochastic sampling during quantitative evaluation.")
    parser.add_argument("--rgb_decoder_path", type=str, default=None, help="Optional RGB decoder checkpoint. If unavailable, latent-only evaluation still runs.")
    parser.add_argument("--pca_ckpt", type=str, default=None, help="Optional PCA checkpoint override if the predictor checkpoint expects one.")
    parser.add_argument("--head_ckpt", type=str, default=None, help="Optional head checkpoint override for downstream eval modalities.")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--qual_num_samples", type=int, default=6)
    parser.add_argument("--qual_seed", type=int, default=7)
    parser.add_argument("--num_samples_per_input", type=int, default=1)
    parser.add_argument("--device", type=str, default=None, help="Optional device override, e.g. cuda:0 or cpu.")
    parser.add_argument("--img_size", type=parse_tuple, default=None, help="Optional override if needed.")
    parser.add_argument("--sequence_length", type=int, default=None, help="Optional override if needed.")
    parser.add_argument("--step", type=int, default=None, help="Optional inference-step override for non-flow-matching checkpoints.")
    parser.add_argument("--eval_modality", type=str, default=None, choices=[None, "segm", "depth", "surface_normals"])
    parser.add_argument("--save_args_json", type=parse_bool, default=True)
    return parser


def normalize_config_values(config: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(config)
    if "img_size" in normalized:
        normalized["img_size"] = parse_tuple(normalized["img_size"])
    if "d_layers" in normalized:
        normalized["d_layers"] = parse_list(normalized["d_layers"])
    return normalized


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_checkpoint_payload(path: str) -> Dict[str, Any]:
    return torch.load(path, map_location="cpu")


def extract_checkpoint_args(payload: Dict[str, Any]) -> argparse.Namespace:
    hyper = payload.get("hyper_parameters", {})
    if isinstance(hyper, dict) and "args" in hyper:
        args = hyper["args"]
    else:
        args = hyper
    if isinstance(args, argparse.Namespace):
        base = deepcopy(args)
    elif isinstance(args, dict):
        base = argparse.Namespace(**deepcopy(args))
    else:
        raise ValueError("Unable to recover checkpoint arguments from checkpoint hyper_parameters.")
    return add_missing_args(base, base)


def parse_step_from_name(path: str) -> Optional[int]:
    stem = Path(path).stem
    patterns = [
        r"step[-_=](\d+)",
        r"global_step[-_=](\d+)",
        r"epoch[-_=]\d+[-_=]step[-_=](\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, stem)
        if match:
            return int(match.group(1))
    return None


def parse_epoch_from_name(path: str) -> Optional[int]:
    stem = Path(path).stem
    match = re.search(r"epoch[-_=](\d+)", stem)
    return int(match.group(1)) if match else None


def checkpoint_label(path: str, step: Optional[int]) -> str:
    if step is not None:
        return f"step-{step}"
    return Path(path).stem


def discover_checkpoints(args) -> List[Dict[str, Any]]:
    if not args.ckpt_dir and not args.ckpt_paths:
        raise ValueError("Provide either --ckpt_dir or --ckpt_paths.")
    if args.ckpt_dir and args.ckpt_paths:
        raise ValueError("Use either --ckpt_dir or --ckpt_paths, not both.")

    ckpt_paths: List[str]
    if args.ckpt_paths:
        ckpt_paths = [str(Path(path)) for path in args.ckpt_paths]
    else:
        root = Path(args.ckpt_dir)
        if not root.is_dir():
            raise FileNotFoundError(f"Checkpoint directory not found: {root}")
        ckpt_paths = sorted(str(path) for path in root.rglob("*.ckpt"))
        if not ckpt_paths:
            raise ValueError(f"No checkpoints found under {root}")

    metadata = []
    for index, ckpt_path in enumerate(ckpt_paths):
        step = parse_step_from_name(ckpt_path)
        epoch = parse_epoch_from_name(ckpt_path)
        metadata.append(
            {
                "path": ckpt_path,
                "step": step,
                "epoch": epoch,
                "sort_step": step if step is not None else int(1e18) + index,
            }
        )

    step_based = [item for item in metadata if item["step"] is not None]
    chosen_pool = step_based if step_based else metadata
    chosen_pool.sort(key=lambda item: (item["sort_step"], item["path"]))

    if len(chosen_pool) <= args.max_ckpts_to_compare:
        selected = chosen_pool
    else:
        positions = np.linspace(0, len(chosen_pool) - 1, num=args.max_ckpts_to_compare)
        indices = []
        used = set()
        for pos in positions:
            idx = int(round(float(pos)))
            idx = min(max(idx, 0), len(chosen_pool) - 1)
            if idx not in used:
                indices.append(idx)
                used.add(idx)
        cursor = 0
        while len(indices) < args.max_ckpts_to_compare and cursor < len(chosen_pool):
            if cursor not in used:
                indices.append(cursor)
                used.add(cursor)
            cursor += 1
        indices.sort()
        selected = [chosen_pool[idx] for idx in indices]

    log("Selected checkpoints for comparison:")
    for item in selected:
        label = checkpoint_label(item["path"], item["step"])
        log(f"  - {label}: {item['path']}")
    return selected


DATASET_OVERRIDE_KEYS = [
    "dataset",
    "data_root",
    "data_split",
    "opendv_root",
    "opendv_train_root",
    "opendv_val_root",
    "opendv_meta_path",
    "opendv_lang_root",
    "opendv_lang_cache_path",
    "opendv_lang_cache_train",
    "opendv_lang_cache_val",
    "opendv_use_lang_annos",
    "opendv_use_lang_features",
    "opendv_lang_feat_name",
    "opendv_lang_feat_key",
    "opendv_lang_mask_key",
    "opendv_feat_ext",
    "opendv_filter_folder",
    "opendv_max_clips",
    "opendv_video_dir",
    "batch_size",
    "num_workers",
    "img_size",
    "sequence_length",
    "eval_modality",
    "pca_ckpt",
    "head_ckpt",
]


MODEL_RUNTIME_OVERRIDE_KEYS = [
    "step",
    "eval_modality",
    "pca_ckpt",
    "head_ckpt",
]


def apply_cli_overrides(base_args: argparse.Namespace, cli_args) -> argparse.Namespace:
    updated = deepcopy(base_args)
    if not hasattr(updated, "dataset"):
        setattr(updated, "dataset", cli_args.dataset)
    for key in DATASET_OVERRIDE_KEYS + MODEL_RUNTIME_OVERRIDE_KEYS:
        if not hasattr(cli_args, key):
            continue
        value = getattr(cli_args, key)
        if value is not None:
            setattr(updated, key, value)

    if cli_args.data_root:
        if updated.dataset == "opendv":
            updated.opendv_root = cli_args.opendv_root or cli_args.data_root
            updated.data_path = cli_args.data_root
        else:
            updated.data_path = cli_args.data_root

    if not hasattr(updated, "data_path"):
        updated.data_path = cli_args.data_root
    if updated.dataset == "opendv" and getattr(updated, "opendv_root", None) is None:
        updated.opendv_root = cli_args.data_root

    updated.eval_mode = True
    updated.return_rgb_path = True
    updated.random_crop = False
    updated.random_horizontal_flip = False
    updated.random_time_flip = False
    updated.num_workers = cli_args.num_workers
    updated.num_workers_val = cli_args.num_workers
    updated.batch_size = cli_args.batch_size
    updated.eval_midterm = False
    updated.evaluate_baseline = False
    updated.use_val_to_train = False
    updated.use_train_to_val = False
    updated.num_gpus = 1
    updated.precision = getattr(updated, "precision", "32-true")
    return add_missing_args(updated, updated)


def build_dataset(eval_args: argparse.Namespace, split: str):
    dataset_name = getattr(eval_args, "dataset", "opendv")
    if dataset_name == "opendv":
        data_module = OpenDV_VideoData(arguments=eval_args, subset=split, batch_size=eval_args.batch_size)
        dataset = data_module._dataset(split, eval_mode=True)
        collate_fn = opendv_collate
    else:
        data_module = CS_VideoData(arguments=eval_args, subset=split, batch_size=eval_args.batch_size)
        dataset = data_module._dataset(split, eval_mode=True)
        collate_fn = None
    return dataset, collate_fn


def decode_opendv_cmd_value(cmd_value: Any) -> Optional[str]:
    if cmd_value is None:
        return None
    if isinstance(cmd_value, (int, np.integer)):
        return OPENDV_CMD_CAPTION_MAP.get(int(cmd_value))
    text = str(cmd_value).strip()
    if not text:
        return None
    try:
        cmd_id = int(text)
    except ValueError:
        return None
    return OPENDV_CMD_CAPTION_MAP.get(cmd_id)


def candidate_texts_for_dataset_index(dataset, dataset_index: int, text_field: str) -> List[str]:
    if hasattr(dataset, "clips") and getattr(dataset, "use_annotations", False):
        clip = dataset.clips[dataset_index]
        values = []
        if text_field in {"cmd", "either"}:
            cmd_value = clip.get("cmd")
            values.append(cmd_value)
            decoded_cmd = decode_opendv_cmd_value(cmd_value)
            if decoded_cmd:
                values.append(decoded_cmd)
        if text_field in {"blip", "either"}:
            values.append(clip.get("blip"))
        return [str(value) for value in values if value]
    return []


def preview_dataset_texts(
    dataset,
    text_field: str,
    max_examples: int = 12,
) -> List[str]:
    previews: List[str] = []
    limit = min(len(dataset), max_examples * 4)
    for dataset_index in range(limit):
        texts = candidate_texts_for_dataset_index(dataset, dataset_index, text_field)
        if not texts:
            continue
        preview = " || ".join(texts).strip()
        if not preview:
            continue
        previews.append(preview)
        if len(previews) >= max_examples:
            break
    return previews


def parse_cmd_group_specs(group_specs: Optional[Sequence[str]]) -> List[Tuple[str, List[str]]]:
    if not group_specs:
        return []
    groups = []
    for spec in group_specs:
        if "=" not in spec:
            raise ValueError(f"Invalid --eval_cmd_groups spec '{spec}'. Expected name=token1,token2")
        name, raw_tokens = spec.split("=", 1)
        name = name.strip()
        tokens = [token.strip().lower() for token in raw_tokens.split(",") if token.strip()]
        if not name or not tokens:
            raise ValueError(f"Invalid --eval_cmd_groups spec '{spec}'.")
        groups.append((name, tokens))
    return groups


def filter_eval_candidates(dataset, args) -> List[int]:
    all_indices = list(range(len(dataset)))
    if not args.eval_cmd_substrings:
        return all_indices

    filters = [token.lower() for token in args.eval_cmd_substrings if token.strip()]
    if not filters:
        return all_indices

    matched = []
    for dataset_index in all_indices:
        texts = candidate_texts_for_dataset_index(dataset, dataset_index, args.eval_text_field)
        haystack = " || ".join(texts).lower()
        if haystack and any(token in haystack for token in filters):
            matched.append(dataset_index)

    if not matched:
        raise ValueError(
            f"No dataset samples matched eval text filters {args.eval_cmd_substrings} "
            f"using field={args.eval_text_field}."
        )

    log(
        f"Filtered eval candidates by {args.eval_text_field}: "
        f"{len(matched)}/{len(all_indices)} samples matched {args.eval_cmd_substrings}"
    )
    return matched


def build_group_buckets(dataset, candidate_indices: Sequence[int], args) -> Dict[str, List[int]]:
    group_specs = parse_cmd_group_specs(args.eval_cmd_groups)
    if not group_specs:
        return {}

    buckets: Dict[str, List[int]] = {name: [] for name, _ in group_specs}
    for dataset_index in candidate_indices:
        texts = candidate_texts_for_dataset_index(dataset, dataset_index, args.eval_text_field)
        haystack = " || ".join(texts).lower()
        if not haystack:
            continue
        for name, tokens in group_specs:
            if any(token in haystack for token in tokens):
                buckets[name].append(dataset_index)
                break

    empty_groups = [name for name, indices in buckets.items() if not indices]
    if empty_groups:
        preview = preview_dataset_texts(dataset, args.eval_text_field)
        preview_text = "; ".join(preview) if preview else "no non-empty text found in scanned samples"
        raise ValueError(
            f"No dataset samples matched eval cmd groups: {empty_groups}. "
            f"field={args.eval_text_field}. Example texts: {preview_text}"
        )

    for name, indices in buckets.items():
        log(f"Eval cmd group '{name}': {len(indices)} matched candidates")
    return buckets


def sample_balanced_group_indices(
    buckets: Dict[str, List[int]],
    subset_size: int,
    seed: int,
) -> List[int]:
    if subset_size <= 0:
        return []
    rng = np.random.default_rng(seed)
    ordered_names = list(buckets.keys())
    shuffled_buckets: Dict[str, List[int]] = {}
    for name in ordered_names:
        arr = np.array(sorted(set(buckets[name])), dtype=int)
        rng.shuffle(arr)
        shuffled_buckets[name] = arr.tolist()

    quotas = {name: 0 for name in ordered_names}
    remaining = subset_size
    active = ordered_names[:]
    while remaining > 0 and active:
        progressed = False
        share = max(1, remaining // len(active))
        next_active = []
        for name in active:
            capacity = len(shuffled_buckets[name]) - quotas[name]
            if capacity <= 0:
                continue
            take = min(capacity, share, remaining)
            if take > 0:
                quotas[name] += take
                remaining -= take
                progressed = True
            if len(shuffled_buckets[name]) - quotas[name] > 0:
                next_active.append(name)
            if remaining == 0:
                break
        active = next_active
        if not progressed:
            break

    if remaining > 0:
        raise ValueError("Not enough samples across cmd groups to satisfy eval_subset_size.")

    indices = []
    for name in ordered_names:
        indices.extend(shuffled_buckets[name][:quotas[name]])
    indices = sorted(indices)
    return indices


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2)


def load_or_create_eval_indices(dataset, args, output_dir: Path) -> List[int]:
    dataset_len = len(dataset)
    if dataset_len <= 0:
        raise ValueError("Dataset is empty.")
    candidate_indices = filter_eval_candidates(dataset, args)
    index_path = Path(args.eval_subset_index_file) if args.eval_subset_index_file else output_dir / "eval_subset_indices.json"
    if index_path.is_file():
        payload = json.load(index_path.open("r"))
        indices = payload["indices"] if isinstance(payload, dict) else payload
        indices = [int(idx) for idx in indices]
        if not indices:
            raise ValueError(f"Index file is empty: {index_path}")
        if max(indices) >= dataset_len or min(indices) < 0:
            raise ValueError(
                f"Existing subset indices in {index_path} are incompatible with current dataset length {dataset_len}."
            )
        if isinstance(payload, dict):
            stored_filters = payload.get("eval_cmd_substrings")
            stored_field = payload.get("eval_text_field")
            stored_groups = payload.get("eval_cmd_groups")
            if (
                stored_filters != args.eval_cmd_substrings
                or stored_field != args.eval_text_field
                or stored_groups != args.eval_cmd_groups
            ):
                raise ValueError(
                    f"Existing subset index file {index_path} was created with different text filters: "
                    f"{stored_filters} field={stored_field} groups={stored_groups}"
                )
        log(f"Reusing evaluation subset from {index_path} ({len(indices)} samples).")
        return indices

    subset_size = min(int(args.eval_subset_size), len(candidate_indices))
    buckets = build_group_buckets(dataset, candidate_indices, args)
    if buckets:
        indices = sample_balanced_group_indices(buckets, subset_size, args.eval_subset_seed)
    else:
        rng = np.random.default_rng(args.eval_subset_seed)
        indices = sorted(rng.choice(np.array(candidate_indices), size=subset_size, replace=False).tolist())
    payload = {
        "dataset_length": dataset_len,
        "candidate_pool_size": len(candidate_indices),
        "subset_size": subset_size,
        "subset_seed": args.eval_subset_seed,
        "data_split": args.data_split,
        "eval_cmd_substrings": args.eval_cmd_substrings,
        "eval_text_field": args.eval_text_field,
        "eval_cmd_groups": args.eval_cmd_groups,
        "indices": indices,
    }
    save_json(index_path, payload)
    log(f"Saved deterministic evaluation subset to {index_path} ({subset_size} samples).")
    return indices


def select_qualitative_indices(eval_indices: Sequence[int], qual_num_samples: int, qual_seed: int, output_dir: Path) -> List[int]:
    if qual_num_samples <= 0 or not eval_indices:
        return []
    count = min(qual_num_samples, len(eval_indices))
    rng = np.random.default_rng(qual_seed)
    chosen = sorted(rng.choice(np.array(eval_indices), size=count, replace=False).tolist())
    save_json(
        output_dir / "qualitative_indices.json",
        {
            "qual_num_samples": count,
            "qual_seed": qual_seed,
            "indices": chosen,
        },
    )
    log(f"Qualitative subset uses {count} samples: {chosen}")
    return chosen


def parse_batch(batch) -> Dict[str, Any]:
    if torch.is_tensor(batch):
        frames = batch
        gt_img = frames[:, -1]
        return {
            "frames": frames,
            "gt_img": gt_img,
            "future_gt": None,
            "text_tokens": None,
            "text_mask": None,
            "rgb_paths": None,
        }

    if not isinstance(batch, (list, tuple)) or len(batch) == 0:
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    frames = batch[0]
    gt_img = None
    future_gt = None
    text_tokens = None
    text_mask = None
    rgb_paths = None

    for item in batch[1:]:
        if torch.is_tensor(item):
            if item.ndim == 4 and item.shape[1] == 3 and gt_img is None:
                gt_img = item
            elif item.ndim == 5 and item.shape[2] == 3:
                future_gt = item
            elif item.ndim == 3:
                text_tokens = item
            elif item.ndim == 2:
                text_mask = item
        elif isinstance(item, (list, tuple)) and item and isinstance(item[0], (str, Path)):
            rgb_paths = [str(value) for value in item]

    if gt_img is None:
        gt_img = frames[:, -1]

    return {
        "frames": frames,
        "gt_img": gt_img,
        "future_gt": future_gt,
        "text_tokens": text_tokens,
        "text_mask": text_mask,
        "rgb_paths": rgb_paths,
    }


def reshape_features(model: Dino_f, feats: torch.Tensor, img_h: int, img_w: int) -> torch.Tensor:
    return feats.reshape(feats.shape[0], img_h // model.patch_size, img_w // model.patch_size, -1)


def set_torch_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tensor_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    return ToPILImage()(image_tensor.detach().cpu().clamp(0, 1))


def add_label(image: Image.Image, label: str) -> Image.Image:
    image = image.convert("RGB")
    canvas = Image.new("RGB", image.size, color=(0, 0, 0))
    canvas.paste(image, (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, image.size[0], 20), fill=(0, 0, 0))
    draw.text((6, 4), label, fill=(255, 255, 255))
    return canvas


def hstack(images: Sequence[Image.Image], background=(0, 0, 0)) -> Image.Image:
    if not images:
        raise ValueError("Need at least one image to concatenate.")
    widths = [img.size[0] for img in images]
    heights = [img.size[1] for img in images]
    canvas = Image.new("RGB", (sum(widths), max(heights)), color=background)
    x = 0
    for image in images:
        canvas.paste(image, (x, 0))
        x += image.size[0]
    return canvas


def vstack(images: Sequence[Image.Image], background=(0, 0, 0)) -> Image.Image:
    if not images:
        raise ValueError("Need at least one image to concatenate.")
    widths = [img.size[0] for img in images]
    heights = [img.size[1] for img in images]
    canvas = Image.new("RGB", (max(widths), sum(heights)), color=background)
    y = 0
    for image in images:
        canvas.paste(image, (0, y))
        y += image.size[1]
    return canvas


def fit_pca_from_latents(feats: Sequence[torch.Tensor], n_components: int = 3) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    flat = torch.cat([feat.reshape(-1, feat.shape[-1]).float().cpu() for feat in feats], dim=0)
    mean = flat.mean(dim=0, keepdim=True)
    centered = flat - mean
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    components = vh[:n_components]
    proj = centered @ components.T
    minv = proj.min(dim=0).values
    maxv = proj.max(dim=0).values
    return mean.squeeze(0), components, minv, maxv


def latent_to_pca_image(
    feat: torch.Tensor,
    pca_mean: torch.Tensor,
    pca_components: torch.Tensor,
    pca_min: torch.Tensor,
    pca_max: torch.Tensor,
) -> Image.Image:
    h, w, c = feat.shape
    flat = feat.reshape(-1, c).float().cpu()
    proj = (flat - pca_mean) @ pca_components.T
    denom = (pca_max - pca_min).clamp(min=1e-6)
    proj = ((proj - pca_min) / denom).clamp(0, 1)
    array = (proj.reshape(h, w, -1).numpy() * 255).astype(np.uint8)
    return Image.fromarray(array)


class DecoderWrapper:
    def __init__(self, model, mode: str):
        self.model = model
        self.mode = mode

    def decode(self, feats_bhwc: torch.Tensor) -> torch.Tensor:
        model_device = next(self.model.parameters()).device
        feats_bhwc = feats_bhwc.to(model_device)
        with torch.inference_mode():
            if self.mode == "feature_rgb":
                pred = self.model(feats_bhwc)
            elif self.mode == "dinov2_rgb":
                decoder = self.model.decoder
                feature_dim = self.model.emb_dim
                b, h, w, c = feats_bhwc.shape
                num_layers = len(self.model.d_layers)
                if c != feature_dim * num_layers:
                    raise ValueError(
                        f"Decoder feature dim mismatch: got {c}, expected {feature_dim * num_layers}."
                    )
                feat_list = [
                    feats_bhwc[:, :, :, i * feature_dim:(i + 1) * feature_dim].reshape(b, h * w, feature_dim)
                    for i in range(num_layers)
                ]
                pred = decoder(feat_list, h, w)
                pred = F.interpolate(pred, size=self.model.img_size, mode="bicubic", align_corners=False)
                pred = torch.sigmoid(pred)
            else:
                raise ValueError(f"Unknown decoder mode: {self.mode}")
        return pred.detach()


def load_optional_decoder(decoder_path: Optional[str], device: torch.device) -> Optional[DecoderWrapper]:
    if not decoder_path:
        return None

    try:
        from train_rgb_decoder_from_feats import FeatureRgbDecoder

        decoder = FeatureRgbDecoder.load_from_checkpoint(decoder_path, strict=False, map_location="cpu")
        decoder.eval().to(device)
        log(f"Loaded feature RGB decoder: {decoder_path}")
        return DecoderWrapper(decoder, mode="feature_rgb")
    except Exception as exc:
        log(f"FeatureRgbDecoder load failed for {decoder_path}: {exc}")

    try:
        from train_rgb_decoder import DinoV2RGBDecoder

        decoder = DinoV2RGBDecoder.load_from_checkpoint(
            decoder_path,
            strict=False,
            map_location="cpu",
            lpips_weight=0,
        )
        decoder.eval().to(device)
        log(f"Loaded DinoV2 RGB decoder: {decoder_path}")
        return DecoderWrapper(decoder, mode="dinov2_rgb")
    except Exception as exc:
        log(f"RGB decoder fallback failed for {decoder_path}: {exc}")
        log("Continuing with latent-only qualitative outputs.")
        return None


def resolve_device(device_arg: Optional[str]) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_for_checkpoint(ckpt_path: str, cli_args, device: torch.device) -> Tuple[Dino_f, argparse.Namespace, Dict[str, Any]]:
    payload = load_checkpoint_payload(ckpt_path)
    base_args = extract_checkpoint_args(payload)
    eval_args = apply_cli_overrides(base_args, cli_args)
    model = Dino_f.load_from_checkpoint(ckpt_path, args=eval_args, strict=False, map_location="cpu")
    model.eval().to(device)
    model.args = eval_args
    if model.dino_v2 is not None:
        model.dino_v2 = model.dino_v2.to(device)
    if model.eva2clip is not None:
        model.eva2clip = model.eva2clip.to(device)
    if model.sam is not None:
        model.sam = model.sam.to(device)
    meta = {
        "step": payload.get("global_step", parse_step_from_name(ckpt_path)),
        "epoch": payload.get("epoch", parse_epoch_from_name(ckpt_path)),
    }
    return model, eval_args, meta


def metric_mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return float(np.mean(values))


def create_dataloader(dataset, collate_fn, batch_size: int, num_workers: int) -> DataLoader:
    kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": True,
        "drop_last": False,
    }
    if collate_fn is not None:
        kwargs["collate_fn"] = collate_fn
    return DataLoader(dataset, **kwargs)


def evaluate_checkpoint(
    checkpoint_info: Dict[str, Any],
    cli_args,
    device: torch.device,
    eval_indices: Sequence[int],
    qual_indices: Sequence[int],
    decoder: Optional[DecoderWrapper],
    qual_cache: Dict[int, Dict[str, Any]],
) -> Tuple[Dict[str, Any], argparse.Namespace]:
    ckpt_path = checkpoint_info["path"]
    model, eval_args, meta = load_model_for_checkpoint(ckpt_path, cli_args, device)
    split = cli_args.data_split
    full_dataset, collate_fn = build_dataset(eval_args, split)
    eval_dataset = Subset(full_dataset, list(eval_indices))
    loader = create_dataloader(eval_dataset, collate_fn, cli_args.batch_size, cli_args.num_workers)

    use_amp = device.type == "cuda"
    label = checkpoint_label(ckpt_path, meta["step"])
    log(f"Evaluating {label} on {len(eval_dataset)} samples...")

    latent_mse_values: List[float] = []
    latent_cos_values: List[float] = []
    motion_ratio_values: List[float] = []
    rgb_l1_values: List[float] = []
    rgb_mse_values: List[float] = []
    rgb_psnr_values: List[float] = []

    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader):
            parsed = parse_batch(batch)
            frames = parsed["frames"].to(device, non_blocking=True)
            gt_img = parsed["gt_img"].to(device, non_blocking=True)
            text_tokens = parsed["text_tokens"]
            text_mask = parsed["text_mask"]
            if text_tokens is not None:
                text_tokens = text_tokens.to(device, non_blocking=True)
            if text_mask is not None:
                text_mask = text_mask.to(device, non_blocking=True)

            set_torch_seed(cli_args.eval_sampling_seed + batch_idx)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                samples, _ = model.sample(
                    frames,
                    sched_mode=model.train_mask_mode,
                    step=getattr(model.args, "step", 1),
                    text_tokens=text_tokens,
                    text_mask=text_mask,
                )

            pred_feats = samples[:, -1]
            batch_size, _, _, img_h, img_w = frames.shape
            gt_feats = reshape_features(model, model.extract_features(gt_img), img_h, img_w)
            ctx_feats = reshape_features(model, model.extract_features(frames[:, -2]), img_h, img_w)

            if getattr(model.args, "crop_feats", False):
                gt_feats = model.crop_feats(gt_feats.unsqueeze(1), use_crop_params=True).squeeze(1)
                ctx_feats = model.crop_feats(ctx_feats.unsqueeze(1), use_crop_params=True).squeeze(1)

            pred_flat = pred_feats.reshape(batch_size, -1, pred_feats.shape[-1]).float()
            gt_flat = gt_feats.reshape(batch_size, -1, gt_feats.shape[-1]).float()
            ctx_flat = ctx_feats.reshape(batch_size, -1, ctx_feats.shape[-1]).float()

            latent_mse_values.append(float(F.mse_loss(pred_flat, gt_flat).item()))
            latent_cos_values.append(float(F.cosine_similarity(pred_flat, gt_flat, dim=-1).mean().item()))

            pred_motion = torch.norm(pred_flat - ctx_flat, dim=-1).mean(dim=-1)
            gt_motion = torch.norm(gt_flat - ctx_flat, dim=-1).mean(dim=-1).clamp(min=1e-8)
            motion_ratio_values.append(float((pred_motion / gt_motion).mean().item()))

            if decoder is not None:
                pred_rgb = decoder.decode(pred_feats.to(device))
                gt_rgb = denormalize_images(gt_img, model.args.feature_extractor).to(pred_rgb.device)
                rgb_l1 = F.l1_loss(pred_rgb, gt_rgb)
                rgb_mse = F.mse_loss(pred_rgb, gt_rgb)
                rgb_psnr = -10.0 * torch.log10(rgb_mse.clamp(min=1e-8))
                rgb_l1_values.append(float(rgb_l1.item()))
                rgb_mse_values.append(float(rgb_mse.item()))
                rgb_psnr_values.append(float(rgb_psnr.item()))

    diversity_mean = None
    if qual_indices and cli_args.num_samples_per_input > 1:
        diversity_values = render_or_collect_qualitative(
            checkpoint_info=checkpoint_info,
            cli_args=cli_args,
            device=device,
            eval_args=eval_args,
            model=model,
            checkpoint_name=label,
            checkpoint_step=meta["step"],
            qual_indices=qual_indices,
            decoder=decoder,
            qual_cache=qual_cache,
            record_diversity_only=False,
        )
        diversity_mean = metric_mean(diversity_values)
    elif qual_indices:
        render_or_collect_qualitative(
            checkpoint_info=checkpoint_info,
            cli_args=cli_args,
            device=device,
            eval_args=eval_args,
            model=model,
            checkpoint_name=label,
            checkpoint_step=meta["step"],
            qual_indices=qual_indices,
            decoder=decoder,
            qual_cache=qual_cache,
            record_diversity_only=False,
        )

    result = {
        "checkpoint": ckpt_path,
        "checkpoint_name": label,
        "step": meta["step"],
        "epoch": meta["epoch"],
        "latent_mse": metric_mean(latent_mse_values),
        "latent_cosine": metric_mean(latent_cos_values),
        "motion_mag_ratio": metric_mean(motion_ratio_values),
        "rgb_l1": metric_mean(rgb_l1_values),
        "rgb_mse": metric_mean(rgb_mse_values),
        "rgb_psnr": metric_mean(rgb_psnr_values),
        "sample_diversity_cosine": diversity_mean,
    }
    return result, eval_args


def render_or_collect_qualitative(
    checkpoint_info: Dict[str, Any],
    cli_args,
    device: torch.device,
    eval_args: argparse.Namespace,
    model: Dino_f,
    checkpoint_name: str,
    checkpoint_step: Optional[int],
    qual_indices: Sequence[int],
    decoder: Optional[DecoderWrapper],
    qual_cache: Dict[int, Dict[str, Any]],
    record_diversity_only: bool = False,
) -> List[float]:
    full_dataset, collate_fn = build_dataset(eval_args, cli_args.data_split)
    qual_dataset = Subset(full_dataset, list(qual_indices))
    loader = create_dataloader(qual_dataset, collate_fn, batch_size=1, num_workers=min(cli_args.num_workers, 2))

    ckpt_path = checkpoint_info["path"]
    label = checkpoint_name
    diversity_values: List[float] = []

    with torch.inference_mode():
        for sample_pos, batch in enumerate(loader):
            dataset_index = int(qual_indices[sample_pos])
            parsed = parse_batch(batch)
            frames = parsed["frames"].to(device, non_blocking=True)
            gt_img = parsed["gt_img"].to(device, non_blocking=True)
            text_tokens = parsed["text_tokens"]
            text_mask = parsed["text_mask"]
            if text_tokens is not None:
                text_tokens = text_tokens.to(device, non_blocking=True)
            if text_mask is not None:
                text_mask = text_mask.to(device, non_blocking=True)

            sample_record = qual_cache.setdefault(
                dataset_index,
                {
                    "dataset_index": dataset_index,
                    "context_images": None,
                    "gt_image": None,
                    "gt_latent": None,
                    "predictions": {},
                    "visual_mode": "rgb_decoder" if decoder is not None else "latent_pca",
                },
            )
            raw_frames = denormalize_images(frames, model.args.feature_extractor).cpu()
            raw_gt = denormalize_images(gt_img, model.args.feature_extractor).cpu()
            if sample_record["context_images"] is None:
                sample_record["context_images"] = [tensor_to_pil(raw_frames[0, idx]) for idx in range(raw_frames.shape[1] - 1)]
                sample_record["gt_image"] = tensor_to_pil(raw_gt[0])
                batch_size, _, _, img_h, img_w = frames.shape
                gt_feats = reshape_features(model, model.extract_features(gt_img), img_h, img_w)
                if getattr(model.args, "crop_feats", False):
                    gt_feats = model.crop_feats(gt_feats.unsqueeze(1), use_crop_params=True).squeeze(1)
                sample_record["gt_latent"] = gt_feats[0].detach().cpu().to(torch.float16)

            latent_samples: List[torch.Tensor] = []
            decoded_samples: List[Image.Image] = []
            for rep in range(cli_args.num_samples_per_input):
                seed = cli_args.qual_seed + sample_pos * 1000 + rep
                set_torch_seed(seed)
                samples, _ = model.sample(
                    frames,
                    sched_mode=model.train_mask_mode,
                    step=getattr(model.args, "step", 1),
                    text_tokens=text_tokens,
                    text_mask=text_mask,
                )
                pred_feat = samples[0, -1].detach().cpu()
                latent_samples.append(pred_feat.to(torch.float16))
                if decoder is not None:
                    pred_rgb = decoder.decode(samples[:, -1].to(device))[0].cpu()
                    decoded_samples.append(tensor_to_pil(pred_rgb))

            if len(latent_samples) >= 2:
                flat_a = latent_samples[0].reshape(-1, latent_samples[0].shape[-1]).float()
                flat_b = latent_samples[1].reshape(-1, latent_samples[1].shape[-1]).float()
                diversity_values.append(float(F.cosine_similarity(flat_a, flat_b, dim=-1).mean().item()))

            sample_record["predictions"][label] = {
                "checkpoint_path": ckpt_path,
                "step": checkpoint_step,
                "latents": latent_samples,
                "decoded_images": decoded_samples,
            }

    return diversity_values


def write_results_csv(path: Path, results: Sequence[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    fieldnames = [
        "checkpoint_name",
        "checkpoint",
        "step",
        "epoch",
        "latent_mse",
        "latent_cosine",
        "motion_mag_ratio",
        "rgb_l1",
        "rgb_mse",
        "rgb_psnr",
        "sample_diversity_cosine",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({key: row.get(key) for key in fieldnames})


def format_metric(value: Optional[float], precision: int = 6) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{precision}f}"


def write_results_markdown(path: Path, results: Sequence[Dict[str, Any]]) -> None:
    headers = [
        "checkpoint_name",
        "step",
        "latent_mse",
        "latent_cosine",
        "motion_mag_ratio",
        "rgb_psnr",
        "sample_diversity_cosine",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in results:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("checkpoint_name")),
                    str(row.get("step")),
                    format_metric(row.get("latent_mse")),
                    format_metric(row.get("latent_cosine")),
                    format_metric(row.get("motion_mag_ratio")),
                    format_metric(row.get("rgb_psnr")),
                    format_metric(row.get("sample_diversity_cosine")),
                ]
            )
            + " |"
        )
    ensure_dir(path.parent)
    path.write_text("\n".join(lines) + "\n")


def best_checkpoint(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return min(
        results,
        key=lambda row: (
            float("inf") if row.get("latent_mse") is None else row["latent_mse"],
            -(row.get("latent_cosine") if row.get("latent_cosine") is not None else -float("inf")),
            -(row.get("rgb_psnr") if row.get("rgb_psnr") is not None else -float("inf")),
        ),
    )


def compare_float(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return a - b


def summarize_trend(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    rows = sorted(results, key=lambda item: ((item.get("step") or int(1e18)), item["checkpoint_name"]))
    best = best_checkpoint(rows)
    first = rows[0]
    latest = rows[-1]
    first_mse = first.get("latent_mse")
    best_mse = best.get("latent_mse")
    latest_mse = latest.get("latent_mse")

    improvement = None
    latest_gap = None
    if first_mse is not None and best_mse is not None and first_mse != 0:
        improvement = (first_mse - best_mse) / abs(first_mse)
    if latest_mse is not None and best_mse is not None and best_mse != 0:
        latest_gap = (latest_mse - best_mse) / abs(best_mse)

    best_is_latest = best["checkpoint_name"] == latest["checkpoint_name"]
    if best_is_latest:
        trend_text = "Later checkpoints are still improving or the latest checkpoint is currently best."
        recommendation = "continue_training_from_latest_checkpoint"
    elif latest_gap is not None and latest_gap <= 0.01:
        trend_text = "Later checkpoints have mostly plateaued near the best checkpoint."
        recommendation = "continue_training_from_latest_checkpoint"
    elif improvement is not None and improvement < 0.01:
        trend_text = "Checkpoint quality is mostly flat across the sampled training range."
        recommendation = "switch_to_new_experiment"
    else:
        trend_text = "Performance improved and then regressed or plateaued after the best checkpoint."
        recommendation = "stop_and_use_best_current_checkpoint"

    return {
        "sorted_results": rows,
        "best": best,
        "latest": latest,
        "improvement_vs_first": improvement,
        "latest_gap_vs_best": latest_gap,
        "trend_text": trend_text,
        "recommendation_code": recommendation,
    }


def qualitative_agreement_text(results: Sequence[Dict[str, Any]]) -> str:
    best_latent = best_checkpoint(results)
    rgb_candidates = [row for row in results if row.get("rgb_psnr") is not None]
    if not rgb_candidates:
        return "Automatic qualitative agreement is inconclusive without decoder-based RGB metrics; inspect the saved qualitative grids."
    best_rgb = max(rgb_candidates, key=lambda row: row["rgb_psnr"])
    if best_rgb["checkpoint_name"] == best_latent["checkpoint_name"]:
        return "Optional RGB metrics agree with the latent ranking for the current best checkpoint."
    return "Latent metrics and optional RGB metrics disagree on the top checkpoint; inspect the saved qualitative grids before deciding."


def heuristic_issue_flags(best_row: Dict[str, Any]) -> List[str]:
    flags = []
    motion_ratio = best_row.get("motion_mag_ratio")
    diversity = best_row.get("sample_diversity_cosine")
    if motion_ratio is not None and motion_ratio < 0.6:
        flags.append("possible oversmoothing or weak motion relative to the context-to-GT change")
    if motion_ratio is not None and motion_ratio > 1.5:
        flags.append("possible noisy or overshooting predictions")
    if diversity is not None and diversity > 0.995:
        flags.append("possible mode collapse or very low sample diversity")
    if diversity is not None and diversity < 0.85:
        flags.append("possible unstable sampling across repeated draws")
    return flags


def save_summary_markdown(
    path: Path,
    selected_checkpoints: Sequence[Dict[str, Any]],
    eval_indices: Sequence[int],
    qual_indices: Sequence[int],
    trend_summary: Dict[str, Any],
    results_md_path: Path,
    qualitative_dir: Path,
) -> None:
    best = trend_summary["best"]
    issue_flags = heuristic_issue_flags(best)
    recommendation_map = {
        "continue_training_from_latest_checkpoint": "continue training from the latest checkpoint",
        "stop_and_use_best_current_checkpoint": "stop and use the best current checkpoint",
        "switch_to_new_experiment": "switch to a new experiment instead",
    }
    lines = [
        "# Checkpoint Comparison Summary",
        "",
        "## Selected checkpoints",
    ]
    for item in selected_checkpoints:
        lines.append(f"- `{checkpoint_label(item['path'], item.get('step'))}`: `{item['path']}`")
    lines.extend(
        [
            "",
            "## Evaluation subset",
            f"- samples: {len(eval_indices)}",
            f"- qualitative samples: {len(qual_indices)}",
            f"- dataset indices: `{list(eval_indices)}`",
            "",
            "## Metrics table",
            f"- full table: `{results_md_path}`",
            "",
            "## Best checkpoint recommendation",
            f"- best checkpoint: `{best['checkpoint_name']}`",
            f"- step: `{best.get('step')}`",
            f"- latent_mse: `{format_metric(best.get('latent_mse'))}`",
            f"- latent_cosine: `{format_metric(best.get('latent_cosine'))}`",
            f"- trend: {trend_summary['trend_text']}",
            f"- qualitative agreement: {qualitative_agreement_text(trend_summary['sorted_results'])}",
            f"- recommendation: {recommendation_map[trend_summary['recommendation_code']]}",
            f"- qualitative outputs: `{qualitative_dir}`",
        ]
    )
    if issue_flags:
        lines.append("- heuristic flags: " + "; ".join(issue_flags))
    else:
        lines.append("- heuristic flags: no strong automatic warning signs detected")
    ensure_dir(path.parent)
    path.write_text("\n".join(lines) + "\n")


def sanitize_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", name)


def render_qualitative_outputs(qual_cache: Dict[int, Dict[str, Any]], output_dir: Path, decoder: Optional[DecoderWrapper]) -> None:
    if not qual_cache:
        return
    qual_root = output_dir / "qualitative"
    combined_root = qual_root / "combined"
    ensure_dir(combined_root)
    html_entries = []

    for dataset_index, record in sorted(qual_cache.items(), key=lambda item: item[0]):
        sample_dir = combined_root / f"sample_{dataset_index:05d}"
        ensure_dir(sample_dir)
        context_strip = hstack([add_label(image, f"context_{idx}") for idx, image in enumerate(record["context_images"])])
        gt_image = add_label(record["gt_image"], "gt_future")

        per_ckpt_main_images = []
        pca_inputs = [record["gt_latent"].float()]
        for pred in record["predictions"].values():
            if pred["latents"]:
                pca_inputs.append(pred["latents"][0].float())

        pca_basis = None
        if decoder is None:
            pca_basis = fit_pca_from_latents(pca_inputs)

        for ckpt_name, pred in sorted(record["predictions"].items()):
            ckpt_dir = qual_root / sanitize_name(ckpt_name) / f"sample_{dataset_index:05d}"
            ensure_dir(ckpt_dir)
            context_strip.save(ckpt_dir / "context.png")
            gt_image.save(ckpt_dir / "gt_future.png")

            latents = [latent.float() for latent in pred["latents"]]
            if decoder is not None:
                images = pred["decoded_images"]
                if not images:
                    stacked = torch.stack(latents, dim=0)
                    decoded = decoder.decode(stacked)
                    images = [tensor_to_pil(decoded[idx].cpu()) for idx in range(decoded.shape[0])]
            else:
                pca_mean, pca_components, pca_min, pca_max = pca_basis
                images = [
                    latent_to_pca_image(latent, pca_mean, pca_components, pca_min, pca_max)
                    for latent in latents
                ]

            labeled_images = [add_label(image, f"{ckpt_name}_sample_{idx}") for idx, image in enumerate(images)]
            for idx, image in enumerate(labeled_images):
                image.save(ckpt_dir / f"pred_{idx:02d}.png")

            pred_main = add_label(images[0], ckpt_name)
            pred_main.save(ckpt_dir / "pred_main.png")
            if len(images) > 1:
                hstack(labeled_images).save(ckpt_dir / "pred_samples_strip.png")
            per_ckpt_main_images.append(pred_main)

        bottom_row = hstack([gt_image] + per_ckpt_main_images)
        combined = vstack([add_label(context_strip, "context_frames"), bottom_row])
        combined_path = sample_dir / "combined.png"
        combined.save(combined_path)
        html_entries.append(
            {
                "dataset_index": dataset_index,
                "combined_path": combined_path.relative_to(qual_root),
            }
        )

    html_lines = [
        "<html><head><meta charset='utf-8'><title>Checkpoint Comparison Qualitative Summary</title>",
        "<style>body{font-family:sans-serif;background:#111;color:#eee;padding:24px;} img{max-width:100%;border:1px solid #444;} .sample{margin-bottom:32px;} a{color:#9cd;}</style>",
        "</head><body>",
        "<h1>Checkpoint Comparison Qualitative Summary</h1>",
    ]
    for entry in html_entries:
        html_lines.append("<div class='sample'>")
        html_lines.append(f"<h2>Sample index {entry['dataset_index']}</h2>")
        html_lines.append(f"<img src='{entry['combined_path'].as_posix()}' alt='sample {entry['dataset_index']}'>")
        html_lines.append("</div>")
    html_lines.append("</body></html>")
    (qual_root / "index.html").write_text("\n".join(html_lines) + "\n")


def main():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config_path", type=str, default=None)
    pre_args, _ = pre_parser.parse_known_args()
    config = normalize_config_values(load_config(pre_args.config_path))
    parser = build_parser(config_defaults=config)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)
    if args.save_args_json:
        save_json(output_dir / "run_args.json", vars(args))

    device = resolve_device(args.device)
    log(f"Using device: {device}")

    selected_checkpoints = discover_checkpoints(args)
    save_json(output_dir / "selected_checkpoints.json", {"checkpoints": selected_checkpoints})

    first_payload = load_checkpoint_payload(selected_checkpoints[0]["path"])
    first_args = apply_cli_overrides(extract_checkpoint_args(first_payload), args)
    first_dataset, _ = build_dataset(first_args, args.data_split)
    eval_indices = load_or_create_eval_indices(first_dataset, args, output_dir)
    qual_indices = select_qualitative_indices(eval_indices, args.qual_num_samples, args.qual_seed, output_dir)

    decoder = load_optional_decoder(args.rgb_decoder_path, device)
    qual_cache: Dict[int, Dict[str, Any]] = {}
    results = []
    for checkpoint_info in selected_checkpoints:
        result, _ = evaluate_checkpoint(
            checkpoint_info=checkpoint_info,
            cli_args=args,
            device=device,
            eval_indices=eval_indices,
            qual_indices=qual_indices,
            decoder=decoder,
            qual_cache=qual_cache,
        )
        results.append(result)

    results = sorted(results, key=lambda row: ((row.get("step") or int(1e18)), row["checkpoint_name"]))
    write_results_csv(output_dir / "results.csv", results)
    write_results_markdown(output_dir / "results.md", results)
    render_qualitative_outputs(qual_cache, output_dir, decoder)

    trend_summary = summarize_trend(results)
    save_summary_markdown(
        path=output_dir / "summary.md",
        selected_checkpoints=selected_checkpoints,
        eval_indices=eval_indices,
        qual_indices=qual_indices,
        trend_summary=trend_summary,
        results_md_path=output_dir / "results.md",
        qualitative_dir=output_dir / "qualitative",
    )

    log(f"Saved quantitative results to {output_dir / 'results.csv'}")
    log(f"Saved markdown table to {output_dir / 'results.md'}")
    log(f"Saved qualitative outputs to {output_dir / 'qualitative'}")
    log(f"Saved summary report to {output_dir / 'summary.md'}")
    best = trend_summary["best"]
    log(
        "Best checkpoint: "
        f"{best['checkpoint_name']} "
        f"(latent_mse={format_metric(best.get('latent_mse'))}, "
        f"latent_cosine={format_metric(best.get('latent_cosine'))})"
    )


if __name__ == "__main__":
    main()
