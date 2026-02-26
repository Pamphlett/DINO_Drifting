import argparse
import json
import os
import os.path as osp
import shutil

import torch
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from src.data import _iter_json_array


def cmd_to_caption(cmd):
    plain_caption_dict = {
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
    return plain_caption_dict.get(int(cmd), "Go straight.")


def resolve_clip_sources(model_id, cache_dir, local_only):
    if osp.isdir(model_id):
        return model_id, model_id
    offline = local_only or os.environ.get("HF_HUB_OFFLINE") or os.environ.get("TRANSFORMERS_OFFLINE")
    if not offline:
        return model_id, model_id
    cache_root = cache_dir or os.environ.get("HF_HOME")
    if cache_root and osp.basename(cache_root.rstrip("/")) != "hub":
        if osp.isdir(osp.join(cache_root, "hub")):
            cache_root = osp.join(cache_root, "hub")
    if not cache_root:
        return model_id, model_id
    model_dir = osp.join(cache_root, "models--" + model_id.replace("/", "--"))
    snapshots_dir = osp.join(model_dir, "snapshots")
    if not osp.isdir(snapshots_dir):
        return model_id, model_id
    snapshot_paths = [
        osp.join(snapshots_dir, name)
        for name in os.listdir(snapshots_dir)
        if osp.isdir(osp.join(snapshots_dir, name))
    ]
    tokenizer_path = None
    text_model_path = None
    for snap in snapshot_paths:
        if tokenizer_path is None and (
            osp.isfile(osp.join(snap, "tokenizer.json"))
            or osp.isfile(osp.join(snap, "vocab.json"))
        ):
            tokenizer_path = snap
        if text_model_path is None and osp.isfile(osp.join(snap, "model.safetensors")):
            text_model_path = snap
    if tokenizer_path and text_model_path and tokenizer_path != text_model_path:
        combined = ensure_combined_clip_dir(tokenizer_path, text_model_path)
        return combined, combined
    return tokenizer_path or model_id, text_model_path or model_id


def ensure_combined_clip_dir(tokenizer_path, text_model_path):
    combined_dir = osp.join(osp.dirname(text_model_path), "combined_clip")
    os.makedirs(combined_dir, exist_ok=True)
    for name in [
        "config.json",
        "tokenizer.json",
        "vocab.json",
        "merges.txt",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ]:
        src = osp.join(tokenizer_path, name)
        dst = osp.join(combined_dir, name)
        if osp.isfile(src) and not osp.isfile(dst):
            shutil.copy2(src, dst)
    model_src = osp.join(text_model_path, "model.safetensors")
    model_dst = osp.join(combined_dir, "model.safetensors")
    if osp.isfile(model_src) and not osp.isfile(model_dst):
        shutil.copy2(model_src, model_dst)
    return combined_dir


def iter_lang_entries(lang_root, split, lang_cache_path=None):
    if lang_cache_path:
        for entry in json.load(open(lang_cache_path, "r")):
            yield entry
        return
    if split == "val":
        ann_files = [osp.join(lang_root, "10hz_YouTube_val.json")]
    else:
        ann_files = [
            osp.join(lang_root, f"10hz_YouTube_train_split{i}.json")
            for i in range(10)
        ]
    for ann_path in ann_files:
        if not osp.isfile(ann_path):
            continue
        for entry in _iter_json_array(ann_path):
            yield entry


def clip_dir_from_entry(entry, opendv_root):
    folder = entry.get("folder")
    if not folder:
        return None
    folder_norm = folder.replace("\\", "/").lstrip("/")
    if folder_norm.startswith("full_images/") or folder_norm.startswith("val_images/"):
        return osp.join(opendv_root, folder_norm)
    return osp.join(opendv_root, folder_norm)


def format_feature_name(pattern, start_id, end_id, pad):
    start = str(start_id).zfill(pad)
    end = str(end_id).zfill(pad)
    if "{start}" in pattern or "{end}" in pattern:
        return pattern.format(start=start, end=end)
    return pattern


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--opendv_root", required=True, help="Root containing full_images/val_images.")
    parser.add_argument("--lang_root", default=None, help="Path to OpenDV-YouTube-Language.")
    parser.add_argument("--lang_cache", default=None, help="Optional JSON cache of language entries.")
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--filter_folder", default=None)
    parser.add_argument("--max_entries", type=int, default=None)
    parser.add_argument("--feature_name", default="lang_clip_{start}_{end}.pt")
    parser.add_argument("--overwrite", action="store_true", default=True)
    parser.add_argument("--clip_model_name", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--clip_cache_dir", type=str, default=None)
    parser.add_argument("--clip_local_files_only", action="store_true", default=False)
    parser.add_argument("--clip_max_length", type=int, default=77)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32"])
    args = parser.parse_args()

    if not args.lang_cache and not args.lang_root:
        raise ValueError("Provide --lang_cache or --lang_root.")

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer_source, text_model_source = resolve_clip_sources(
        args.clip_model_name,
        args.clip_cache_dir,
        args.clip_local_files_only,
    )
    tokenizer = CLIPTokenizer.from_pretrained(
        tokenizer_source,
        cache_dir=args.clip_cache_dir,
        local_files_only=args.clip_local_files_only,
    )
    text_model = CLIPTextModel.from_pretrained(
        text_model_source,
        use_safetensors=True,
        cache_dir=args.clip_cache_dir,
        local_files_only=args.clip_local_files_only,
    )
    text_model.eval()
    text_model.to(device)

    pending = []
    total = 0
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    entries = iter_lang_entries(args.lang_root, args.split, lang_cache_path=args.lang_cache)
    for entry in tqdm(entries, desc="Queueing entries"):
        if args.max_entries and total >= args.max_entries:
            break
        folder = entry.get("folder")
        if not folder:
            continue
        if args.filter_folder and args.filter_folder not in folder:
            continue
        first_frame = entry.get("first_frame")
        last_frame = entry.get("last_frame")
        if not first_frame or not last_frame:
            continue
        clip_dir = clip_dir_from_entry(entry, args.opendv_root)
        if not clip_dir or not osp.isdir(clip_dir):
            continue
        start_id = int(osp.splitext(first_frame)[0])
        end_id = int(osp.splitext(last_frame)[0])
        pad = len(osp.splitext(first_frame)[0])
        feat_name = format_feature_name(args.feature_name, start_id, end_id, pad)
        feat_path = osp.join(clip_dir, feat_name)
        if osp.isfile(feat_path) and not args.overwrite:
            continue
        cmd = entry.get("cmd")
        if cmd is None:
            continue
        caption = cmd_to_caption(cmd)
        pending.append((caption, feat_path, cmd, entry.get("blip")))
        if args.max_entries and len(pending) + total >= args.max_entries:
            remaining = args.max_entries - total
            total += encode_and_save(
                tokenizer,
                text_model,
                pending[:remaining],
                device,
                dtype,
                args.clip_max_length,
            )
            pending = []
            break
        if len(pending) >= args.batch_size:
            total += encode_and_save(tokenizer, text_model, pending, device, dtype, args.clip_max_length)
            pending = []

    if pending:
        total += encode_and_save(tokenizer, text_model, pending, device, dtype, args.clip_max_length)
    print(f"Saved {total} feature files.")


def encode_and_save(tokenizer, text_model, pending, device, dtype, max_length):
    captions = [item[0] for item in pending]
    tokens = tokenizer(
        captions,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = tokens["input_ids"].to(device)
    attention_mask = tokens["attention_mask"].to(device)
    with torch.no_grad():
        outputs = text_model(input_ids=input_ids, attention_mask=attention_mask)
    text_tokens = outputs.last_hidden_state.to("cpu", dtype=dtype)
    attention_mask_cpu = attention_mask.to("cpu")
    for idx, (_, feat_path, cmd, blip) in enumerate(pending):
        payload = {
            "text_tokens": text_tokens[idx],
            "attention_mask": attention_mask_cpu[idx],
            "cmd": cmd,
            "blip": blip,
        }
        os.makedirs(osp.dirname(feat_path), exist_ok=True)
        torch.save(payload, feat_path)
    return len(pending)


if __name__ == "__main__":
    main()
