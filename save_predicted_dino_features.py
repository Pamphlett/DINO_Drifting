import argparse
import os
import os.path as osp
from itertools import islice
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.data import CS_VideoData, OpenDV_VideoData
from src.dino_f import Dino_f


def parse_tuple(x):
    return tuple(map(int, x.split(",")))


def parse_list(x):
    return list(map(int, x.split(",")))


def denormalize_images(tensor, feature_extractor):
    if feature_extractor in ("dino", "sam"):
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    else:
        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)
    mean = torch.tensor(mean, device=tensor.device).view(1, 3, 1, 1)
    std = torch.tensor(std, device=tensor.device).view(1, 3, 1, 1)
    return torch.clamp(tensor * std + mean, 0, 1)


def rollout_predictions(model, frames, unroll_steps, text_tokens=None, text_mask=None):
    preds = []
    with torch.no_grad():
        x = model.preprocess(frames)
        if not getattr(model, "use_language_condition", False):
            text_tokens = None
            text_mask = None
        if text_tokens is not None:
            text_tokens = text_tokens.to(x.device)
            if hasattr(model, "text_proj"):
                in_dim = model.text_proj.in_features
                out_dim = model.text_proj.out_features
                if text_tokens.shape[-1] == in_dim:
                    text_tokens = text_tokens.to(dtype=model.text_proj.weight.dtype)
                    text_tokens = model.text_proj(text_tokens)
                elif text_tokens.shape[-1] != out_dim:
                    raise ValueError(
                        f"Unexpected text token dim {text_tokens.shape[-1]} (expected {in_dim} or {out_dim})."
                    )
        if text_mask is not None:
            text_mask = text_mask.to(x.device)
        for _ in range(unroll_steps):
            masked_x, mask = model.get_mask_tokens(
                x, mode="full_mask", mask_frames=1
            )
            mask = mask.to(x.device)
            if model.args.vis_attn:
                _, final_tokens, _ = model.forward(
                    x, masked_x, mask, text_tokens=text_tokens, text_mask=text_mask
                )
            else:
                _, final_tokens = model.forward(
                    x, masked_x, mask, text_tokens=text_tokens, text_mask=text_mask
                )
            x[:, -1] = final_tokens[:, -1]
            x[:, 0:-1] = x[:, 1:].clone()
            pred_feats = model.postprocess(x)[:, -1]
            preds.append(pred_feats.detach().cpu())
    return torch.stack(preds, dim=1)  # [B, S, H, W, C]


def add_missing_args(args, model_args):
    defaults = {
        "opendv_root": None,
        "opendv_train_root": None,
        "opendv_val_root": None,
        "opendv_meta_path": None,
        "opendv_lang_root": None,
        "opendv_use_lang_annos": False,
        "opendv_filter_folder": None,
        "opendv_max_clips": None,
        "opendv_video_dir": None,
        "opendv_return_language": False,
        "opendv_lang_cache_path": None,
        "opendv_lang_cache_train": None,
        "opendv_lang_cache_val": None,
        "opendv_feat_ext": ".dinov2.pt",
        "opendv_use_lang_features": False,
        "opendv_lang_feat_name": "lang_clip_{start}_{end}.pt",
        "opendv_lang_feat_key": "text_tokens",
        "opendv_lang_mask_key": "attention_mask",
        "num_workers_val": None,
        "random_crop": False,
        "random_horizontal_flip": False,
        "random_time_flip": False,
        "timestep_augm": None,
        "no_timestep_augm": False,
        "ddp_stage_log_path": None,
        "use_val_to_train": False,
        "use_train_to_val": False,
        "dataloader_timeout": 0,
        "dataloader_log_path": None,
        "dataloader_log_every": 0,
        "eval_mode": True,
        "eval_midterm": False,
        "eval_modality": None,
        "use_language_condition": False,
        "use_precomputed_text": False,
        "use_precomputed_feats": False,
        "clip_model_name": "openai/clip-vit-base-patch32",
        "clip_max_length": 77,
        "clip_cache_dir": None,
        "clip_local_files_only": False,
        "clip_text_dim": 512,
        "return_rgb_path": False,
    }
    for key, value in defaults.items():
        if not hasattr(args, key):
            setattr(args, key, value)
    if not hasattr(args, "feature_extractor"):
        setattr(args, "feature_extractor", model_args.feature_extractor)
    if not hasattr(args, "dinov2_variant"):
        setattr(args, "dinov2_variant", model_args.dinov2_variant)
    return args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--dataset", choices=["opendv", "cityscapes"], default="opendv")
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--opendv_root", type=str, default=None)
    parser.add_argument("--opendv_lang_root", type=str, default=None)
    parser.add_argument("--opendv_use_lang_annos", action="store_true", default=False)
    parser.add_argument("--opendv_lang_cache_train", type=str, default=None)
    parser.add_argument("--opendv_lang_cache_val", type=str, default=None)
    parser.add_argument("--opendv_use_lang_features", action="store_true", default=False)
    parser.add_argument("--opendv_lang_feat_name", type=str, default="lang_clip_{start}_{end}.pt")
    parser.add_argument("--opendv_video_dir", type=str, default=None)
    parser.add_argument("--sequence_length", type=int, default=5)
    parser.add_argument("--img_size", type=parse_tuple, default=(196, 392))
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--subset", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--unroll_steps", type=int, default=1)
    parser.add_argument("--save_dtype", type=str, default="float16", choices=["float16", "float32"])
    parser.add_argument("--save_rgb", action="store_true", default=False)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Dino_f.load_from_checkpoint(args.ckpt, strict=False).to(device)
    model.eval()
    model._init_feature_extractor()
    if model.dino_v2 is not None:
        model.dino_v2 = model.dino_v2.to(device)
    if model.eva2clip is not None:
        model.eva2clip = model.eva2clip.to(device)
    if model.sam is not None:
        model.sam = model.sam.to(device)

    args = add_missing_args(args, model.args)
    args.feature_extractor = model.args.feature_extractor
    args.dinov2_variant = getattr(model.args, "dinov2_variant", "vitb14_reg")
    args.return_rgb_path = True
    torch.backends.cuda.matmul.allow_tf32 = True

    if args.dataset == "opendv":
        data = OpenDV_VideoData(arguments=args, subset=args.subset, batch_size=args.batch_size)
    else:
        data = CS_VideoData(arguments=args, subset=args.subset, batch_size=args.batch_size)

    loader = data.val_dataloader() if args.subset == "val" else data.train_dataloader()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    save_dtype = torch.float16 if args.save_dtype == "float16" else torch.float32
    try:
        loader_len = len(loader)
    except TypeError:
        loader_len = None
    if args.max_batches is not None:
        loader_iter = islice(loader, args.max_batches)
        if loader_len is not None:
            total_batches = min(loader_len, args.max_batches)
        else:
            total_batches = args.max_batches
    else:
        loader_iter = loader
        total_batches = loader_len

    num_saved = 0
    progress = tqdm(loader_iter, total=total_batches, desc="Batches", unit="batch")
    printed_text_info = False
    prev_end = time.time()
    accum_data = 0.0
    accum_compute = 0.0
    accum_save = 0.0
    accum_batches = 0
    log_every = 20
    with torch.inference_mode():
        for batch_idx, batch in enumerate(progress):
            t0 = time.time()
            if isinstance(batch, (list, tuple)):
                frames = batch[0]
                gt_img = None
                text_tokens = None
                text_mask = None
                rgb_paths = None
                for item in batch[1:]:
                    if not torch.is_tensor(item):
                        if isinstance(item, (list, tuple)) and item:
                            first = item[0]
                            if isinstance(first, (str, Path)):
                                first_str = str(first).lower()
                                if first_str.endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
                                    rgb_paths = [str(p) for p in item]
                        continue
                    if item.ndim == 4 and item.shape[1] == 3:
                        gt_img = item
                    elif item.ndim == 3:
                        text_tokens = item
                    elif item.ndim == 2:
                        text_mask = item
            else:
                frames = batch
                gt_img = None
                text_tokens = None
                text_mask = None
                rgb_paths = None

            frames = frames.to(device, non_blocking=True)
            if text_tokens is not None:
                text_tokens = text_tokens.to(device, non_blocking=True)
            if text_mask is not None:
                text_mask = text_mask.to(device, non_blocking=True)
            t1 = time.time()
            if not printed_text_info:
                if text_tokens is not None:
                    progress.write(
                        f"Using text tokens: shape={tuple(text_tokens.shape)} "
                        f"mask_shape={tuple(text_mask.shape) if text_mask is not None else None} "
                        f"use_language_condition={getattr(model, 'use_language_condition', False)}"
                    )
                    printed_text_info = True
                else:
                    progress.write(
                        f"No text tokens in batch; use_language_condition="
                        f"{getattr(model, 'use_language_condition', False)}"
                    )
                    printed_text_info = True
            amp_enabled = device.type == "cuda"
            amp_dtype = torch.float16
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                preds = rollout_predictions(
                    model,
                    frames,
                    args.unroll_steps,
                    text_tokens=text_tokens,
                    text_mask=text_mask,
                )
            t2 = time.time()

            if gt_img is not None:
                gt_rgb = denormalize_images(gt_img, args.feature_extractor).to("cpu")
            elif args.save_rgb and frames.ndim == 5 and frames.shape[2] == 3:
                gt_rgb = denormalize_images(frames[:, -1], args.feature_extractor).to("cpu")
            else:
                gt_rgb = None

            for i in range(preds.shape[0]):
                payload = {
                    "pred_features": preds[i].to(dtype=save_dtype),
                    "feature_extractor": model.args.feature_extractor,
                    "d_layers": model.args.d_layers,
                    "img_size": list(args.img_size),
                    "patch_size": model.patch_size,
                    "source_ckpt": args.ckpt,
                    "unroll_steps": args.unroll_steps,
                }
                if rgb_paths is not None:
                    payload["rgb_path"] = rgb_paths[i]
                if gt_rgb is not None:
                    payload["rgb"] = gt_rgb[i].to(dtype=save_dtype)
                out_path = out_root / f"pred_{num_saved:07d}.pt"
                torch.save(payload, out_path)
                num_saved += 1
            progress.set_postfix(saved=num_saved)
            t3 = time.time()
            accum_data += t1 - t0
            accum_compute += t2 - t1
            accum_save += t3 - t2
            accum_batches += 1
            if accum_batches >= log_every:
                progress.write(
                    f"avg/batch data={accum_data/accum_batches:.3f}s "
                    f"compute={accum_compute/accum_batches:.3f}s "
                    f"save={accum_save/accum_batches:.3f}s"
                )
                accum_data = 0.0
                accum_compute = 0.0
                accum_save = 0.0
                accum_batches = 0

    print(f"Saved {num_saved} samples to {out_root}")


if __name__ == "__main__":
    main()
