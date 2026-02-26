import argparse
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from train_rgb_decoder_from_feats import FeatureRgbDataset, FeatureRgbDecoder, parse_tuple


def resolve_ckpt(ckpt_path, dst_path):
    if ckpt_path:
        return Path(ckpt_path)
    root = Path(dst_path)
    if not root.exists():
        raise FileNotFoundError(f"Checkpoint root not found: {root}")
    candidates = list(root.rglob("last.ckpt"))
    if not candidates:
        candidates = list(root.rglob("*.ckpt"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint found under {root}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def coalesce(value, fallback):
    return value if value is not None else fallback


def tensor_to_pil(tensor):
    tensor = tensor.detach().cpu().clamp(0, 1)
    if tensor.dim() == 3:
        tensor = tensor.permute(1, 2, 0)
    arr = (tensor.numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def make_side_by_side(gt_img, pred_img, label_left="gt", label_right="pred"):
    w, h = gt_img.size
    canvas = Image.new("RGB", (w * 2, h), color=(0, 0, 0))
    canvas.paste(gt_img, (0, 0))
    canvas.paste(pred_img, (w, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 6), label_left, fill=(255, 255, 255))
    draw.text((w + 6, 6), label_right, fill=(255, 255, 255))
    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--dst_path", type=str, default="./logs/rgb_decoder_from_feats")
    parser.add_argument("--train_feat_root", type=str, default=None)
    parser.add_argument("--train_rgb_root", type=str, default=None)
    parser.add_argument("--val_feat_root", type=str, default=None)
    parser.add_argument("--val_rgb_root", type=str, default=None)
    parser.add_argument("--img_size", type=parse_tuple, default=None)
    parser.add_argument("--feat_ext", type=str, default=None)
    parser.add_argument("--rgb_ext", type=str, default=None)
    parser.add_argument("--feat_key", type=str, default=None)
    parser.add_argument("--rgb_key", type=str, default=None)
    parser.add_argument("--feat_time_index", type=int, default=None)
    parser.add_argument("--index", type=int, default=None)
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--out_dir", type=str, default="./rgb_decoder_from_feats_val_samples")
    parser.add_argument("--out_path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    ckpt_path = resolve_ckpt(args.ckpt, args.dst_path)
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = FeatureRgbDecoder.load_from_checkpoint(str(ckpt_path), strict=False).to(device)
    model.eval()
    model_args = model.args

    train_feat_root = coalesce(args.train_feat_root, getattr(model_args, "train_feat_root", None))
    train_rgb_root = coalesce(args.train_rgb_root, getattr(model_args, "train_rgb_root", None))
    val_feat_root = coalesce(args.val_feat_root, getattr(model_args, "val_feat_root", None)) or train_feat_root
    val_rgb_root = coalesce(args.val_rgb_root, getattr(model_args, "val_rgb_root", None)) or train_rgb_root
    img_size = coalesce(args.img_size, getattr(model_args, "img_size", None))
    feat_ext = coalesce(args.feat_ext, getattr(model_args, "feat_ext", ".pt"))
    rgb_ext = coalesce(args.rgb_ext, getattr(model_args, "rgb_ext", ".png"))
    feat_key = coalesce(args.feat_key, getattr(model_args, "feat_key", "features"))
    rgb_key = coalesce(args.rgb_key, getattr(model_args, "rgb_key", "rgb_path"))
    feat_time_index = coalesce(args.feat_time_index, getattr(model_args, "feat_time_index", None))

    if not val_feat_root:
        raise ValueError("val_feat_root is required (or available in checkpoint args).")
    if img_size is None:
        raise ValueError("img_size is required (or available in checkpoint args).")

    dataset = FeatureRgbDataset(
        val_feat_root,
        val_rgb_root,
        img_size,
        feat_ext=feat_ext,
        rgb_ext=rgb_ext,
        feat_key=feat_key,
        rgb_key=rgb_key,
        feat_time_index=feat_time_index,
    )

    if len(dataset) == 0:
        raise ValueError("Validation dataset is empty.")

    rng = np.random.default_rng(args.seed)
    num_samples = max(1, int(args.num_samples))
    if num_samples == 1:
        if args.index is None:
            indices = [int(rng.integers(len(dataset)))]
        else:
            indices = [int(args.index)]
    else:
        replace = len(dataset) < num_samples
        indices = rng.choice(len(dataset), size=num_samples, replace=replace).tolist()

    if num_samples == 1 and args.out_path:
        out_dir = None
    else:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    for sample_idx, index in enumerate(indices):
        feats, rgb = dataset[int(index)]
        feats = feats.float().unsqueeze(0).to(device)
        rgb = rgb.float().unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(feats)

        gt_img = tensor_to_pil(rgb[0])
        pred_img = tensor_to_pil(pred[0])
        comparison = make_side_by_side(gt_img, pred_img)

        if out_dir is None:
            out_path = Path(args.out_path)
        else:
            out_path = out_dir / f"sample_{sample_idx:03d}_idx_{int(index)}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        comparison.save(out_path)

        feat_path = dataset.files[index] if hasattr(dataset, "files") else "unknown"
        print(f"Saved: {out_path}")
        print(f"Index: {int(index)}")
        print(f"Feature file: {feat_path}")

    if out_dir is not None:
        print(f"Wrote {len(indices)} samples to {out_dir}")


if __name__ == "__main__":
    main()
