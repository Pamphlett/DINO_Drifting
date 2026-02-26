import argparse
import os
import os.path as osp
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw

from src.data import OpenDV_VideoData, IMAGE_EXTS, opendv_collate, process_trainmode
from src.dino_f import Dino_f


def parse_tuple(x):
    return tuple(map(int, x.split(",")))


def denormalize_images(tensor, feature_extractor):
    if feature_extractor in ("dino", "sam"):
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    else:
        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)
    if tensor.dim() == 5:
        mean = torch.tensor(mean, device=tensor.device).view(1, 1, 3, 1, 1)
        std = torch.tensor(std, device=tensor.device).view(1, 1, 3, 1, 1)
    else:
        mean = torch.tensor(mean, device=tensor.device).view(1, 3, 1, 1)
        std = torch.tensor(std, device=tensor.device).view(1, 3, 1, 1)
    return torch.clamp(tensor * std + mean, 0, 1)


def fit_pca_from_feats(feats, n_components=3, max_samples=20000, seed=0):
    flat = feats.reshape(-1, feats.shape[-1]).float().cpu()
    if flat.shape[0] > max_samples:
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(flat.shape[0], generator=g)[:max_samples]
        flat = flat[idx]
    mean = flat.mean(0, keepdim=True)
    centered = flat - mean
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    components = vh[:n_components]
    proj = centered @ components.T
    minv = proj.min(0).values
    maxv = proj.max(0).values
    return mean.squeeze(0), components, minv, maxv


def pca_feats_to_rgb(feats, mean, components, minv, maxv):
    if feats.dim() == 4:
        feats = feats[0]
    h, w, c = feats.shape
    flat = feats.reshape(-1, c).float().cpu()
    proj = (flat - mean) @ components.T
    denom = (maxv - minv).clamp(min=1e-6)
    proj = (proj - minv) / denom
    proj = proj.clamp(0, 1)
    rgb = (proj.reshape(h, w, -1).numpy() * 255).astype(np.uint8)
    return Image.fromarray(rgb)


def rollout_predictions_from_feats(model, feats, unroll_steps):
    preds = []
    x = feats
    with torch.no_grad():
        for _ in range(unroll_steps):
            masked_x, mask = model.get_mask_tokens(x, mode="full_mask", mask_frames=1)
            mask = mask.to(x.device)
            if model.args.vis_attn:
                _, final_tokens, _ = model.forward(x, masked_x, mask)
            else:
                _, final_tokens = model.forward(x, masked_x, mask)
            x[:, -1] = final_tokens[:, -1]
            x[:, 0:-1] = x[:, 1:].clone()
            pred_feats = model.postprocess(x)[:, -1]
            preds.append(pred_feats.detach().cpu())
    return preds


def select_frames(dataset, seed, clip_idx=None, video_idx=None):
    rng = np.random.default_rng(seed)
    if getattr(dataset, "use_annotations", False):
        if not dataset.clips:
            raise ValueError("No OpenDV language clips found to sample from.")
        idx = int(clip_idx) if clip_idx is not None else int(rng.integers(len(dataset.clips)))
        clip = dataset.clips[idx]
        frames_filepaths = dataset._clip_frame_paths(clip, min_frames=dataset._min_frames_required())
        info = {"type": "clip", "index": idx}
        return frames_filepaths, info
    if not dataset.video_dirs:
        raise ValueError("No OpenDV video directories found.")
    idx = int(video_idx) if video_idx is not None else int(rng.integers(len(dataset.video_dirs)))
    video_dir = dataset.video_dirs[idx]
    frames_filepaths = sorted(
        [
            osp.join(video_dir, name)
            for name in os.listdir(video_dir)
            if name.lower().endswith(IMAGE_EXTS)
        ]
    )
    info = {"type": "video", "index": idx, "video_dir": video_dir}
    return frames_filepaths, info


def stack_two_rows(top_img, bottom_img, label_top="context", label_bottom="pred"):
    w, h = top_img.size
    canvas = Image.new("RGB", (w, h * 2), color=(0, 0, 0))
    canvas.paste(top_img, (0, 0))
    canvas.paste(bottom_img, (0, h))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 6), label_top, fill=(255, 255, 255))
    draw.text((6, h + 6), label_bottom, fill=(255, 255, 255))
    return canvas


def save_stacked_gif(context_frames, pred_frames, out_path, size, duration_ms):
    if not context_frames and not pred_frames:
        raise ValueError("No frames to render.")
    if not context_frames:
        context_frames = [Image.new("RGB", size, color=(0, 0, 0))]
    if not pred_frames:
        pred_frames = [Image.new("RGB", size, color=(0, 0, 0))]
    num_frames = max(len(context_frames), len(pred_frames))
    frames = []
    for i in range(num_frames):
        ctx = context_frames[i % len(context_frames)]
        pred = pred_frames[i % len(pred_frames)]
        if ctx.size != size:
            ctx = ctx.resize(size, resample=Image.BILINEAR)
        if pred.size != size:
            pred = pred.resize(size, resample=Image.BILINEAR)
        frames.append(stack_two_rows(ctx, pred))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--opendv_root", type=str, default=None)
    parser.add_argument("--opendv_train_root", type=str, default=None)
    parser.add_argument("--opendv_val_root", type=str, default=None)
    parser.add_argument("--opendv_meta_path", type=str, default=None)
    parser.add_argument("--opendv_lang_root", type=str, default=None)
    parser.add_argument("--opendv_use_lang_annos", action="store_true", default=None)
    parser.add_argument("--opendv_filter_folder", type=str, default=None)
    parser.add_argument("--opendv_max_clips", type=int, default=None)
    parser.add_argument("--opendv_video_dir", type=str, default=None)
    parser.add_argument("--opendv_return_language", action="store_true", default=None)
    parser.add_argument("--opendv_lang_cache_path", type=str, default=None)
    parser.add_argument("--opendv_lang_cache_train", type=str, default=None)
    parser.add_argument("--opendv_lang_cache_val", type=str, default=None)
    parser.add_argument("--opendv_feat_ext", type=str, default=".dinov2.pt")
    parser.add_argument("--opendv_use_lang_features", action="store_true", default=None)
    parser.add_argument("--opendv_lang_feat_name", type=str, default="lang_clip_{start}_{end}.pt")
    parser.add_argument("--opendv_lang_feat_key", type=str, default="text_tokens")
    parser.add_argument("--opendv_lang_mask_key", type=str, default="attention_mask")
    parser.add_argument("--sequence_length", type=int, default=None)
    parser.add_argument("--img_size", type=parse_tuple, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--num_workers_val", type=int, default=None)
    parser.add_argument("--random_crop", action="store_true", default=None)
    parser.add_argument("--random_horizontal_flip", action="store_true", default=None)
    parser.add_argument("--random_time_flip", action="store_true", default=None)
    parser.add_argument("--timestep_augm", type=list, default=None)
    parser.add_argument("--no_timestep_augm", action="store_true", default=None)
    parser.add_argument("--unroll_steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--video_idx", type=int, default=None)
    parser.add_argument("--clip_idx", type=int, default=None)
    parser.add_argument("--out_dir", type=str, default="./val_inference")
    parser.add_argument("--duration_ms", type=int, default=90)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model = Dino_f.load_from_checkpoint(args.ckpt, strict=False).to(device)
    model.eval()
    model._init_feature_extractor()
    if model.dino_v2 is not None:
        model.dino_v2 = model.dino_v2.to(device)
    if model.eva2clip is not None:
        model.eva2clip = model.eva2clip.to(device)
    if model.sam is not None:
        model.sam = model.sam.to(device)

    model_args = argparse.Namespace(**vars(model.args))
    for key in vars(args):
        if getattr(args, key) is not None:
            setattr(model_args, key, getattr(args, key))
    if model_args.data_path is None and model_args.opendv_root is None:
        raise ValueError("Provide --data_path or --opendv_root to locate OpenDV data.")
    if model_args.opendv_root is None and model_args.data_path is not None:
        model_args.opendv_root = model_args.data_path
    if model_args.sequence_length is None:
        model_args.sequence_length = model.args.sequence_length
    if model_args.img_size is None:
        model_args.img_size = model.args.img_size
    model_args.feature_extractor = model.args.feature_extractor
    model_args.dinov2_variant = getattr(model.args, "dinov2_variant", "vitb14_reg")
    model_args.eval_mode = True

    data = OpenDV_VideoData(arguments=model_args, subset="val", batch_size=model_args.batch_size)
    dataset = data._dataset("val", eval_mode=False)
    frames_filepaths, info = select_frames(dataset, args.seed, args.clip_idx, args.video_idx)

    frames = process_trainmode(
        frames_filepaths,
        model_args.img_size,
        subset="train",
        augmentations=dataset.augmentations,
        sequence_length=model_args.sequence_length,
        feature_extractor=model_args.feature_extractor,
    )
    batch = opendv_collate([frames])
    if isinstance(batch, (list, tuple)):
        frames = batch[0]
    else:
        frames = batch

    frames = frames.to(device)
    denorm = denormalize_images(frames, model_args.feature_extractor).cpu()
    context_len = max(0, frames.shape[1] - 1)
    to_pil = T.ToPILImage()
    context_rgbs = [to_pil(denorm[0, i]) for i in range(context_len)]

    with torch.no_grad():
        ctx_feats = model.preprocess(frames)
    pred_feats_list = rollout_predictions_from_feats(model, ctx_feats.clone(), args.unroll_steps)
    ctx_feats = ctx_feats.detach().cpu()

    pca_mean, pca_components, pca_min, pca_max = fit_pca_from_feats(ctx_feats, n_components=3, seed=args.seed)
    context_pca = [pca_feats_to_rgb(ctx_feats[:, -1], pca_mean, pca_components, pca_min, pca_max)]
    pred_pca = [
        pca_feats_to_rgb(pred_feats, pca_mean, pca_components, pca_min, pca_max)
        for pred_feats in pred_feats_list
    ]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{info['type']}_{info['index']}"
    out_gif = out_dir / f"val_pca_rollout_{name}.gif"
    save_stacked_gif(context_rgbs, pred_pca, out_gif, model_args.img_size[::-1], args.duration_ms)

    if context_rgbs:
        context_rgbs[0].save(out_dir / f"val_context_{name}.png")
    if pred_pca:
        pred_pca[-1].save(out_dir / f"val_pred_pca_last_{name}.png")

    info_path = out_dir / f"val_inference_{name}.txt"
    with open(info_path, "w") as f:
        f.write(f"ckpt: {args.ckpt}\n")
        f.write(f"video_source: {info}\n")
        f.write(f"frames: {len(frames_filepaths)}\n")
        f.write(f"img_size: {model_args.img_size}\n")
        f.write(f"sequence_length: {model_args.sequence_length}\n")
        f.write(f"unroll_steps: {args.unroll_steps}\n")
        f.write(f"out_gif: {out_gif}\n")

    print(f"Saved: {out_gif}")
    print(f"Info: {info_path}")


if __name__ == "__main__":
    main()
