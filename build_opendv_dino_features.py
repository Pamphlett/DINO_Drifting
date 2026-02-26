import argparse
import json
import os
import os.path as osp

import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import timm
import einops

from src.data import _iter_json_array


def parse_list(x):
    return list(map(int, x.split(",")))


def iter_lang_entries(lang_root, split, lang_cache=None):
    if lang_cache:
        for entry in json.load(open(lang_cache, "r")):
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
    return osp.join(opendv_root, folder_norm)


def gather_frame_items(args):
    frame_items = []
    seen_frames = set()
    entries = iter_lang_entries(args.lang_root, args.split, lang_cache=args.lang_cache)
    for entry in entries:
        folder = entry.get("folder")
        if not folder:
            continue
        if args.filter_folder and args.filter_folder not in folder:
            continue
        clip_dir = clip_dir_from_entry(entry, args.opendv_root)
        if not clip_dir or not osp.isdir(clip_dir):
            continue
        first_frame = entry.get("first_frame")
        last_frame = entry.get("last_frame")
        if not first_frame or not last_frame:
            continue
        start_id = int(osp.splitext(first_frame)[0])
        end_id = int(osp.splitext(last_frame)[0])
        pad = len(osp.splitext(first_frame)[0])
        ext = osp.splitext(first_frame)[1]
        for idx in range(start_id, end_id + 1):
            frame_name = f"{str(idx).zfill(pad)}{ext}"
            frame_path = osp.join(clip_dir, frame_name)
            if not osp.isfile(frame_path):
                continue
            feat_path = osp.splitext(frame_path)[0] + args.feat_ext
            if not args.overwrite and osp.isfile(feat_path):
                continue
            if frame_path in seen_frames:
                continue
            seen_frames.add(frame_path)
            frame_items.append((frame_path, feat_path))
            if args.max_frames and len(frame_items) >= args.max_frames:
                return frame_items
    return frame_items


class FrameDataset(torch.utils.data.Dataset):
    def __init__(self, items, img_size, feature_extractor):
        self.items = items
        if feature_extractor in ["dino", "sam"]:
            mean = (0.485, 0.456, 0.406)
            std = (0.229, 0.224, 0.225)
        elif feature_extractor == "eva2-clip":
            mean = (0.48145466, 0.4578275, 0.40821073)
            std = (0.26862954, 0.26130258, 0.27577711)
        else:
            raise ValueError(f"Unknown feature_extractor: {feature_extractor}")
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        frame_path, feat_path = self.items[idx]
        image = Image.open(frame_path).convert("RGB")
        return self.transform(image), feat_path


def build_model(args, device):
    if args.feature_extractor == "dino":
        model = torch.hub.load(
            "facebookresearch/dinov2",
            "dinov2_" + args.dinov2_variant,
            pretrained=True,
        )
        model.eval().to(device)
        return model
    if args.feature_extractor == "eva2-clip":
        model = timm.create_model(
            "eva02_base_patch14_448.mim_in22k_ft_in1k",
            pretrained=True,
            img_size=(args.img_size[0], args.img_size[1]),
        )
        model.eval().to(device)
        return model
    if args.feature_extractor == "sam":
        model = timm.create_model(
            "timm/samvit_base_patch16.sa1b",
            pretrained=True,
            pretrained_cfg_overlay={"input_size": (3, args.img_size[0], args.img_size[1])},
        )
        model.eval().to(device)
        return model
    raise ValueError(f"Unknown feature_extractor: {args.feature_extractor}")


def run_model(model, batch, args):
    if args.feature_extractor == "dino":
        feats = model.get_intermediate_layers(batch, n=args.d_layers, reshape=False)
        if len(args.d_layers) > 1:
            feats = torch.cat(feats, dim=-1)
        else:
            feats = feats[0]
        return feats
    if args.feature_extractor == "eva2-clip":
        feats = model.forward_intermediates(
            batch,
            indices=args.d_layers,
            output_fmt="NLC",
            norm=True,
            intermediates_only=True,
        )
        return torch.cat(feats, dim=-1)
    if args.feature_extractor == "sam":
        feats = model.forward_intermediates(
            batch,
            indices=args.d_layers,
            norm=False,
            intermediates_only=True,
        )
        feats = [einops.rearrange(f, "b c h w -> b (h w) c") for f in feats]
        return torch.cat(feats, dim=-1)
    raise ValueError(f"Unknown feature_extractor: {args.feature_extractor}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--opendv_root", required=True, help="Root containing full_images/val_images.")
    parser.add_argument("--lang_root", default=None, help="Path to OpenDV-YouTube-Language.")
    parser.add_argument("--lang_cache", default=None, help="Optional JSON cache of language entries.")
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--filter_folder", default=None)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--feat_ext", type=str, default=".dinov2.pt")
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--feature_extractor", type=str, default="dino", choices=["dino", "eva2-clip", "sam"])
    parser.add_argument("--dinov2_variant", type=str, default="vitb14_reg", choices=["vits14_reg", "vitb14_reg"])
    parser.add_argument("--d_layers", type=parse_list, default=[2, 5, 8, 11])
    parser.add_argument("--img_size", type=parse_list, default=[224, 448])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save_dtype", type=str, default="float16", choices=["float16", "float32"])
    args = parser.parse_args()

    if not args.lang_cache and not args.lang_root:
        raise ValueError("Provide --lang_cache or --lang_root.")

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    save_dtype = torch.float16 if args.save_dtype == "float16" else torch.float32

    items = gather_frame_items(args)
    if not items:
        print("No frames to process.")
        return

    dataset = FrameDataset(items, tuple(args.img_size), args.feature_extractor)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False,
    )
    model = build_model(args, device)

    if args.feature_extractor in ["dino", "eva2-clip"]:
        patch = 14
    else:
        patch = 16
    h = args.img_size[0] // patch
    w = args.img_size[1] // patch

    processed = 0
    for batch, feat_paths in tqdm(loader, desc="Extracting features"):
        batch = batch.to(device)
        with torch.no_grad():
            feats = run_model(model, batch, args)
        feats = feats.view(feats.shape[0], h, w, -1).to("cpu", dtype=save_dtype)
        for i, feat_path in enumerate(feat_paths):
            payload = {
                "features": feats[i],
                "feature_extractor": args.feature_extractor,
                "d_layers": args.d_layers,
                "img_size": args.img_size,
                "patch_size": patch,
            }
            os.makedirs(osp.dirname(feat_path), exist_ok=True)
            torch.save(payload, feat_path)
            processed += 1
    print(f"Saved {processed} feature files.")


if __name__ == "__main__":
    main()
