import argparse
import glob
import json
import os.path as osp
import random

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.decomposition import IncrementalPCA, PCA
from torchvision import transforms
from tqdm import tqdm
import einops

from src.data import _iter_json_array

def _canonical_feature_extractor(name):
    if name == "dinov2":
        return "dino"
    return name


def _build_transform(feature_extractor, img_size):
    feature_extractor = _canonical_feature_extractor(feature_extractor)
    if feature_extractor in ["dino", "sam"]:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    elif feature_extractor == "eva2-clip":
        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)
    else:
        raise ValueError(f"Unknown feature extractor: {feature_extractor}")
    return transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


class cityscapes_sequence_data(torch.utils.data.Dataset):
    def __init__(self, root_dir, subset="train", img_size=(448, 896), feature_extractor="dinov2"):
        self.root_dir = root_dir
        self.transform = _build_transform(feature_extractor, img_size)
        self.files = glob.glob(osp.join(self.root_dir, subset, "**", "*.png"))
        self.files.sort()
        print(f"Found {len(self.files)} files")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.files[idx]
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        return image


def _iter_lang_entries(lang_root, split, lang_cache=None):
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


def _clip_dir_from_entry(entry, opendv_root):
    folder = entry.get("folder")
    if not folder:
        return None
    folder_norm = folder.replace("\\", "/").lstrip("/")
    return osp.join(opendv_root, folder_norm)


def gather_opendv_frames(
    opendv_root,
    lang_root,
    lang_cache,
    split,
    filter_folder,
    max_frames,
    seed,
    max_clips=None,
    clip_stride=1,
    clip_sample=None,
    clip_sample_mode="uniform",
    check_frames=True,
):
    if not lang_cache and not lang_root:
        raise ValueError("Provide --opendv_lang_root or --opendv_lang_cache for OpenDV.")
    rng = random.Random(seed)
    frames = []
    seen = set()
    total = 0

    def maybe_add(frame_path):
        nonlocal total
        if max_frames is None:
            frames.append(frame_path)
            return
        total += 1
        if len(frames) < max_frames:
            frames.append(frame_path)
            return
        j = rng.randint(0, total - 1)
        if j < max_frames:
            frames[j] = frame_path

    entries = _iter_lang_entries(lang_root, split, lang_cache=lang_cache)
    entries = tqdm(entries, desc=f"Indexing OpenDV clips ({split})")
    clip_count = 0
    for entry in entries:
        folder = entry.get("folder")
        if not folder:
            continue
        if filter_folder and filter_folder not in folder:
            continue
        clip_dir = _clip_dir_from_entry(entry, opendv_root)
        if not clip_dir or not osp.isdir(clip_dir):
            continue
        first_frame = entry.get("first_frame")
        last_frame = entry.get("last_frame")
        if not first_frame or not last_frame:
            continue
        if max_clips and clip_count >= max_clips:
            break
        start_id = int(osp.splitext(first_frame)[0])
        end_id = int(osp.splitext(last_frame)[0])
        pad = len(osp.splitext(first_frame)[0])
        ext = osp.splitext(first_frame)[1]
        if end_id < start_id:
            continue
        frame_ids = list(range(start_id, end_id + 1, max(1, int(clip_stride))))
        if clip_sample:
            target = int(clip_sample)
            if target <= 0:
                continue
            if len(frame_ids) > target:
                if clip_sample_mode == "random":
                    frame_ids = rng.sample(frame_ids, target)
                else:
                    idxs = np.linspace(0, len(frame_ids) - 1, target)
                    frame_ids = [frame_ids[int(round(i))] for i in idxs]
        for idx in frame_ids:
            frame_name = f"{str(idx).zfill(pad)}{ext}"
            frame_path = osp.join(clip_dir, frame_name)
            if check_frames and not osp.isfile(frame_path):
                continue
            if frame_path in seen:
                continue
            seen.add(frame_path)
            maybe_add(frame_path)
            if max_frames and len(frames) >= max_frames and total >= max_frames:
                continue
        clip_count += 1
    if max_frames is not None:
        rng.shuffle(frames)
    return frames


class opendv_frame_data(torch.utils.data.Dataset):
    def __init__(self, frames, img_size=(448, 896), feature_extractor="dinov2"):
        self.frames = frames
        self.transform = _build_transform(feature_extractor, img_size)

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame_path = self.frames[idx]
        image = Image.open(frame_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

class dinov2(nn.Module):
    def __init__(self, dlayers=[2,5,8,11]):
        super(dinov2, self).__init__()
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg', pretrained=True)
        self.model = self.model.eval()
        self.model = self.model.to('cuda')
        self.dlayers = dlayers
    
    def forward(self, x):
        with torch.no_grad():
            x = self.model.get_intermediate_layers(x,self.dlayers, reshape=False)
            if len(self.dlayers) > 1:
                x = torch.cat(x,dim=-1)
            else:
                x = x[0]
        return x

class eva2_clip(nn.Module):
    def __init__(self, dlayers=[2,5,8,11], img_size=(448,896)):
        super(eva2_clip, self).__init__()
        self.model = timm.create_model(
            'eva02_base_patch14_448.mim_in22k_ft_in1k',
            pretrained=True,
            img_size=tuple(img_size),
        )
        self.model = self.model.eval()
        self.model = self.model.to('cuda')
        self.dlayers = dlayers
    
    def forward(self, x):
        with torch.no_grad():
            x = self.model.forward_intermediates(x, indices=self.dlayers, output_fmt  = 'NLC', norm=True, intermediates_only=True)
            x = torch.cat(x,dim=-1)
        return x

class sam(nn.Module):
    def __init__(self, dlayers=[2,5,8,11],img_size=(448,896)):
        super(sam, self).__init__()
        self.model =  timm.create_model('timm/samvit_base_patch16.sa1b', pretrained=True,  pretrained_cfg_overlay={'input_size': (3,img_size[0],img_size[1]),})
        self.model = self.model.eval()
        self.model = self.model.to('cuda')
        self.dlayers = dlayers
    
    def forward(self, x):
        with torch.no_grad():
            x = self.model.forward_intermediates(x, indices=self.dlayers, norm=False, intermediates_only=True)
            x = [einops.rearrange(f, 'b c h w -> b (h w) c') for f in x]
            x = torch.cat(x,dim=-1)
        return x

# Create a DataLoader for the dataset
def dataloader(args, split):
    if args.dataset == "cityscapes":
        dataset = cityscapes_sequence_data(
            root_dir=args.cityscapes_root,
            subset=split,
            img_size=tuple(args.img_size),
            feature_extractor=args.feature_extractor,
        )
    elif args.dataset == "opendv":
        if not args.opendv_root:
            raise ValueError("--opendv_root is required for OpenDV dataset.")
        lang_cache = args.opendv_lang_cache
        if lang_cache is None:
            if split == "train" and args.opendv_lang_cache_train:
                lang_cache = args.opendv_lang_cache_train
            elif split == "val" and args.opendv_lang_cache_val:
                lang_cache = args.opendv_lang_cache_val
        if lang_cache is None and args.opendv_lang_root:
            candidate = osp.join(args.opendv_lang_root, f"mini_{split}_cache.json")
            if osp.isfile(candidate):
                lang_cache = candidate
        frames = gather_opendv_frames(
            opendv_root=args.opendv_root,
            lang_root=args.opendv_lang_root,
            lang_cache=lang_cache,
            split=split,
            filter_folder=args.opendv_filter_folder,
            max_frames=args.opendv_max_frames,
            seed=args.seed,
            max_clips=args.opendv_max_clips,
            clip_stride=args.opendv_clip_stride,
            clip_sample=args.opendv_clip_sample,
            clip_sample_mode=args.opendv_clip_sample_mode,
            check_frames=not args.opendv_skip_frame_check,
        )
        if not frames:
            raise ValueError("No OpenDV frames found for PCA.")
        dataset = opendv_frame_data(
            frames=frames,
            img_size=tuple(args.img_size),
            feature_extractor=args.feature_extractor,
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=args.num_workers,
        pin_memory=True,
    )


def _update_running_stats(mean, m2, count, batch):
    batch = batch.astype(np.float64)
    batch_count = batch.shape[0]
    if batch_count == 0:
        return mean, m2, count
    batch_mean = batch.mean(axis=0)
    batch_var = batch.var(axis=0)
    if count == 0:
        mean = batch_mean
        m2 = batch_var * batch_count
        count = batch_count
        return mean, m2, count
    delta = batch_mean - mean
    total = count + batch_count
    mean = mean + delta * batch_count / total
    m2 = m2 + batch_var * batch_count + (delta ** 2) * count * batch_count / total
    count = total
    return mean, m2, count


def _compute_mean_std(loader, model, device, dtype, inspect_fn=None, inspect_stage="meanstd"):
    mean = None
    m2 = None
    count = 0
    for batch in tqdm(loader, desc="Computing mean/std"):
        batch = batch.to(device).to(dtype)
        with torch.no_grad():
            feats = model(batch)
        if inspect_fn is not None:
            inspect_fn(feats, inspect_stage)
        feats = feats.flatten(end_dim=-2).float().cpu().numpy()
        mean, m2, count = _update_running_stats(mean, m2, count, feats)
    if count == 0:
        raise ValueError("No samples available to compute mean/std.")
    std = np.sqrt(m2 / count)
    std[std == 0] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def _fit_incremental_pca(
    loader,
    model,
    device,
    dtype,
    mean,
    std,
    n_components,
    inspect_fn=None,
    inspect_stage="ipca",
):
    ipca = IncrementalPCA(n_components=n_components)
    buffer = []
    buffer_count = 0
    fitted = False
    for batch in tqdm(loader, desc="Fitting IncrementalPCA"):
        batch = batch.to(device).to(dtype)
        with torch.no_grad():
            feats = model(batch)
        if inspect_fn is not None:
            inspect_fn(feats, inspect_stage)
        feats = feats.flatten(end_dim=-2).float().cpu().numpy()
        feats = (feats - mean) / std
        buffer.append(feats)
        buffer_count += feats.shape[0]
        if buffer_count >= n_components:
            combined = np.concatenate(buffer, axis=0)
            ipca.partial_fit(combined)
            fitted = True
            buffer = []
            buffer_count = 0
    if buffer_count >= n_components:
        combined = np.concatenate(buffer, axis=0)
        ipca.partial_fit(combined)
        fitted = True
        buffer = []
        buffer_count = 0
    if not fitted:
        raise ValueError("Not enough samples to fit IncrementalPCA; reduce n_components or increase data.")
    return ipca


def _get_patch_size_from_model(model):
    patch_size = None
    if hasattr(model, "model") and hasattr(model.model, "patch_embed"):
        patch_size = getattr(model.model.patch_embed, "patch_size", None)
    if patch_size is None:
        return None, None
    if isinstance(patch_size, int):
        return patch_size, patch_size
    if isinstance(patch_size, (tuple, list)) and len(patch_size) == 2:
        return int(patch_size[0]), int(patch_size[1])
    return None, None

def parse_list(s, ): 
    return list(map(int, s.split(',')))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cityscapes", choices=["cityscapes", "opendv"])
    parser.add_argument(
        "--feature_extractor",
        type=str,
        default="dinov2",
        choices=["dinov2", "dino", "eva2-clip", "sam"],
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--shuffle", action="store_true", default=False)
    parser.add_argument("--n_components", type=int, default=1152)
    parser.add_argument("--dlayers", type=parse_list, default=[2, 5, 8, 11])
    parser.add_argument("--img_size", type=parse_list, default=[448, 896])
    parser.add_argument("--split", type=str, default="train", choices=["train", "val"])
    parser.add_argument("--eval_split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--skip_eval", action="store_true", default=False)
    parser.add_argument("--incremental", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--cityscapes_root", type=str, default="/storage/cityscapes/leftImg8bit")
    parser.add_argument("--opendv_root", type=str, default=None, help="Root containing full_images/val_images.")
    parser.add_argument("--opendv_lang_root", type=str, default=None)
    parser.add_argument("--opendv_lang_cache", type=str, default=None)
    parser.add_argument("--opendv_lang_cache_train", type=str, default=None)
    parser.add_argument("--opendv_lang_cache_val", type=str, default=None)
    parser.add_argument("--opendv_filter_folder", type=str, default=None)
    parser.add_argument("--opendv_max_clips", type=int, default=None)
    parser.add_argument("--opendv_max_frames", type=int, default=None)
    parser.add_argument("--opendv_clip_stride", type=int, default=1)
    parser.add_argument("--opendv_clip_sample", type=int, default=None)
    parser.add_argument(
        "--opendv_clip_sample_mode",
        type=str,
        default="uniform",
        choices=["uniform", "random"],
    )
    parser.add_argument(
        "--opendv_skip_frame_check",
        action="store_true",
        default=False,
        help="Skip per-frame file existence checks when sampling clips.",
    )
    parser.add_argument(
        "--inspect_tokens",
        action="store_true",
        default=False,
        help="Print token shape diagnostics once after model(batch).",
    )
    parser.add_argument(
        "--inspect_tokens_only",
        action="store_true",
        default=False,
        help="Run one batch for token inspection, then exit without PCA fitting/saving.",
    )
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    if args.inspect_tokens_only:
        args.inspect_tokens = True
    print(args)
    n_components = args.n_components
    dtype = torch.float32
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    feature_extractor = _canonical_feature_extractor(args.feature_extractor)
    if feature_extractor == 'dino':
        model  = dinov2(dlayers=args.dlayers).to(device).to(dtype)
    elif feature_extractor == 'eva2-clip':
        model = eva2_clip(dlayers=args.dlayers, img_size=args.img_size).to(device).to(dtype)
    elif feature_extractor == 'sam':
        model = sam(dlayers=args.dlayers, img_size=args.img_size).to(device).to(dtype)
    else:
        raise ValueError(f"feature extractor {args.feature_extractor} not supported")

    train_loader = dataloader(args, args.split)
    print(f'Number of batches: {len(train_loader)}')
    print("Starting feature extraction for PCA...")
    patch_h, patch_w = _get_patch_size_from_model(model)
    token_info_printed = [False]

    def inspect_tokens_once(x, stage):
        if token_info_printed[0] or not args.inspect_tokens:
            return
        if x.ndim != 3:
            print(f"[TokenInspect][{stage}] x.shape = {tuple(x.shape)} (expected [B, N, C])")
            token_info_printed[0] = True
            return
        _, n_tokens, _ = x.shape
        print(f"[TokenInspect][{stage}] x.shape = {tuple(x.shape)}")
        if patch_h is None or patch_w is None:
            print(f"[TokenInspect][{stage}] Patch size unavailable from model; cannot compute expected patch token count.")
            token_info_printed[0] = True
            return
        img_h, img_w = int(args.img_size[0]), int(args.img_size[1])
        grid_h = img_h // patch_h
        grid_w = img_w // patch_w
        patch_tokens = grid_h * grid_w
        print(
            f"[TokenInspect][{stage}] expected patch tokens = {grid_h} x {grid_w} = {patch_tokens} "
            f"(img={img_h}x{img_w}, patch={patch_h}x{patch_w})"
        )
        special_tokens = n_tokens - patch_tokens
        if special_tokens > 0:
            print(
                f"[TokenInspect][{stage}] N={n_tokens} > {patch_tokens}: includes {special_tokens} special token(s) "
                f"(e.g., CLS/register)."
            )
        elif special_tokens == 0:
            print(f"[TokenInspect][{stage}] N={n_tokens} == {patch_tokens}: patch tokens only.")
        else:
            print(
                f"[TokenInspect][{stage}] N={n_tokens} < {patch_tokens}: fewer than expected patch tokens "
                f"(check img_size/patch size/model output)."
            )
        token_info_printed[0] = True

    if args.inspect_tokens_only:
        for batch in tqdm(train_loader, desc="Inspecting tokens only"):
            batch = batch.to(device).to(dtype)
            with torch.no_grad():
                x = model(batch)
            inspect_tokens_once(x, "inspect-only")
            break
        if not token_info_printed[0]:
            raise ValueError("Token inspection requested, but no batch was available.")
        print("Token inspection only mode: skipping PCA fitting, checkpoint saving, and eval.")
        raise SystemExit(0)

    if args.incremental:
        mean, std = _compute_mean_std(
            train_loader,
            model,
            device,
            dtype,
            inspect_fn=inspect_tokens_once,
            inspect_stage="train-meanstd",
        )
        pca_model = _fit_incremental_pca(
            train_loader,
            model,
            device,
            dtype,
            mean,
            std,
            n_components,
            inspect_fn=inspect_tokens_once,
            inspect_stage="train-ipca",
        )
        print('IncrementalPCA fitted')
    else:
        f_list = []
        for batch in tqdm(train_loader, desc="Collecting features"):
            batch = batch.to(device).to(dtype)
            x = model(batch)
            inspect_tokens_once(x, "train-collect")
            f_list.append(x.flatten(end_dim=-2).float().cpu().numpy())

        f = np.concatenate(f_list)
        # Standardize the data
        mean = np.mean(f, axis=0)
        std = np.std(f, axis=0)
        std[std == 0] = 1.0
        f = (f - mean) / std
        print(f.shape)
        print('Fitting PCA')
        pca_model = PCA(n_components=n_components)
        pca_model.fit(f)
        print('PCA fitted')

    # Save the PCA model and mean/std in the same file
    checkpoint = {
        'pca_model': pca_model,
        'mean': mean,
        'std': std
    }
    if len(args.dlayers) > 1:
        torch.save(checkpoint, args.feature_extractor+'_pca_'+str(args.img_size[0])+'_l'+str(args.dlayers).replace(" ", "_")+'_'+str(n_components)+'.pth')
    else:
        torch.save(checkpoint, args.feature_extractor+'_pca_'+str(args.img_size[0])+'_l'+str(args.dlayers[0])+'_'+str(n_components)+'.pth')

    # np.save('pca_mean_448_768.npy', mean)
    # np.save('pca_std_448_768.npy', std)
    # torch.save(PCA, 'pca_model_448_ms_768.pth')
    # torch.save(PCA, 'pca_model_224.pth')
    
    # Test
    if not args.skip_eval:
        print('Testing PCA')
        if len(args.dlayers) > 1:
            checkpoint = torch.load(args.feature_extractor+'_pca_'+str(args.img_size[0])+'_l'+str(args.dlayers).replace(" ", "_")+'_'+str(n_components)+'.pth')
        else:
            checkpoint = torch.load(args.feature_extractor+'_pca_'+str(args.img_size[0])+'_l'+str(args.dlayers[0])+'_'+str(n_components)+'.pth')
        pca_loaded = checkpoint['pca_model']
        mean = checkpoint['mean']
        std = checkpoint['std']
        eval_loader = dataloader(args, args.eval_split)
        f_list = []
        for batch in tqdm(eval_loader, desc="Collecting eval features"):
            batch = batch.to(device).to(dtype)
            x = model(batch)
            inspect_tokens_once(x, "eval-collect")
            f_list.append(x.flatten(end_dim=-2).float().cpu().numpy())

        f = np.concatenate(f_list)
        print(f.shape)
        print('Standardizing')
        f = (f - mean) / std
        print('Transforming')
        f_pca = pca_loaded.transform(f)
        print(f_pca.shape)
        var = pca_loaded.explained_variance_ratio_
        print(f'Explained variance: {var.sum()}')
        print(np.sum(np.var(f_pca, axis=0)) / np.sum(np.var(f, axis=0)))
    
