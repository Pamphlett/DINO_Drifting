import argparse
import json
import math
import os
import os.path as osp
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
import numpy as np
import torch
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from src.data import _iter_json_array

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_float_list(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def iter_lang_entries(lang_root: Optional[str], split: str, lang_cache: Optional[str]) -> Iterable[dict]:
    if lang_cache:
        for entry in _iter_json_array(lang_cache):
            yield entry
        return
    if not lang_root:
        raise ValueError("Provide either --lang_cache or --lang_root.")
    if split == "val":
        ann_files = [osp.join(lang_root, "10hz_YouTube_val.json")]
    else:
        ann_files = [osp.join(lang_root, f"10hz_YouTube_train_split{i}.json") for i in range(10)]
    for ann_path in ann_files:
        if not osp.isfile(ann_path):
            continue
        for entry in _iter_json_array(ann_path):
            yield entry


def resolve_clip_dir(folder: str, opendv_root: str) -> Optional[str]:
    folder_norm = folder.replace("\\", "/").lstrip("/")
    candidates = [osp.join(opendv_root, folder_norm), osp.join(osp.dirname(opendv_root), folder_norm)]
    root_base = osp.basename(opendv_root.rstrip("/"))
    if root_base in ("full_images", "val_images"):
        candidates.append(osp.join(osp.dirname(opendv_root.rstrip("/")), folder_norm))
    for path in candidates:
        if osp.isdir(path):
            return path
    return None


def pick_middle_existing_frame(clip_dir: str, first_frame: str, last_frame: str) -> Optional[str]:
    first_stem, ext = osp.splitext(first_frame)
    last_stem, _ = osp.splitext(last_frame)
    start_id = int(first_stem)
    end_id = int(last_stem)
    if end_id < start_id:
        return None
    pad = len(first_stem)
    mid = (start_id + end_id) // 2
    max_offset = end_id - start_id
    seen = set()
    for offset in range(max_offset + 1):
        for candidate_id in (mid - offset, mid + offset):
            if candidate_id in seen:
                continue
            seen.add(candidate_id)
            if candidate_id < start_id or candidate_id > end_id:
                continue
            frame_name = f"{str(candidate_id).zfill(pad)}{ext}"
            frame_path = osp.join(clip_dir, frame_name)
            if osp.isfile(frame_path):
                return frame_path
    return None


def sample_frame_items(
    opendv_root: str,
    lang_root: Optional[str],
    lang_cache: Optional[str],
    split: str,
    n_frames: int,
    seed: int,
) -> List[dict]:
    rng = random.Random(seed)
    sampled: List[dict] = []
    seen_clips = set()
    seen_frames = set()
    total_valid = 0

    for entry in tqdm(iter_lang_entries(lang_root, split, lang_cache), desc="Sampling clips"):
        folder = entry.get("folder")
        first_frame = entry.get("first_frame")
        last_frame = entry.get("last_frame")
        if not folder or not first_frame or not last_frame:
            continue
        clip_dir = resolve_clip_dir(folder, opendv_root)
        if clip_dir is None:
            continue
        clip_key = (clip_dir, first_frame, last_frame)
        if clip_key in seen_clips:
            continue
        seen_clips.add(clip_key)
        frame_path = pick_middle_existing_frame(clip_dir, first_frame, last_frame)
        if frame_path is None or frame_path in seen_frames:
            continue
        seen_frames.add(frame_path)
        total_valid += 1
        item = {"frame_path": frame_path, "clip_dir": clip_dir, "folder": folder}
        if len(sampled) < n_frames:
            sampled.append(item)
        else:
            j = rng.randint(0, total_valid - 1)
            if j < n_frames:
                sampled[j] = item

    rng.shuffle(sampled)
    print(f"Collected {len(sampled)} sampled frames from {total_valid} valid unique clips.")
    return sampled


def load_pca_metadata(pca_ckpt_path: str) -> Dict[str, torch.Tensor]:
    pca_dict = torch.load(pca_ckpt_path, map_location="cpu", weights_only=False)
    pca_model = pca_dict["pca_model"]
    mean = torch.as_tensor(pca_dict["mean"], dtype=torch.float32).flatten()
    std = torch.as_tensor(pca_dict["std"], dtype=torch.float32).flatten()
    pca_mean = torch.as_tensor(pca_model.mean_, dtype=torch.float32).flatten()
    pca_components = torch.as_tensor(pca_model.components_, dtype=torch.float32)
    if mean.numel() != pca_components.shape[1]:
        raise ValueError("PCA checkpoint normalization mean has incompatible dimension.")
    if std.numel() != pca_components.shape[1]:
        raise ValueError("PCA checkpoint normalization std has incompatible dimension.")
    if pca_mean.numel() != pca_components.shape[1]:
        raise ValueError("PCA model mean has incompatible dimension.")
    return {
        "mean": mean,
        "std": std,
        "pca_mean": pca_mean,
        "pca_components": pca_components,
    }


def pca_transform_tokens(raw_tokens: torch.Tensor, pca_meta: Dict[str, torch.Tensor]) -> torch.Tensor:
    x = raw_tokens.float()
    x = (x - pca_meta["mean"]) / pca_meta["std"]
    x = x - pca_meta["pca_mean"]
    x = torch.matmul(x, pca_meta["pca_components"].T)
    return x


def _feature_candidates(
    frame_path: str,
    opendv_root: str,
    feat_ext: str,
    feature_roots: Sequence[str],
) -> List[str]:
    frame_noext = osp.splitext(frame_path)[0]
    candidates = [frame_noext + feat_ext]
    for root in feature_roots:
        if not root:
            continue
        try:
            rel = osp.relpath(frame_path, opendv_root)
        except ValueError:
            continue
        if rel.startswith(".."):
            continue
        candidates.append(osp.splitext(osp.join(root, rel))[0] + feat_ext)
    deduped = []
    seen = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        deduped.append(path)
    return deduped


def load_precomputed_tokens(
    frame_path: str,
    opendv_root: str,
    feat_ext: str,
    feature_roots: Sequence[str],
) -> Tuple[Optional[torch.Tensor], Optional[str]]:
    feat_path = None
    for path in _feature_candidates(frame_path, opendv_root, feat_ext, feature_roots):
        if osp.isfile(path):
            feat_path = path
            break
    if feat_path is None:
        return None, None
    payload = torch.load(feat_path, map_location="cpu", weights_only=False)
    if "features" not in payload:
        return None, feat_path
    feats = payload["features"]
    if not torch.is_tensor(feats):
        feats = torch.as_tensor(feats)
    if feats.dim() == 3:
        feats = feats.reshape(-1, feats.shape[-1])
    elif feats.dim() != 2:
        return None, feat_path
    return feats.float(), feat_path


class FrameDataset(Dataset):
    def __init__(self, frame_paths: Sequence[str], img_size: Tuple[int, int]):
        self.frame_paths = list(frame_paths)
        self.transform = transforms.Compose(
            [
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    def __len__(self) -> int:
        return len(self.frame_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        frame_path = self.frame_paths[idx]
        image = Image.open(frame_path).convert("RGB")
        return self.transform(image), frame_path


def build_dino_model(variant: str, device: torch.device, local_repo: Optional[str] = None):
    model_name = "dinov2_" + variant
    if local_repo:
        model = torch.hub.load(local_repo, model_name, source="local", pretrained=True)
    else:
        model = torch.hub.load("facebookresearch/dinov2", model_name, pretrained=True)
    model = model.eval().to(device)
    return model


@torch.no_grad()
def extract_tokens_from_images(
    frame_paths: Sequence[str],
    img_size: Tuple[int, int],
    d_layers: Sequence[int],
    dinov2_variant: str,
    dinov2_local_repo: Optional[str],
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    if len(frame_paths) == 0:
        return {}
    model = build_dino_model(dinov2_variant, device, local_repo=dinov2_local_repo)
    dataset = FrameDataset(frame_paths, img_size=img_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        shuffle=False,
    )
    output: Dict[str, torch.Tensor] = {}
    for batch, paths in tqdm(loader, desc="Extracting DINOv2 features"):
        batch = batch.to(device, non_blocking=True)
        feats = model.get_intermediate_layers(batch, n=list(d_layers), reshape=False)
        if len(d_layers) > 1:
            feats = torch.cat(feats, dim=-1)
        else:
            feats = feats[0]
        feats = feats.float().cpu()
        for i, path in enumerate(paths):
            output[path] = feats[i]
    return output


def collect_feature_vectors(
    frame_items: Sequence[dict],
    pca_meta: Dict[str, torch.Tensor],
    opendv_root: str,
    feat_ext: str,
    feature_roots: Sequence[str],
    use_precomputed: bool,
    allow_extraction_fallback: bool,
    img_size: Tuple[int, int],
    d_layers: Sequence[int],
    dinov2_variant: str,
    dinov2_local_repo: Optional[str],
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
    frame_paths = [item["frame_path"] for item in frame_items]
    tokens_map: Dict[str, torch.Tensor] = {}
    missing: List[str] = []

    for frame_path in tqdm(frame_paths, desc="Loading precomputed tokens"):
        if use_precomputed:
            tokens, _ = load_precomputed_tokens(
                frame_path=frame_path,
                opendv_root=opendv_root,
                feat_ext=feat_ext,
                feature_roots=feature_roots,
            )
            if tokens is not None:
                tokens_map[frame_path] = tokens
                continue
        missing.append(frame_path)

    if missing:
        if use_precomputed and not allow_extraction_fallback:
            example = missing[0]
            searched = _feature_candidates(example, opendv_root, feat_ext, feature_roots)
            msg = [
                "Missing precomputed features and extraction fallback is disabled.",
                f"Missing: {len(missing)}/{len(frame_paths)} frames.",
                "Searched paths for first missing frame:",
            ]
            msg.extend([f"  - {p}" for p in searched[:6]])
            msg.append("Use --feature_roots and/or --feat_ext to point to feature files,")
            msg.append("or pass --allow_extraction_fallback (and optionally --dinov2_local_repo) to extract on the fly.")
            raise RuntimeError("\n".join(msg))
        print(f"Need extraction fallback for {len(missing)} frames.")
        extracted = extract_tokens_from_images(
            missing,
            img_size=img_size,
            d_layers=d_layers,
            dinov2_variant=dinov2_variant,
            dinov2_local_repo=dinov2_local_repo,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
        )
        tokens_map.update(extracted)

    kept_paths: List[str] = []
    raw_vecs: List[torch.Tensor] = []
    pca_vecs: List[torch.Tensor] = []
    expected_raw_dim = pca_meta["pca_components"].shape[1]

    for frame_path in tqdm(frame_paths, desc="Projecting PCA features"):
        tokens = tokens_map.get(frame_path)
        if tokens is None:
            continue
        if tokens.dim() != 2 or tokens.shape[1] != expected_raw_dim:
            continue
        raw_mean = tokens.mean(dim=0)
        pca_tokens = pca_transform_tokens(tokens, pca_meta)
        pca_mean = pca_tokens.mean(dim=0)
        kept_paths.append(frame_path)
        raw_vecs.append(raw_mean)
        pca_vecs.append(pca_mean)

    if not kept_paths:
        raise RuntimeError("No valid feature vectors were produced.")

    raw_tensor = torch.stack(raw_vecs, dim=0).float()
    pca_tensor = torch.stack(pca_vecs, dim=0).float()
    return kept_paths, raw_tensor, pca_tensor


def compute_pca2_features(features_1152: torch.Tensor, pca2_dim: int, seed: int) -> Tuple[np.ndarray, PCA]:
    feat_np = features_1152.cpu().numpy()
    n_samples, n_dim = feat_np.shape
    n_components = min(pca2_dim, n_samples, n_dim)
    if n_components < 2:
        raise ValueError("Need at least 2 components/samples for PCA2 analysis.")
    pca2 = PCA(n_components=n_components, random_state=seed)
    feat_64 = pca2.fit_transform(feat_np)
    return feat_64.astype(np.float32), pca2


def cluster_features(features_64: np.ndarray, n_clusters: int, seed: int) -> np.ndarray:
    n_samples = features_64.shape[0]
    k = min(n_clusters, n_samples)
    if k < 2:
        raise ValueError("Need at least 2 samples/clusters for clustering.")
    model = KMeans(n_clusters=k, n_init=20, random_state=seed)
    return model.fit_predict(features_64)


def save_cluster_samples(
    frame_paths: Sequence[str],
    labels: np.ndarray,
    output_dir: str,
    n_clusters: int,
    seed: int,
    samples_per_cluster: int = 5,
) -> None:
    ensure_dir(output_dir)
    rng = random.Random(seed)
    frame_paths = list(frame_paths)
    labels = np.asarray(labels)
    for cluster_id in range(n_clusters):
        idxs = np.where(labels == cluster_id)[0].tolist()
        if len(idxs) == 0:
            fig, ax = plt.subplots(1, 1, figsize=(5, 3))
            ax.axis("off")
            ax.set_title(f"cluster {cluster_id}: empty")
            fig.savefig(osp.join(output_dir, f"cluster_{cluster_id}.png"), dpi=140, bbox_inches="tight")
            plt.close(fig)
            continue
        n_pick = min(samples_per_cluster, len(idxs))
        picked = rng.sample(idxs, n_pick)
        fig, axes = plt.subplots(1, samples_per_cluster, figsize=(3 * samples_per_cluster, 3))
        if samples_per_cluster == 1:
            axes = [axes]
        for col in range(samples_per_cluster):
            ax = axes[col]
            ax.axis("off")
            if col >= n_pick:
                continue
            path = frame_paths[picked[col]]
            img = Image.open(path).convert("RGB")
            ax.imshow(np.asarray(img))
            ax.set_title(osp.basename(path), fontsize=8)
        fig.suptitle(f"Cluster {cluster_id} (n={len(idxs)})")
        fig.savefig(osp.join(output_dir, f"cluster_{cluster_id}.png"), dpi=140, bbox_inches="tight")
        plt.close(fig)


@dataclass
class DistanceStats:
    intra_mean: float
    intra_std: float
    inter_mean: float
    inter_std: float
    separation_ratio: float
    cov: float
    intra_hist: np.ndarray
    inter_hist: np.ndarray
    all_pairwise: np.ndarray


def compute_distance_stats(features: torch.Tensor, labels: np.ndarray, device: torch.device) -> DistanceStats:
    x = features.to(device=device, dtype=torch.float32)
    n = x.shape[0]
    labels_t = torch.as_tensor(labels, device=device, dtype=torch.long)
    with torch.no_grad():
        dmat = torch.cdist(x, x, p=2)
        tri = torch.triu(torch.ones((n, n), device=device, dtype=torch.bool), diagonal=1)
        same = labels_t[:, None] == labels_t[None, :]
        intra = dmat[tri & same]
        inter = dmat[tri & (~same)]
        all_pairwise = dmat[tri]

    def _safe_mean(t: torch.Tensor) -> float:
        return float(t.mean().item()) if t.numel() > 0 else float("nan")

    def _safe_std(t: torch.Tensor) -> float:
        return float(t.std(unbiased=False).item()) if t.numel() > 0 else float("nan")

    intra_mean = _safe_mean(intra)
    inter_mean = _safe_mean(inter)
    sep = inter_mean / max(intra_mean, 1e-12) if np.isfinite(intra_mean) and np.isfinite(inter_mean) else float("nan")
    all_mean = _safe_mean(all_pairwise)
    all_std = _safe_std(all_pairwise)
    cov = all_std / max(all_mean, 1e-12) if np.isfinite(all_std) and np.isfinite(all_mean) else float("nan")

    return DistanceStats(
        intra_mean=intra_mean,
        intra_std=_safe_std(intra),
        inter_mean=inter_mean,
        inter_std=_safe_std(inter),
        separation_ratio=sep,
        cov=cov,
        intra_hist=intra.detach().cpu().numpy(),
        inter_hist=inter.detach().cpu().numpy(),
        all_pairwise=all_pairwise.detach().cpu().numpy(),
    )


def simulate_kernel(
    features: torch.Tensor,
    labels: np.ndarray,
    taus: Sequence[float],
    n_queries: int,
    pool_size: int,
    seed: int,
    device: torch.device,
) -> Dict[float, Dict[str, List[float]]]:
    x = features.to(device=device, dtype=torch.float32)
    labels_t = torch.as_tensor(labels, device=device, dtype=torch.long)
    n, d = x.shape
    sqrt_d = math.sqrt(d)
    q = min(n_queries, n)
    pool_size = min(pool_size, max(1, n - 1))
    rng = random.Random(seed)
    if q < n:
        query_ids = rng.sample(range(n), q)
    else:
        query_ids = list(range(n))
        rng.shuffle(query_ids)

    stats: Dict[float, Dict[str, List[float]]] = {
        tau: {"entropy": [], "max_affinity": [], "same_sum": [], "diff_sum": [], "same_diff_ratio": []}
        for tau in taus
    }

    all_indices = list(range(n))
    for qid in tqdm(query_ids, desc=f"Kernel simulation (D={d})"):
        query = x[qid]
        query_label = labels_t[qid]
        candidates = all_indices[:qid] + all_indices[qid + 1 :]
        if len(candidates) <= pool_size:
            pool_ids = candidates
        else:
            pool_ids = rng.sample(candidates, pool_size)
        pool = x[pool_ids]
        pool_labels = labels_t[pool_ids]

        all_samples = torch.cat([query.unsqueeze(0), pool], dim=0)
        scale = torch.pdist(all_samples, p=2).mean() / sqrt_d
        scale = scale.detach().clamp(min=1e-8)

        query_norm = query / scale
        pool_norm = pool / scale
        dists = torch.cdist(query_norm.unsqueeze(0), pool_norm, p=2).squeeze(0)

        same_mask = pool_labels == query_label
        diff_mask = ~same_mask
        for tau in taus:
            t_eff = float(tau) * sqrt_d
            logits = -dists / t_eff
            aff = torch.softmax(logits, dim=0)
            entropy = -(aff * torch.log(aff.clamp(min=1e-12))).sum()
            max_aff = aff.max()
            same_sum = aff[same_mask].sum() if same_mask.any() else aff.new_tensor(0.0)
            diff_sum = aff[diff_mask].sum() if diff_mask.any() else aff.new_tensor(0.0)
            ratio = same_sum / diff_sum.clamp(min=1e-12)

            stats[tau]["entropy"].append(float(entropy.item()))
            stats[tau]["max_affinity"].append(float(max_aff.item()))
            stats[tau]["same_sum"].append(float(same_sum.item()))
            stats[tau]["diff_sum"].append(float(diff_sum.item()))
            stats[tau]["same_diff_ratio"].append(float(ratio.item()))

    return stats


def summarize_kernel(kernel_stats: Dict[float, Dict[str, List[float]]]) -> Dict[float, Dict[str, float]]:
    summary: Dict[float, Dict[str, float]] = {}
    for tau, values in kernel_stats.items():
        summary[tau] = {
            "entropy_mean": float(np.mean(values["entropy"])),
            "entropy_std": float(np.std(values["entropy"])),
            "max_affinity_mean": float(np.mean(values["max_affinity"])),
            "same_mean": float(np.mean(values["same_sum"])),
            "diff_mean": float(np.mean(values["diff_sum"])),
            "same_diff_ratio_mean": float(np.mean(values["same_diff_ratio"])),
        }
    return summary


def compute_2d_embedding(features_64: np.ndarray, seed: int) -> Tuple[np.ndarray, str]:
    try:
        import umap.umap_ as umap

        reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1, random_state=seed)
        emb = reducer.fit_transform(features_64)
        return emb, "UMAP"
    except Exception:
        n = features_64.shape[0]
        perplexity = min(30, max(5, (n - 1) // 3))
        reducer = TSNE(n_components=2, random_state=seed, init="pca", learning_rate="auto", perplexity=perplexity)
        emb = reducer.fit_transform(features_64)
        return emb, "t-SNE"


def plot_results(
    dist_stats: Dict[str, DistanceStats],
    kernel_stats: Dict[str, Dict[float, Dict[str, List[float]]]],
    kernel_summary: Dict[str, Dict[float, Dict[str, float]]],
    embedding_2d: np.ndarray,
    labels: np.ndarray,
    taus: Sequence[float],
    pool_size: int,
    output_path: str,
) -> None:
    space_order = ["raw3072", "pca1152", "pca64"]
    space_title = {
        "raw3072": "Raw 3072-dim",
        "pca1152": "PCA 1152-dim",
        "pca64": "PCA2 64-dim",
    }

    fig = plt.figure(figsize=(18, 20))
    gs = fig.add_gridspec(4, 3, height_ratios=[1.0, 1.2, 1.0, 1.0])

    for i, space in enumerate(space_order):
        ax = fig.add_subplot(gs[0, i])
        ds = dist_stats[space]
        ax.hist(ds.inter_hist, bins=60, alpha=0.5, color="red", label="Inter")
        ax.hist(ds.intra_hist, bins=60, alpha=0.5, color="blue", label="Intra")
        ax.set_title(
            f"{space_title[space]}\nsep={ds.separation_ratio:.3f}, CoV={100.0 * ds.cov:.2f}%",
            fontsize=11,
        )
        ax.set_xlabel("L2 distance")
        ax.set_ylabel("Count")
        ax.legend()

    ax_emb = fig.add_subplot(gs[1, :])
    scatter = ax_emb.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=labels, cmap="tab10", s=12, alpha=0.85)
    ax_emb.set_title("64-dim embedding colored by K-Means cluster")
    ax_emb.set_xlabel("Dim-1")
    ax_emb.set_ylabel("Dim-2")
    cbar = fig.colorbar(scatter, ax=ax_emb, fraction=0.02, pad=0.01)
    cbar.set_label("Cluster")

    max_entropy = math.log(pool_size)
    tau_labels = [f"{t:g}" for t in taus]
    x_pos = np.arange(len(taus))
    width = 0.38

    for i, space in enumerate(space_order):
        ax = fig.add_subplot(gs[2, i])
        data = [kernel_stats[space][tau]["entropy"] for tau in taus]
        ax.boxplot(data, labels=tau_labels, showfliers=False)
        ax.axhline(max_entropy, color="black", linestyle="--", linewidth=1.0, label=f"log({pool_size})")
        ax.set_title(f"Kernel entropy ({space_title[space]})")
        ax.set_xlabel("Temperature")
        ax.set_ylabel("Entropy")
        ax.legend()

    for i, space in enumerate(space_order):
        ax = fig.add_subplot(gs[3, i])
        same_means = [kernel_summary[space][tau]["same_mean"] for tau in taus]
        diff_means = [kernel_summary[space][tau]["diff_mean"] for tau in taus]
        ax.bar(x_pos - width / 2, same_means, width=width, color="green", alpha=0.8, label="Same cluster")
        ax.bar(x_pos + width / 2, diff_means, width=width, color="orange", alpha=0.8, label="Diff cluster")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(tau_labels)
        ax.set_title(f"Same vs diff affinity ({space_title[space]})")
        ax.set_xlabel("Temperature")
        ax.set_ylabel("Affinity sum")
        ax.set_ylim(0.0, 1.0)
        ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def verdict(dist: DistanceStats, kernel: Dict[float, Dict[str, float]]) -> str:
    sep_ok = dist.separation_ratio > 1.3
    kernel_ok = any(v["same_diff_ratio_mean"] > 1.5 for v in kernel.values())
    return "CAN" if (sep_ok and kernel_ok) else "CANNOT"


def make_summary_text(
    n_frames: int,
    n_clusters: int,
    dist_stats: Dict[str, DistanceStats],
    kernel_summary: Dict[str, Dict[float, Dict[str, float]]],
    taus: Sequence[float],
    pool_size: int,
) -> str:
    max_entropy = math.log(pool_size)
    raw = dist_stats["raw3072"]
    pca1152 = dist_stats["pca1152"]
    pca64 = dist_stats["pca64"]
    lines = []
    lines.append("=== CLUSTER STRUCTURE ANALYSIS ===")
    lines.append(f"N_frames: {n_frames}, N_clusters: {n_clusters}")
    lines.append("")
    lines.append("[Raw 3072-dim]")
    lines.append(f"  Intra-cluster dist: mean={raw.intra_mean:.4f}, std={raw.intra_std:.4f}")
    lines.append(f"  Inter-cluster dist: mean={raw.inter_mean:.4f}, std={raw.inter_std:.4f}")
    lines.append(f"  Separation ratio: {raw.separation_ratio:.4f}")
    lines.append(f"  Distance CoV: {100.0 * raw.cov:.2f}%")
    lines.append("")
    lines.append("[PCA 1152-dim - actual drifting space]")
    lines.append(f"  Intra-cluster dist: mean={pca1152.intra_mean:.4f}, std={pca1152.intra_std:.4f}")
    lines.append(f"  Inter-cluster dist: mean={pca1152.inter_mean:.4f}, std={pca1152.inter_std:.4f}")
    lines.append(f"  Separation ratio: {pca1152.separation_ratio:.4f}")
    lines.append(f"  Distance CoV: {100.0 * pca1152.cov:.2f}%")
    lines.append("")
    lines.append("[PCA2 64-dim - further reduced]")
    lines.append(f"  Intra-cluster dist: mean={pca64.intra_mean:.4f}, std={pca64.intra_std:.4f}")
    lines.append(f"  Inter-cluster dist: mean={pca64.inter_mean:.4f}, std={pca64.inter_std:.4f}")
    lines.append(f"  Separation ratio: {pca64.separation_ratio:.4f}")
    lines.append(f"  Distance CoV: {100.0 * pca64.cov:.2f}%")
    lines.append("")
    lines.append("[Kernel Simulation - PCA 1152-dim (actual drifting space)]")
    for tau in taus:
        v = kernel_summary["pca1152"][tau]
        lines.append(
            f"  tau={tau:g}: entropy={v['entropy_mean']:.4f}/{max_entropy:.4f}(max), "
            f"same_cluster={v['same_mean']:.4f}, diff_cluster={v['diff_mean']:.4f}, "
            f"ratio={v['same_diff_ratio_mean']:.4f}"
        )
    lines.append("")
    lines.append("[Kernel Simulation - PCA2 64-dim]")
    for tau in taus:
        v = kernel_summary["pca64"][tau]
        lines.append(
            f"  tau={tau:g}: entropy={v['entropy_mean']:.4f}/{max_entropy:.4f}(max), "
            f"same_cluster={v['same_mean']:.4f}, diff_cluster={v['diff_mean']:.4f}, "
            f"ratio={v['same_diff_ratio_mean']:.4f}"
        )
    lines.append("")
    lines.append(
        f"VERDICT: [Kernel {verdict(pca1152, kernel_summary['pca1152'])} distinguish driving scene types in 1152-dim PCA space]"
    )
    lines.append(
        f"VERDICT: [Kernel {verdict(pca64, kernel_summary['pca64'])} distinguish driving scene types in 64-dim PCA2 space]"
    )
    return "\n".join(lines)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze scene cluster structure in DINO feature spaces.")
    parser.add_argument("--opendv_root", required=True, type=str)
    parser.add_argument("--lang_root", default=None, type=str)
    parser.add_argument("--lang_cache", default=None, type=str)
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--pca_ckpt", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)

    parser.add_argument("--n_frames", type=int, default=2000)
    parser.add_argument("--n_clusters", type=int, default=10)
    parser.add_argument("--pca2_dim", type=int, default=64)
    parser.add_argument("--dinov2_variant", type=str, default="vitb14_reg")
    parser.add_argument("--d_layers", type=parse_int_list, default=[2, 5, 8, 11])
    parser.add_argument("--img_size", type=parse_int_list, default=[224, 448])

    parser.add_argument("--use_precomputed", action="store_true", default=False)
    parser.add_argument("--feat_ext", type=str, default=".dinov2.pt")
    parser.add_argument(
        "--feature_roots",
        type=str,
        default="",
        help="Comma-separated alternate roots to search for precomputed features by relpath from --opendv_root.",
    )
    parser.add_argument(
        "--allow_extraction_fallback",
        action="store_true",
        default=False,
        help="If set, extract missing features with DINO when precomputed features are absent.",
    )
    parser.add_argument(
        "--dinov2_local_repo",
        type=str,
        default=None,
        help="Optional local path to facebookresearch/dinov2 repo for offline torch.hub loading.",
    )
    parser.add_argument("--save_features", action="store_true", default=False)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_queries", type=int, default=100)
    parser.add_argument("--pool_size", type=int, default=50)
    parser.add_argument("--taus", type=parse_float_list, default=[0.02, 0.05, 0.2])
    parser.add_argument("--cluster_samples_per_cluster", type=int, default=5)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if len(args.img_size) != 2:
        raise ValueError("--img_size must contain two integers, e.g. 224,448")
    if len(args.d_layers) == 0:
        raise ValueError("--d_layers must be non-empty")
    if len(args.taus) == 0:
        raise ValueError("--taus must be non-empty")
    if not args.feat_ext.startswith("."):
        raise ValueError("--feat_ext should start with '.' (e.g. .dinov2.pt)")

    feature_roots = [x.strip() for x in args.feature_roots.split(",") if x.strip()]

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ensure_dir(args.output_dir)
    cluster_samples_dir = osp.join(args.output_dir, "cluster_samples")
    ensure_dir(cluster_samples_dir)

    print("Step 1/6: sampling frames")
    frame_items = sample_frame_items(
        opendv_root=args.opendv_root,
        lang_root=args.lang_root,
        lang_cache=args.lang_cache,
        split=args.split,
        n_frames=args.n_frames,
        seed=args.seed,
    )
    if len(frame_items) < 10:
        raise RuntimeError("Too few sampled frames for analysis.")

    print("Step 2/6: loading or extracting features + PCA transform")
    pca_meta = load_pca_metadata(args.pca_ckpt)
    kept_paths, raw3072, pca1152 = collect_feature_vectors(
        frame_items=frame_items,
        pca_meta=pca_meta,
        opendv_root=args.opendv_root,
        feat_ext=args.feat_ext,
        feature_roots=feature_roots,
        use_precomputed=args.use_precomputed,
        allow_extraction_fallback=args.allow_extraction_fallback,
        img_size=(args.img_size[0], args.img_size[1]),
        d_layers=args.d_layers,
        dinov2_variant=args.dinov2_variant,
        dinov2_local_repo=args.dinov2_local_repo,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )
    print(f"Using {len(kept_paths)} frames after feature filtering.")

    print("Step 3/6: image-level vectors and PCA2")
    feat64_np, pca2_model = compute_pca2_features(pca1152, pca2_dim=args.pca2_dim, seed=args.seed)
    pca64 = torch.from_numpy(feat64_np)
    labels = cluster_features(feat64_np, n_clusters=args.n_clusters, seed=args.seed)
    n_clusters_eff = int(np.max(labels)) + 1
    save_cluster_samples(
        frame_paths=kept_paths,
        labels=labels,
        output_dir=cluster_samples_dir,
        n_clusters=n_clusters_eff,
        seed=args.seed,
        samples_per_cluster=args.cluster_samples_per_cluster,
    )

    print("Step 4/6: pairwise distance distributions")
    dist_stats = {
        "raw3072": compute_distance_stats(raw3072, labels, device=device),
        "pca1152": compute_distance_stats(pca1152, labels, device=device),
        "pca64": compute_distance_stats(pca64, labels, device=device),
    }

    print("Step 5/6: softmax kernel simulation")
    kernel_stats = {
        "raw3072": simulate_kernel(raw3072, labels, args.taus, args.n_queries, args.pool_size, args.seed, device=device),
        "pca1152": simulate_kernel(pca1152, labels, args.taus, args.n_queries, args.pool_size, args.seed, device=device),
        "pca64": simulate_kernel(pca64, labels, args.taus, args.n_queries, args.pool_size, args.seed, device=device),
    }
    kernel_summary = {space: summarize_kernel(stats) for space, stats in kernel_stats.items()}

    print("Step 6/6: visualizations and summary")
    embedding_2d, embed_name = compute_2d_embedding(feat64_np, seed=args.seed)
    plot_results(
        dist_stats=dist_stats,
        kernel_stats=kernel_stats,
        kernel_summary=kernel_summary,
        embedding_2d=embedding_2d,
        labels=labels,
        taus=args.taus,
        pool_size=min(args.pool_size, max(1, len(kept_paths) - 1)),
        output_path=osp.join(args.output_dir, "cluster_analysis.png"),
    )
    summary_txt = make_summary_text(
        n_frames=len(kept_paths),
        n_clusters=n_clusters_eff,
        dist_stats=dist_stats,
        kernel_summary=kernel_summary,
        taus=args.taus,
        pool_size=min(args.pool_size, max(1, len(kept_paths) - 1)),
    )
    print(summary_txt)
    with open(osp.join(args.output_dir, "summary.txt"), "w") as f:
        f.write(summary_txt + "\n")

    if args.save_features:
        cache_payload = {
            "frame_paths": kept_paths,
            "cluster_labels": labels.astype(np.int64),
            "raw3072": raw3072.cpu(),
            "pca1152": pca1152.cpu(),
            "pca64": pca64.cpu(),
            "pca2_components": pca2_model.components_,
            "pca2_mean": pca2_model.mean_,
            "embedding_2d_method": embed_name,
            "embedding_2d": embedding_2d,
        }
        torch.save(cache_payload, osp.join(args.output_dir, "features_cache.pt"))
        with open(osp.join(args.output_dir, "frame_list.json"), "w") as f:
            json.dump({"frame_paths": kept_paths, "cluster_labels": labels.tolist()}, f)


if __name__ == "__main__":
    main()
