import math
from typing import Iterable, Optional

import torch


def build_token_sample_ids(batch_size: int, tokens_per_sample: int, device: torch.device) -> torch.Tensor:
    if batch_size <= 0 or tokens_per_sample <= 0:
        raise ValueError("batch_size and tokens_per_sample must be positive.")
    return torch.arange(batch_size, device=device, dtype=torch.long).repeat_interleave(tokens_per_sample)


def build_same_sample_mask(
    x_sample_ids: Optional[torch.Tensor],
    y_sample_ids: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    if x_sample_ids is None or y_sample_ids is None:
        return None
    if x_sample_ids.dim() != 1 or y_sample_ids.dim() != 1:
        raise ValueError("sample id tensors must be rank-1.")
    if x_sample_ids.device != y_sample_ids.device:
        raise ValueError("sample id tensors must be on the same device.")
    return x_sample_ids[:, None] == y_sample_ids[None, :]


def _mean_pairwise_l2(samples: torch.Tensor) -> torch.Tensor:
    if samples.shape[0] < 2:
        return samples.new_tensor(1.0)
    distances = torch.pdist(samples, p=2)
    if distances.numel() == 0:
        return samples.new_tensor(1.0)
    return distances.mean()


@torch.no_grad()
def compute_V(
    x: torch.Tensor,
    y_pos: torch.Tensor,
    y_neg: torch.Tensor,
    temperatures: Iterable[float] = (0.02, 0.05, 0.2),
    x_sample_ids: Optional[torch.Tensor] = None,
    y_pos_sample_ids: Optional[torch.Tensor] = None,
    y_neg_sample_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if x.dim() != 2 or y_pos.dim() != 2 or y_neg.dim() != 2:
        raise ValueError("x, y_pos, y_neg must have shape [N, D].")
    if x.shape[1] != y_pos.shape[1] or x.shape[1] != y_neg.shape[1]:
        raise ValueError("x, y_pos, y_neg must have the same feature dimension.")
    if not temperatures:
        raise ValueError("temperatures must be non-empty.")

    x = x.float()
    y_pos = y_pos.float()
    y_neg = y_neg.float()

    n, d = x.shape
    sqrt_d = math.sqrt(d)
    v_total = torch.zeros_like(x)
    scale = x.new_tensor(1.0)
    pos_mask = build_same_sample_mask(x_sample_ids, y_pos_sample_ids)
    neg_mask = build_same_sample_mask(x_sample_ids, y_neg_sample_ids)

    for temperature in temperatures:
        all_samples = torch.cat([x, y_pos, y_neg], dim=0)
        scale = (_mean_pairwise_l2(all_samples) / sqrt_d).detach().clamp(min=1e-8)

        x_norm = x / scale
        y_pos_norm = y_pos / scale
        y_neg_norm = y_neg / scale
        t_eff = float(temperature) * sqrt_d

        dist_pos = torch.cdist(x_norm, y_pos_norm, p=2)
        dist_neg = torch.cdist(x_norm, y_neg_norm, p=2)

        if pos_mask is not None:
            dist_pos = dist_pos + (~pos_mask).to(dist_pos.dtype) * 1e6
        if neg_mask is not None:
            dist_neg = dist_neg + neg_mask.to(dist_neg.dtype) * 1e6
        elif n == y_neg.shape[0] and x.data_ptr() == y_neg.data_ptr():
            dist_neg = dist_neg + torch.eye(n, device=dist_neg.device, dtype=dist_neg.dtype) * 1e6

        logit_pos = -dist_pos / t_eff
        logit_neg = -dist_neg / t_eff
        logits = torch.cat([logit_pos, logit_neg], dim=1)

        a_row = torch.softmax(logits, dim=1)
        a_col = torch.softmax(logits, dim=0)
        a_row = torch.nan_to_num(a_row, nan=0.0, posinf=0.0, neginf=0.0)
        a_col = torch.nan_to_num(a_col, nan=0.0, posinf=0.0, neginf=0.0)
        a = torch.sqrt(a_row * a_col)

        m = y_pos.shape[0]
        a_pos = a[:, :m]
        a_neg = a[:, m:]

        w_pos = a_pos * a_neg.sum(dim=1, keepdim=True)
        w_neg = a_neg * a_pos.sum(dim=1, keepdim=True)

        drift_pos = w_pos @ y_pos_norm
        drift_neg = w_neg @ y_neg_norm
        v_tau = drift_pos - drift_neg

        lambda_tau = torch.sqrt(torch.mean(v_tau.pow(2).sum(dim=-1) / d)).detach().clamp(min=1e-8)
        v_total = v_total + (v_tau / lambda_tau)

    return torch.nan_to_num(v_total * scale, nan=0.0, posinf=0.0, neginf=0.0)
