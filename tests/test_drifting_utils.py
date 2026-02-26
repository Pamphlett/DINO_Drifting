import unittest

import torch

from src.drifting_utils import build_token_sample_ids, compute_V


class TestDriftingUtils(unittest.TestCase):
    def test_antisymmetry(self):
        torch.manual_seed(0)
        x = torch.randn(64, 32)
        y_pos = torch.randn(64, 32)
        y_neg = torch.randn(64, 32)
        v_ab = compute_V(x=x, y_pos=y_pos, y_neg=y_neg, temperatures=(0.02, 0.05, 0.2))
        v_ba = compute_V(x=x, y_pos=y_neg, y_neg=y_pos, temperatures=(0.02, 0.05, 0.2))
        rel_err = (v_ab + v_ba).norm(dim=-1).mean() / v_ab.norm(dim=-1).mean().clamp(min=1e-8)
        self.assertLess(rel_err.item(), 1e-3)

    def test_zero_drift_when_x_close_to_pos_and_pos_equals_neg(self):
        torch.manual_seed(1)
        y = torch.randn(48, 24)
        x = y + 1e-4 * torch.randn(48, 24)
        v = compute_V(x=x, y_pos=y, y_neg=y, temperatures=(0.05,))
        self.assertLess(v.abs().max().item(), 1e-6)

    def test_block_mask_finite_output(self):
        torch.manual_seed(2)
        batch_size = 4
        tokens_per_sample = 16
        dim = 20
        x = torch.randn(batch_size * tokens_per_sample, dim)
        y_pos = torch.randn(batch_size * tokens_per_sample, dim)
        ids = build_token_sample_ids(batch_size, tokens_per_sample, x.device)
        v = compute_V(
            x=x,
            y_pos=y_pos,
            y_neg=x,
            temperatures=(0.02, 0.05),
            x_sample_ids=ids,
            y_neg_sample_ids=ids,
        )
        self.assertTrue(torch.isfinite(v).all().item())


if __name__ == "__main__":
    unittest.main()
