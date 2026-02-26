import argparse
from pathlib import Path
from datetime import timedelta

import numpy as np
from PIL import Image
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
import torchvision.transforms as T

from dpt import DPTHead


def parse_tuple(x):
    return tuple(map(int, x.split(",")))


def parse_list(x):
    return list(map(int, x.split(",")))


class FeatureRgbDataset(Dataset):
    def __init__(
        self,
        feature_root,
        rgb_root,
        img_size,
        feat_ext=".pt",
        rgb_ext=".png",
        feat_key="features",
        rgb_key="rgb_path",
        feat_time_index=None,
    ):
        self.feature_root = Path(feature_root)
        self.rgb_root = Path(rgb_root) if rgb_root else None
        self.feat_ext = feat_ext
        self.rgb_ext = rgb_ext
        self.feat_key = feat_key
        self.rgb_key = rgb_key
        self.feat_time_index = feat_time_index
        self.img_size = img_size
        self.files = sorted(self.feature_root.rglob(f"*{self.feat_ext}"))
        if not self.files:
            raise ValueError(f"No feature files found under {self.feature_root}.")
        self.resize = T.Resize(img_size, interpolation=InterpolationMode.BICUBIC)
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.files)

    def _resolve_rgb_path(self, feat_path, payload):
        if isinstance(payload, dict) and self.rgb_key in payload:
            return Path(payload[self.rgb_key])
        if self.rgb_root is None:
            raise ValueError("rgb_root is required when rgb_path is not stored in feature files.")
        rel = feat_path.relative_to(self.feature_root)
        return (self.rgb_root / rel).with_suffix(self.rgb_ext)

    def __getitem__(self, idx):
        feat_path = self.files[idx]
        payload = torch.load(feat_path, map_location="cpu")
        feats = payload[self.feat_key] if isinstance(payload, dict) else payload
        feats = torch.as_tensor(feats)
        if self.feat_time_index is not None and feats.dim() == 4:
            feats = feats[self.feat_time_index]
        if feats.dim() == 3:
            c_last = feats.shape[-1]
            if c_last >= feats.shape[0] and c_last >= feats.shape[1]:
                # Assume HWC, keep as-is.
                pass
            elif feats.shape[0] >= feats.shape[1] and feats.shape[0] >= feats.shape[2]:
                # Likely CHW -> HWC.
                feats = feats.permute(1, 2, 0)
        if feats.dim() != 3:
            raise ValueError(f"Expected feature shape [H,W,C], got {tuple(feats.shape)} at {feat_path}")
        rgb = None
        if isinstance(payload, dict) and "rgb" in payload:
            rgb = torch.as_tensor(payload["rgb"]).float()
            if rgb.dim() == 3 and rgb.shape[0] != 3 and rgb.shape[-1] == 3:
                rgb = rgb.permute(2, 0, 1)
            if rgb.dim() != 3 or rgb.shape[0] != 3:
                raise ValueError(f"Invalid rgb tensor in {feat_path}: {tuple(rgb.shape)}")
            if rgb.max() > 1.5:
                rgb = rgb / 255.0
            if rgb.shape[1:] != self.img_size:
                rgb = F.interpolate(
                    rgb.unsqueeze(0),
                    size=self.img_size,
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
        else:
            rgb_path = self._resolve_rgb_path(feat_path, payload)
            img = Image.open(rgb_path).convert("RGB")
            img = self.resize(img)
            rgb = self.to_tensor(img)
        return feats, rgb


class FeatureRgbDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.num_workers_val = args.num_workers_val or args.num_workers

    def setup(self, stage=None):
        self.train_ds = FeatureRgbDataset(
            self.args.train_feat_root,
            self.args.train_rgb_root,
            self.args.img_size,
            feat_ext=self.args.feat_ext,
            rgb_ext=self.args.rgb_ext,
            feat_key=self.args.feat_key,
            rgb_key=self.args.rgb_key,
            feat_time_index=self.args.feat_time_index,
        )
        self.val_ds = FeatureRgbDataset(
            self.args.val_feat_root or self.args.train_feat_root,
            self.args.val_rgb_root or self.args.train_rgb_root,
            self.args.img_size,
            feat_ext=self.args.feat_ext,
            rgb_ext=self.args.rgb_ext,
            feat_key=self.args.feat_key,
            rgb_key=self.args.rgb_key,
            feat_time_index=self.args.feat_time_index,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers_val,
            pin_memory=True,
            drop_last=False,
        )


class FeatureRgbDecoder(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.img_size = args.img_size
        self.feature_dim = args.feature_dim
        self.d_layers = args.d_layers
        self.l1_weight = args.l1_weight
        self.lpips_weight = args.lpips_weight

        if len(self.d_layers) not in (2, 4):
            raise ValueError("DPT decoder expects 2 or 4 feature layers.")

        self.decoder = DPTHead(
            nclass=3,
            in_channels=self.feature_dim,
            features=args.nfeats,
            use_bn=args.use_bn,
            out_channels=args.dpt_out_channels,
            use_clstoken=args.use_cls,
        )

        if self.lpips_weight > 0:
            try:
                import lpips  # type: ignore
            except ImportError as exc:
                raise ImportError("lpips is required for --lpips_weight > 0") from exc
            self.lpips = lpips.LPIPS(net=args.lpips_net)
            for p in self.lpips.parameters():
                p.requires_grad = False
            self.lpips.eval()
        else:
            self.lpips = None

        self.save_hyperparameters()

    def forward(self, feats):
        b, h, w, c = feats.shape
        expected = self.feature_dim * len(self.d_layers)
        if c != expected:
            raise ValueError(f"Feature dim mismatch: got {c}, expected {expected}.")
        feat_list = [
            feats[:, :, :, i * self.feature_dim:(i + 1) * self.feature_dim]
            for i in range(len(self.d_layers))
        ]
        if len(self.d_layers) == 2:
            # Duplicate to satisfy 4-scale DPT head with 2-layer features.
            feat_list = [feat_list[0], feat_list[0], feat_list[1], feat_list[1]]
        feat_list = [f.reshape(b, h * w, self.feature_dim) for f in feat_list]
        pred = self.decoder(feat_list, h, w)
        pred = F.interpolate(pred, size=self.img_size, mode="bicubic", align_corners=False)
        pred = torch.sigmoid(pred)
        return pred

    def _loss(self, pred, target):
        l1 = F.l1_loss(pred, target)
        loss = self.l1_weight * l1
        lpips_val = torch.tensor(0.0, device=pred.device)
        if self.lpips is not None:
            pred_lp = pred * 2.0 - 1.0
            target_lp = target * 2.0 - 1.0
            lpips_val = self.lpips(pred_lp, target_lp).mean()
            loss = loss + self.lpips_weight * lpips_val
        return loss, l1, lpips_val

    def training_step(self, batch, batch_idx):
        feats, rgb = batch
        pred = self.forward(feats)
        loss, l1, lpips_val = self._loss(pred, rgb)
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/l1", l1, prog_bar=False)
        if self.lpips is not None:
            self.log("train/lpips", lpips_val, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        feats, rgb = batch
        pred = self.forward(feats)
        loss, l1, lpips_val = self._loss(pred, rgb)
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/l1", l1, prog_bar=False)
        if self.lpips is not None:
            self.log("val/lpips", lpips_val, prog_bar=False)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.decoder.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )
        return optimizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_feat_root", type=str, required=True)
    parser.add_argument("--train_rgb_root", type=str, required=True)
    parser.add_argument("--val_feat_root", type=str, default=None)
    parser.add_argument("--val_rgb_root", type=str, default=None)
    parser.add_argument("--img_size", type=parse_tuple, default=(224, 448))
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--num_workers_val", type=int, default=None)
    parser.add_argument("--seed", type=int, default=123)

    parser.add_argument("--feat_ext", type=str, default=".pt")
    parser.add_argument("--rgb_ext", type=str, default=".png")
    parser.add_argument("--feat_key", type=str, default="features")
    parser.add_argument("--rgb_key", type=str, default="rgb_path")
    parser.add_argument("--feat_time_index", type=int, default=None)
    parser.add_argument("--feature_dim", type=int, default=768)
    parser.add_argument("--d_layers", type=parse_list, default=[2, 5, 8, 11])
    parser.add_argument("--use_bn", action="store_true", default=False)
    parser.add_argument("--use_cls", action="store_true", default=False)
    parser.add_argument("--nfeats", type=int, default=256)
    parser.add_argument("--dpt_out_channels", type=parse_list, default=[128, 256, 512, 512])

    parser.add_argument("--l1_weight", type=float, default=1.0)
    parser.add_argument("--lpips_weight", type=float, default=1.0)
    parser.add_argument("--lpips_net", type=str, default="vgg", choices=["vgg", "alex", "squeeze"])

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--precision", type=str, default="16-mixed", choices=["16-true", "16-mixed", "32-true"])
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--dst_path", type=str, default="./logs/rgb_decoder_from_feats")
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default="disabled", choices=["online", "offline", "disabled"])
    parser.add_argument("--strategy", type=str, default="auto", choices=["auto", "ddp", "ddp_spawn"])
    parser.add_argument("--find_unused_parameters", action="store_true", default=False)
    parser.add_argument("--ddp_timeout", type=int, default=1800)

    args = parser.parse_args()

    pl.seed_everything(args.seed, workers=True)

    data = FeatureRgbDataModule(args)
    model = FeatureRgbDecoder(args)

    callbacks = [
        ModelCheckpoint(
            monitor="val/loss",
            mode="min",
            save_top_k=1,
            save_last=True,
        )
    ]

    if args.wandb_mode == "disabled":
        logger = False
    else:
        from pytorch_lightning.loggers import WandbLogger

        logger = WandbLogger(
            project=args.wandb_project,
            name=args.wandb_name,
            mode=args.wandb_mode,
        )

    if args.strategy == "auto":
        strategy = (
            DDPStrategy(
                find_unused_parameters=args.find_unused_parameters,
                timeout=timedelta(seconds=args.ddp_timeout),
            )
            if args.num_gpus > 1
            else "auto"
        )
    elif args.strategy == "ddp":
        strategy = DDPStrategy(
            find_unused_parameters=args.find_unused_parameters,
            timeout=timedelta(seconds=args.ddp_timeout),
        )
    else:
        strategy = "ddp_spawn"

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args.num_gpus,
        strategy=strategy,
        callbacks=callbacks,
        max_epochs=args.max_epochs,
        default_root_dir=args.dst_path,
        precision=args.precision,
        logger=logger,
        log_every_n_steps=10,
    )

    trainer.fit(model, datamodule=data)


if __name__ == "__main__":
    main()
