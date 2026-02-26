import argparse
import os
import os.path as osp
from pathlib import Path

import numpy as np
from PIL import Image
import pytorch_lightning as pl
from datetime import timedelta
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
import torchvision.transforms as T

from dpt import DPTHead


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")


def parse_tuple(x):
    return tuple(map(int, x.split(",")))


def parse_list(x):
    return list(map(int, x.split(",")))


def list_images(root):
    root = Path(root)
    return sorted([p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS])


class ImageReconstructionDataset(Dataset):
    def __init__(self, paths, img_size, feature_extractor="dino"):
        self.paths = list(paths)
        self.resize = T.Resize(img_size, interpolation=InterpolationMode.BICUBIC)
        self.to_tensor = T.ToTensor()
        if feature_extractor in ("dino", "sam"):
            mean = (0.485, 0.456, 0.406)
            std = (0.229, 0.224, 0.225)
        else:
            mean = (0.48145466, 0.4578275, 0.40821073)
            std = (0.26862954, 0.26130258, 0.27577711)
        self.normalize = T.Normalize(mean=mean, std=std)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = self.resize(img)
        raw = self.to_tensor(img)
        norm = self.normalize(raw)
        return norm, raw


class ImageDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.train_root = args.train_root
        self.val_root = args.val_root
        self.data_root = args.data_root
        self.img_size = args.img_size
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.num_workers_val = args.num_workers_val or args.num_workers
        self.val_split = args.val_split
        self.seed = args.seed

    def setup(self, stage=None):
        if self.train_root:
            all_train = list_images(self.train_root)
            if self.val_root:
                train_paths = all_train
                val_paths = list_images(self.val_root)
            else:
                rng = np.random.default_rng(self.seed)
                rng.shuffle(all_train)
                split = int(len(all_train) * (1.0 - self.val_split))
                train_paths = all_train[:split]
                val_paths = all_train[split:]
        else:
            all_paths = list_images(self.data_root)
            rng = np.random.default_rng(self.seed)
            rng.shuffle(all_paths)
            split = int(len(all_paths) * (1.0 - self.val_split))
            train_paths = all_paths[:split]
            val_paths = all_paths[split:]

        self.train_ds = ImageReconstructionDataset(
            train_paths, self.img_size, feature_extractor=self.args.feature_extractor
        )
        self.val_ds = ImageReconstructionDataset(
            val_paths, self.img_size, feature_extractor=self.args.feature_extractor
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


class DinoV2RGBDecoder(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.img_size = args.img_size
        self.d_layers = args.d_layers
        self.l1_weight = args.l1_weight
        self.lpips_weight = args.lpips_weight

        if args.feature_extractor != "dino":
            raise ValueError("This script targets DINOv2 ViT-B as the backbone.")

        repo_or_dir = os.environ.get("DINO_REPO", "facebookresearch/dinov2")
        hub_source = "local" if osp.isdir(repo_or_dir) else "github"
        self.backbone = torch.hub.load(
            repo_or_dir,
            "dinov2_" + args.dinov2_variant,
            pretrained=True,
            source=hub_source,
        )
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.patch_size = 14
        self.patch_h = self.img_size[0] // self.patch_size
        self.patch_w = self.img_size[1] // self.patch_size
        self.emb_dim = self.backbone.embed_dim

        if isinstance(self.d_layers, list) and len(self.d_layers) != 4:
            raise ValueError("DPT decoder expects 4 feature layers. Use --d_layers with 4 entries.")

        self.decoder = DPTHead(
            nclass=3,
            in_channels=self.emb_dim,
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

    def forward(self, x_norm):
        with torch.no_grad():
            feats = self.backbone.get_intermediate_layers(
                x_norm, n=self.d_layers, reshape=False
            )
        pred = self.decoder(feats, self.patch_h, self.patch_w)
        pred = F.interpolate(pred, size=self.img_size, mode="bicubic", align_corners=False)
        pred = torch.sigmoid(pred)
        return pred

    def _loss(self, pred, target):
        l1 = F.l1_loss(pred, target)
        loss = self.l1_weight * l1
        lpips_val = torch.tensor(0.0, device=pred.device)
        if self.lpips is not None:
            pred_lp = (pred * 2.0 - 1.0).float()
            target_lp = (target * 2.0 - 1.0).float()
            lpips_val = self.lpips(pred_lp, target_lp).mean()
            loss = loss + self.lpips_weight * lpips_val
        return loss, l1, lpips_val

    def training_step(self, batch, batch_idx):
        x_norm, x_raw = batch
        pred = self.forward(x_norm)
        loss, l1, lpips_val = self._loss(pred, x_raw)
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/l1", l1, prog_bar=False)
        if self.lpips is not None:
            self.log("train/lpips", lpips_val, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x_norm, x_raw = batch
        pred = self.forward(x_norm)
        loss, l1, lpips_val = self._loss(pred, x_raw)
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
    parser.add_argument("--data_root", type=str, default=None, help="Root with images (used if train/val roots are not set).")
    parser.add_argument("--train_root", type=str, default=None, help="Optional train images root.")
    parser.add_argument("--val_root", type=str, default=None, help="Optional val images root.")
    parser.add_argument("--val_split", type=float, default=0.05, help="Val split if only data_root is given.")
    parser.add_argument("--img_size", type=parse_tuple, default=(224, 448))
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--num_workers_val", type=int, default=None)
    parser.add_argument("--seed", type=int, default=123)

    parser.add_argument("--feature_extractor", type=str, default="dino", choices=["dino"])
    parser.add_argument("--dinov2_variant", type=str, default="vitb14_reg")
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
    parser.add_argument("--dst_path", type=str, default="./logs/rgb_decoder")
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default="disabled", choices=["online", "offline", "disabled"])
    parser.add_argument("--strategy", type=str, default="auto", choices=["auto", "ddp", "ddp_spawn"])
    parser.add_argument("--find_unused_parameters", action="store_true", default=False)
    parser.add_argument("--ddp_timeout", type=int, default=1800)

    args = parser.parse_args()
    if not args.data_root and not args.train_root:
        raise ValueError("Provide --data_root or --train_root.")

    pl.seed_everything(args.seed, workers=True)

    data = ImageDataModule(args)
    model = DinoV2RGBDecoder(args)

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
