import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import pytorch_lightning as pl
from torch import optim
import einops
from time import time
import torchvision.transforms as T
from torchvision.transforms import functional as TF
import numpy as np
from torchmetrics import JaccardIndex
from torchmetrics.aggregation import MeanMetric
from src.attention_masked import MaskTransformer
import math
from dpt import DPTHead
import timm
import time as time_mod
import shutil
from src.drifting_utils import compute_V, build_token_sample_ids

def update_depth_metrics(pred, gt, d1_m, d2_m, d3_m, abs_rel_m, rmse_m, log_10_m, rmse_log_m, silog_m, sq_rel_m):
    valid_pixels = gt > 0
    pred = pred[valid_pixels]
    gt = gt[valid_pixels]
    thresh = torch.maximum((gt / pred), (pred / gt))
    d1 = (thresh < 1.25).float().mean()
    d2 = (thresh < 1.25 ** 2).float().mean()
    d3 = (thresh < 1.25 ** 3).float().mean()
    d1_m.update(d1)
    d2_m.update(d2)
    d3_m.update(d3)
    abs_rel = torch.mean(torch.abs(gt - pred) / gt)
    sq_rel = torch.mean(((gt - pred) ** 2) / gt)
    rmse = (gt - pred) ** 2
    rmse = torch.sqrt(rmse.float().mean())
    rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())
    err = torch.log(pred) - torch.log(gt)
    silog = torch.sqrt(torch.mean(err ** 2) - torch.mean(err) ** 2) * 100
    log_10 = (torch.abs(torch.log10(gt) - torch.log10(pred))).mean()
    abs_rel_m.update(abs_rel)
    rmse_m.update(rmse)
    log_10_m.update(log_10)
    rmse_log_m.update(rmse_log)
    silog_m.update(silog)
    sq_rel_m.update(sq_rel)

def compute_depth_metrics(d1_m, d2_m, d3_m, abs_rel_m, rmse_m, log_10_m, rmse_log_m, silog_m, sq_rel_m):
    d1 = d1_m.compute()
    d2 = d2_m.compute()
    d3 = d3_m.compute()
    abs_rel = abs_rel_m.compute()
    rmse = rmse_m.compute()
    log_10 = log_10_m.compute()
    rmse_log = rmse_log_m.compute()
    silog = silog_m.compute()
    sq_rel = sq_rel_m.compute()
    return d1, d2, d3, abs_rel, rmse, log_10, rmse_log, silog, sq_rel
    
def reset_depth_metrics(d1_m, d2_m, d3_m, abs_rel_m, rmse_m, log_10_m, rmse_log_m, silog_m, sq_rel_m):
    d1_m.reset()
    d2_m.reset()
    d3_m.reset()
    abs_rel_m.reset()
    rmse_m.reset()
    log_10_m.reset()
    rmse_log_m.reset()
    silog_m.reset()
    sq_rel_m.reset()

def update_normal_metrics(pred, gt, mean_ae_m, median_ae_m, rmse_m, a1_m, a2_m, a3_m, a4_m, a5_m):
    """ compute per-pixel surface normal error in degrees
        NOTE: pred_norm and gt_norm should be torch tensors of shape (B, 3, ...)
    """
    pred_error = torch.cosine_similarity(pred, gt, dim=1)
    pred_error = torch.clamp(pred_error, min=-1.0, max=1.0)
    pred_error = torch.acos(pred_error) * 180.0 / np.pi
    pred_error = pred_error.unsqueeze(1)    # (B, 1, ...)
    mean_ae = pred_error.mean()
    median_ae = pred_error.median()
    rmse = torch.sqrt((pred_error ** 2).mean())
    a1 = 100*(pred_error < 5).float().mean()
    a2 = 100*(pred_error < 7.5).float().mean()
    a3 = 100*(pred_error < 11.25).float().mean()
    a4 = 100*(pred_error < 22.5).float().mean()
    a5 = 100*(pred_error < 30).float().mean()
    mean_ae_m.update(mean_ae)
    median_ae_m.update(median_ae)
    rmse_m.update(rmse)
    a1_m.update(a1)
    a2_m.update(a2)
    a3_m.update(a3)
    a4_m.update(a4)
    a5_m.update(a5)

def compute_normal_metrics(mean_ae_m, median_ae_m, rmse_m, a1_m, a2_m, a3_m, a4_m, a5_m):
    mean_ae = mean_ae_m.compute()
    median_ae = median_ae_m.compute()
    rmse = rmse_m.compute()
    a1 = a1_m.compute()
    a2 = a2_m.compute()
    a3 = a3_m.compute()
    a4 = a4_m.compute()
    a5 = a5_m.compute()
    return mean_ae, median_ae, rmse, a1, a2, a3, a4, a5

def reset_normal_metrics(mean_ae_m, median_ae_m, rmse_m, a1_m, a2_m, a3_m, a4_m, a5_m):
    mean_ae_m.reset()
    median_ae_m.reset()
    rmse_m.reset()
    a1_m.reset()
    a2_m.reset()
    a3_m.reset()
    a4_m.reset()
    a5_m.reset()


class Dino_f(pl.LightningModule):
    def __init__(self,args):
        super(Dino_f,self).__init__()
        self.args = args
        self.sequence_length = args.sequence_length # 4
        self.batch_size = args.batch_size 
        self.hidden_dim = args.hidden_dim 
        self.heads = args.heads
        self.layers = args.layers 
        self.dropout = args.dropout
        self.loss_type = args.loss_type
        self.img_size  = args.img_size
        self.d_layers = args.d_layers
        self.patch_size = 14 if self.args.feature_extractor in ['dino', 'eva2-clip'] else 16
        self.d_num_layers = len(self.d_layers) if isinstance(self.d_layers, list) else self.d_layers
        if not self.args.crop_feats and not self.args.sliding_window_inference:
            self.shape = (self.sequence_length,self.img_size[0]//(self.patch_size), self.img_size[1]//(self.patch_size))
        else:
            self.shape = (self.sequence_length,self.img_size[0]//(self.patch_size*2), self.img_size[1]//(self.patch_size*2))
        self.dino_v2 = None
        self.eva2clip = None
        self.sam = None
        self.feature_dim = self._feature_dim_from_args()
        self.embedding_dim = self.d_num_layers * self.feature_dim
        self.use_precomputed_feats = getattr(args, "use_precomputed_feats", False)
        self.pca_whiten_alpha = max(0.0, float(getattr(args, "pca_whiten_alpha", 0.0)))
        self.pca_whiten_eps = max(1e-12, float(getattr(args, "pca_whiten_eps", 1e-8)))
        if self.args.pca_ckpt:
            self.pca_dict = torch.load(self.args.pca_ckpt, weights_only=False)
            self.pca = self.pca_dict['pca_model']
            self.pca_mean = torch.nn.Parameter(torch.as_tensor(self.pca.mean_, dtype=torch.float32), requires_grad=False)
            self.pca_components = torch.nn.Parameter(torch.as_tensor(self.pca.components_, dtype=torch.float32), requires_grad=False)
            self.mean = torch.nn.Parameter(torch.as_tensor(self.pca_dict['mean'], dtype=torch.float32), requires_grad=False)
            self.std = torch.nn.Parameter(torch.as_tensor(self.pca_dict['std'], dtype=torch.float32), requires_grad=False)
            pca_var = getattr(self.pca, "explained_variance_", None)
            if pca_var is None:
                pca_var_t = torch.ones(self.pca_components.shape[0], dtype=torch.float32)
            else:
                pca_var_t = torch.as_tensor(pca_var, dtype=torch.float32).flatten()
                if pca_var_t.numel() < self.pca_components.shape[0]:
                    pad = torch.ones(self.pca_components.shape[0] - pca_var_t.numel(), dtype=torch.float32)
                    pca_var_t = torch.cat([pca_var_t, pad], dim=0)
                elif pca_var_t.numel() > self.pca_components.shape[0]:
                    pca_var_t = pca_var_t[:self.pca_components.shape[0]]
            self.pca_explained_variance = torch.nn.Parameter(pca_var_t, requires_grad=False)
            self.embedding_dim = self.pca_components.shape[0]
        self.maskvit = MaskTransformer(shape=self.shape, embedding_dim=self.embedding_dim, hidden_dim=self.hidden_dim, depth=self.layers,
                                       heads=self.heads, mlp_dim=4*self.hidden_dim, dropout=self.dropout,use_fc_bias=args.use_fc_bias,
                                       seperable_attention=args.seperable_attention,seperable_window_size=args.seperable_window_size, use_first_last=args.use_first_last)
        self.use_drifting_loss = getattr(args, "use_drifting_loss", False)
        self.drift_temperatures = tuple(getattr(args, "drift_temperatures", [0.02, 0.05, 0.2]))
        self.drift_log_interval = max(1, int(getattr(args, "drift_log_interval", 20)))
        self.drift_antisymmetry_interval = max(1, int(getattr(args, "drift_antisymmetry_interval", 200)))
        self.drift_metric_token_cap = max(8, int(getattr(args, "drift_metric_token_cap", 512)))
        self.drift_train_token_cap = max(0, int(getattr(args, "drift_train_token_cap", 0)))
        self.drift_diversity_k = max(2, int(getattr(args, "drift_diversity_k", 3)))
        self.drift_v_clip = max(0.0, float(getattr(args, "drift_v_clip", 50.0)))
        self.drift_all_gather_neg = bool(getattr(args, "drift_all_gather_neg", False))
        self.drift_adaptive_temp = bool(getattr(args, "drift_adaptive_temp", False))
        self.drift_temp_ema_decay = min(0.999, max(0.0, float(getattr(args, "drift_temp_ema_decay", 0.95))))
        self.drift_temp_update_interval = max(1, int(getattr(args, "drift_temp_update_interval", 20)))
        self.drift_temp_token_cap = max(64, int(getattr(args, "drift_temp_token_cap", 1024)))
        self.drift_temp_min_scale = max(1e-3, float(getattr(args, "drift_temp_min_scale", 0.5)))
        self.drift_temp_max_scale = max(self.drift_temp_min_scale, float(getattr(args, "drift_temp_max_scale", 2.0)))
        self._drift_temp_ref_dist = float(getattr(args, "drift_temp_ref_dist", 0.0))
        if self._drift_temp_ref_dist <= 0.0:
            self._drift_temp_ref_dist = None
        self._drift_temp_scale_ema = 1.0
        self.use_language_condition = args.use_language_condition
        self.use_precomputed_text = getattr(args, "use_precomputed_text", False)
        if self.use_language_condition:
            clip_cache_dir = args.clip_cache_dir
            if clip_cache_dir:
                if os.path.basename(clip_cache_dir.rstrip("/")) == "huggingface":
                    clip_cache_dir = os.path.join(clip_cache_dir, "hub")
                elif os.path.isdir(os.path.join(clip_cache_dir, "hub")):
                    clip_cache_dir = os.path.join(clip_cache_dir, "hub")
            self.text_proj = nn.Linear(args.clip_text_dim, self.hidden_dim, bias=True)
            if not self.use_precomputed_text:
                self.clip_sources = {
                    "model_id": args.clip_model_name,
                    "cache_dir": clip_cache_dir,
                    "local_files_only": args.clip_local_files_only,
                }
                self.clip_tokenizer = None
                self.clip_text_model = None
            else:
                self.clip_sources = None
                self.clip_tokenizer = None
                self.clip_text_model = None
        self.train_mask_frames = args.train_mask_frames
        self.train_mask_mode = args.train_mask_mode
        self.masking = args.masking
        assert self.masking in ("half_half", "simple_replace", "half_half_previous")
        if self.masking in ("half_half", "half_half_previous"): # default 
            self.mask_vector = torch.nn.Parameter(torch.randn(1, 1, 1, 1, self.hidden_dim//2))
            self.unmask_vector = torch.nn.Parameter(torch.randn(1, 1, 1, 1, self.hidden_dim//2))
            if self.masking=="half_half":
                self.replace_vector = torch.nn.Parameter(torch.randn(1, 1, 1, 1, self.hidden_dim//2))
            self.embed = nn.Linear(self.embedding_dim, self.hidden_dim//2)
        elif self.masking == "simple_replace":
            self.embed = nn.Linear(self.embedding_dim, self.hidden_dim, bias=True)
            self.replace_vector = nn.Parameter(torch.zeros(1, 1, 1, 1, self.hidden_dim))
            torch.nn.init.normal_(self.replace_vector, std=.02)
            self.maskvit.fc_in = nn.Identity()
        self.activation = nn.Sigmoid() if args.output_activation == "sigmoid" else nn.Identity()
        # Necessary for evaluation
        self.mean_metric = MeanMetric()
        if self.args.eval_modality in ["segm", "depth", "surface_normals"]:   
            self.ignore_index = 255 if self.args.eval_modality == "segm" else 0
            self.head = DPTHead(nclass=self.args.num_classes,in_channels=self.feature_dim, features=self.args.nfeats,
                                use_bn=self.args.use_bn, out_channels=self.args.dpt_out_channels, use_clstoken=self.args.use_cls)
            self.patch_h = self.img_size[0] // self.patch_size
            self.patch_w = self.img_size[1] // self.patch_size
            # Load the head checkpoint
            if self.args.head_ckpt is not None:
                state_dict = {}
                for k, v in torch.load(self.args.head_ckpt)["state_dict"].items():
                    state_dict[k.replace("head.","")] = v
                self.head.load_state_dict(state_dict)
                self.head.eval()
                for param in self.head.parameters():
                    param.requires_grad = False
            if self.args.eval_modality == "segm":
                self.iou_metric = JaccardIndex(task="multiclass", num_classes=self.args.num_classes, ignore_index=self.ignore_index, average=None)
            elif self.args.eval_modality == "depth":
                self.ignore_index = 0
                self.d1 = MeanMetric()
                self.d2 = MeanMetric()
                self.d3 = MeanMetric()
                self.abs_rel = MeanMetric()
                self.rmse = MeanMetric()
                self.log_10 = MeanMetric()
                self.rmse_log = MeanMetric()
                self.silog = MeanMetric()
                self.sq_rel = MeanMetric()
            elif self.args.eval_modality == "surface_normals":
                self.mean_ae = MeanMetric()
                self.median_ae = MeanMetric()
                self.rmse = MeanMetric()
                self.a1 = MeanMetric()
                self.a2 = MeanMetric()
                self.a3 = MeanMetric()
                self.a4 = MeanMetric()
                self.a5 = MeanMetric()
        self.batch_crops = []
        self.random_crop = T.RandomCrop(16,32)
        self.save_hyperparameters()

    def _init_clip_models(self):
        from transformers import CLIPTokenizer, CLIPTextModel
        tokenizer_source, text_model_source = self._resolve_clip_sources(
            self.clip_sources["model_id"],
            self.clip_sources["cache_dir"],
            self.clip_sources["local_files_only"],
        )
        self.clip_tokenizer = CLIPTokenizer.from_pretrained(
            tokenizer_source,
            cache_dir=self.clip_sources["cache_dir"],
            local_files_only=self.clip_sources["local_files_only"],
        )
        self.clip_text_model = CLIPTextModel.from_pretrained(
            text_model_source,
            use_safetensors=True,
            cache_dir=self.clip_sources["cache_dir"],
            local_files_only=self.clip_sources["local_files_only"],
        )
        for param in self.clip_text_model.parameters():
            param.requires_grad = False
        self.clip_text_model.eval()
        if hasattr(self, "device"):
            self.clip_text_model.to(self.device)

    def _resolve_clip_sources(self, model_id, cache_dir, local_only):
        if os.path.isdir(model_id):
            return model_id, model_id
        offline = local_only or os.environ.get("HF_HUB_OFFLINE") or os.environ.get("TRANSFORMERS_OFFLINE")
        if not offline:
            return model_id, model_id
        cache_root = cache_dir or os.environ.get("HF_HOME")
        if cache_root and os.path.basename(cache_root.rstrip("/")) != "hub":
            if os.path.isdir(os.path.join(cache_root, "hub")):
                cache_root = os.path.join(cache_root, "hub")
        if not cache_root:
            return model_id, model_id
        model_dir = os.path.join(cache_root, "models--" + model_id.replace("/", "--"))
        snapshots_dir = os.path.join(model_dir, "snapshots")
        if not os.path.isdir(snapshots_dir):
            return model_id, model_id
        snapshot_paths = [
            os.path.join(snapshots_dir, name)
            for name in os.listdir(snapshots_dir)
            if os.path.isdir(os.path.join(snapshots_dir, name))
        ]
        tokenizer_path = None
        text_model_path = None
        for snap in snapshot_paths:
            if tokenizer_path is None and (
                os.path.isfile(os.path.join(snap, "tokenizer.json"))
                or os.path.isfile(os.path.join(snap, "vocab.json"))
            ):
                tokenizer_path = snap
            if text_model_path is None and os.path.isfile(os.path.join(snap, "model.safetensors")):
                text_model_path = snap
        if tokenizer_path and text_model_path and tokenizer_path != text_model_path:
            combined = self._ensure_combined_clip_dir(tokenizer_path, text_model_path)
            return combined, combined
        return tokenizer_path or model_id, text_model_path or model_id

    def _ensure_combined_clip_dir(self, tokenizer_path, text_model_path):
        combined_dir = os.path.join(os.path.dirname(text_model_path), "combined_clip")
        os.makedirs(combined_dir, exist_ok=True)
        for name in [
            "config.json",
            "tokenizer.json",
            "vocab.json",
            "merges.txt",
            "tokenizer_config.json",
            "special_tokens_map.json",
        ]:
            src = os.path.join(tokenizer_path, name)
            dst = os.path.join(combined_dir, name)
            if os.path.isfile(src) and not os.path.isfile(dst):
                shutil.copy2(src, dst)
        model_src = os.path.join(text_model_path, "model.safetensors")
        model_dst = os.path.join(combined_dir, "model.safetensors")
        if os.path.isfile(model_src) and not os.path.isfile(model_dst):
            shutil.copy2(model_src, model_dst)
        return combined_dir

    def _feature_dim_from_args(self):
        if self.args.feature_extractor == "dino":
            if str(self.args.dinov2_variant).startswith("vits14"):
                return 384
            if str(self.args.dinov2_variant).startswith("vitb14"):
                return 768
        elif self.args.feature_extractor == "eva2-clip":
            return 768
        elif self.args.feature_extractor == "sam":
            return 768
        raise ValueError(f"Unknown feature_extractor: {self.args.feature_extractor}")

    def _init_feature_extractor(self):
        if self.args.feature_extractor == "dino" and self.dino_v2 is None:
            dino_repo = os.environ.get("DINO_REPO", "facebookresearch/dinov2")
            if os.path.isdir(dino_repo):
                repo_or_dir = dino_repo
                hub_source = "local"
            else:
                repo_or_dir = "facebookresearch/dinov2"
                hub_source = "github"
            self.dino_v2 = torch.hub.load(
                repo_or_dir,
                'dinov2_' + self.args.dinov2_variant,
                pretrained=True,
                source=hub_source,
            )
            for param in self.dino_v2.parameters():
                param.requires_grad = False
            self.dino_v2.eval()
            self.feature_dim = self.dino_v2.embed_dim
        elif self.args.feature_extractor == "eva2-clip" and self.eva2clip is None:
            self.eva2clip = timm.create_model(
                'eva02_base_patch14_448.mim_in22k_ft_in1k',
                pretrained=True,
                img_size=(self.img_size[0], self.img_size[1])
            )
            for param in self.eva2clip.parameters():
                param.requires_grad = False
            self.eva2clip.eval()
            self.feature_dim = self.eva2clip.embed_dim
        elif self.args.feature_extractor == "sam" and self.sam is None:
            self.sam = timm.create_model(
                'timm/samvit_base_patch16.sa1b',
                pretrained=True,
                pretrained_cfg_overlay={'input_size': (3, self.img_size[0], self.img_size[1])}
            )
            for param in self.sam.parameters():
                param.requires_grad = False
            self.sam.eval()
            self.feature_dim = self.sam.embed_dim

    def _ddp_barrier(self):
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return
        if getattr(self, "trainer", None) is not None and hasattr(self.trainer, "strategy"):
            self.trainer.strategy.barrier()
            return
        device_ids = [torch.cuda.current_device()] if torch.cuda.is_available() else None
        torch.distributed.barrier(device_ids=device_ids)

    def _log_stage(self, message):
        log_path = getattr(self.args, "ddp_stage_log_path", None)
        if not log_path:
            return
        try:
            log_dir = os.path.dirname(log_path)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))
            pid = os.getpid()
            ts = time_mod.strftime("%Y-%m-%d %H:%M:%S")
            with open(log_path, "a") as f:
                f.write(f"{ts} rank={rank} pid={pid} {message}\n")
        except Exception:
            pass

    def setup(self, stage=None):
        self._log_stage("setup_start")
        if self.use_language_condition and not self.use_precomputed_text and self.clip_text_model is None:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                if self.global_rank == 0:
                    self._log_stage("clip_init_rank0_start")
                    self._init_clip_models()
                    self._log_stage("clip_init_rank0_done")
                self._log_stage("clip_barrier_1")
                self._ddp_barrier()
                if self.global_rank != 0:
                    self._log_stage("clip_init_rankN_start")
                    self._init_clip_models()
                    self._log_stage("clip_init_rankN_done")
                self._log_stage("clip_barrier_2")
                self._ddp_barrier()
            else:
                self._log_stage("clip_init_single_start")
                self._init_clip_models()
                self._log_stage("clip_init_single_done")
        if self.use_precomputed_feats:
            self._log_stage("setup_end_precomputed_feats")
            return
        if self.dino_v2 is None and self.eva2clip is None and self.sam is None:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                if self.global_rank == 0:
                    self._log_stage("feat_init_rank0_start")
                    self._init_feature_extractor()
                    self._log_stage("feat_init_rank0_done")
                self._log_stage("feat_barrier_1")
                self._ddp_barrier()
                if self.global_rank != 0:
                    self._log_stage("feat_init_rankN_start")
                    self._init_feature_extractor()
                    self._log_stage("feat_init_rankN_done")
                self._log_stage("feat_barrier_2")
                self._ddp_barrier()
            else:
                self._log_stage("feat_init_single_start")
                self._init_feature_extractor()
                self._log_stage("feat_init_single_done")
        self._log_stage("setup_end")

    def cmd_to_caption(self, cmd):
        plain_caption_dict = {
            0: "Go straight.",
            1: "Pass the intersection.",
            2: "Turn left.",
            3: "Turn right.",
            4: "Change to the left lane.",
            5: "Change to the right lane.",
            6: "Go to the left lane branch.",
            7: "Go to the right lane branch.",
            8: "Pass the crosswalk.",
            9: "Pass the railroad.",
            10: "Merge.",
            11: "Make a U-turn.",
            12: "Stop.",
            13: "Deviate.",
        }
        return plain_caption_dict.get(int(cmd), "Go straight.")

    def encode_text(self, captions, device):
        if self.use_precomputed_text:
            raise RuntimeError("encode_text called with use_precomputed_text=True.")
        if self.clip_text_model is None or self.clip_tokenizer is None:
            self._init_clip_models()
        tokens = self.clip_tokenizer(
            captions,
            padding=True,
            truncation=True,
            max_length=self.args.clip_max_length,
            return_tensors="pt",
        )
        input_ids = tokens["input_ids"].to(device)
        attention_mask = tokens["attention_mask"].to(device)
        with torch.no_grad():
            outputs = self.clip_text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_tokens = self.text_proj(outputs.last_hidden_state)
        text_mask = attention_mask == 0
        return text_tokens, text_mask
       

    def on_load_checkpoint(self, checkpoint):
        if self.args.high_res_adapt:
            # Interpolate positional embeddings
            pos_emb_h = checkpoint['state_dict']['maskvit.pos_embd.emb.d_1']
            pos_emb_h = F.interpolate(pos_emb_h.unsqueeze(0).unsqueeze(0), size=(self.img_size[0]//(self.patch_size), pos_emb_h.shape[1]), mode="bilinear")
            checkpoint['state_dict']['maskvit.pos_embd.emb.d_1'] = pos_emb_h.squeeze(0).squeeze(0)
            pos_emb_w = checkpoint['state_dict']['maskvit.pos_embd.emb.d_2']
            pos_emb_w = F.interpolate(pos_emb_w.unsqueeze(0).unsqueeze(0), size=(self.img_size[1]//(self.patch_size), pos_emb_w.shape[1]), mode="bilinear")
            checkpoint['state_dict']['maskvit.pos_embd.emb.d_2'] = pos_emb_w.squeeze(0).squeeze(0)
  
    def sliding_window(self, im, window_size, stride):
        B, SL, H, W, C = im.shape
        ws_h, ws_w = window_size
        s_h, s_w = stride
        windows = []
        for i in range(0, H-ws_h+1, s_h):
            for j in range(0, W-ws_w+1, s_w):
                windows.append(im[:,:,i:i+ws_h,j:j+ws_w])
        return torch.stack(windows)

    def merge_windows(self, windows_res, original_shape, window_size, stride):
        B, SL, H, W, C = original_shape
        ws_h, ws_w = window_size
        s_h, s_w = stride
        merged = torch.zeros(B, SL, H, W, C, dtype=windows_res.dtype)
        count = torch.zeros(B, SL, H, W, C, dtype=windows_res.dtype)
        idx = 0
        for i in range(0, H-ws_h+1, s_h):
            for j in range(0, W-ws_w+1, s_w):
                merged[:,:,i:i+ws_h,j:j+ws_w] += windows_res[idx].to(merged.device)
                count[:,:,i:i+ws_h,j:j+ws_w] += 1
                idx += 1
        merged /= count
        return merged


    def crop_feats(self, x, use_crop_params=False):
        B, SL, H, W, c= x.shape
        x = x.permute(0, 1, 4, 2, 3)
        if not use_crop_params:
            self.batch_crops = [self.random_crop.get_params(torch.zeros(H, W),(16,32)) for _ in range(B)]
        cropped_tensor = torch.stack([torch.stack([TF.crop(x[b, s], *self.batch_crops[b]) for s in range(SL)]) for b in range(B)])
        cropped_tensor = cropped_tensor.permute(0, 1, 3, 4, 2)
        return cropped_tensor

    def extract_features(self, x, reshape=False):
        if self.dino_v2 is None and self.eva2clip is None and self.sam is None:
            self._init_feature_extractor()
        with torch.no_grad():
            if self.args.feature_extractor == 'dino':
                x = self.dino_v2.get_intermediate_layers(x,n=self.d_layers, reshape=reshape)
            elif self.args.feature_extractor == 'eva2-clip':
                x = self.eva2clip.forward_intermediates(x, indices=self.d_layers, output_fmt  = 'NLC', norm=True, intermediates_only=True)
            elif self.args.feature_extractor == 'sam':
                x = self.sam.forward_intermediates(x, indices=self.d_layers, norm=False, intermediates_only=True) # Norm is False to avoide neck layer that reduces feature_dim to 256. Also output is in NCHW format
                x = [einops.rearrange(f, 'b c h w -> b (h w) c') for f in x]
            if self.d_num_layers > 1:
                x = torch.cat(x,dim=-1)
            else:
                x = x[0]
        return x
            
    def get_mask_tokens(self, x, mode="arccos", mask_frames=1):
        B, sl, h, w, c = x.shape # x.shape [B,T,H,W,C]
        assert mask_frames <= sl
        if mode == "full_mask":
            if self.sequence_length == 7:
                assert mask_frames <= 3
            else:
                assert mask_frames == 1
            mask = torch.ones(B,mask_frames, h,w, dtype=torch.bool)
        else:
            r = torch.rand(B) # Batch size
            if mode == "linear":                # linear scheduler
                val_to_mask = r
            elif mode == "square":              # square scheduler
                val_to_mask = (r ** 2)
            elif mode == "cosine":              # cosine scheduler
                val_to_mask = torch.cos(r * math.pi * 0.5)
            elif mode == "arccos":              # arc cosine scheduler
                val_to_mask = torch.arccos(r) / (math.pi * 0.5)
            else:
                val_to_mask = None
            # Create a mask of size [Batch,1,Height, Width] for the last frame of each sequence
            mask = torch.rand(size=(B, mask_frames, h, w)) < val_to_mask.view(B, 1, 1, 1)

        # Create the mask for all frames, by concatenating the mask for the last frame to a tensor of zeros(no mask) for first T-1 frames
        mask = torch.cat([torch.zeros(B,sl-mask_frames,h,w).bool(), mask], dim=1).to(x.device)

        if self.masking in ("half_half", "half_half_previous"):
            # Create the mask_tokens tensor
            mask_tokens = mask.unsqueeze(-1).float()*self.mask_vector.expand(B,sl,h,w,-1) + (~mask.unsqueeze(-1)).float()*self.unmask_vector.expand(B,sl,h,w,-1)
            # Embed the soft tokens
            embedded_tokens = self.embed(x)
            if self.masking == "half_half_previous":
                # Replace the embedded tokens at masked locations with the embedded tokens from the previous frames
                replace = torch.cat((torch.zeros((B,1,h,w,embedded_tokens.shape[-1]), dtype=embedded_tokens.dtype, device=embedded_tokens.device), embedded_tokens[:,:-1]), dim=1)
            else:
                # Replace the embedded tokens at masked locations with the replace vector
                replace = self.replace_vector.expand(B,sl,h,w,-1)
            embedded_tokens = torch.where(mask.unsqueeze(-1), replace, embedded_tokens)
            # Concatenate the masked tokens to the embedded tokens. Only take half of each to get the right hidden size
            final_tokens = torch.cat((embedded_tokens,mask_tokens), dim=-1)
        elif self.masking=="simple_replace":
            # Embed the soft tokens
            embedded_tokens = self.embed(x)
            # Replace the embedded tokens at masked locations with the replace vector
            final_tokens = torch.where(mask.unsqueeze(-1), self.replace_vector.expand(B,sl,h,w,-1), embedded_tokens)

        return final_tokens, mask

    def adap_sche(self, step, mode="arccos", leave=False):
        """ Create a sampling scheduler
        :param
            step  -> int:  number of prediction during inference
            mode  -> str:  the rate of value to unmask
            leave -> bool: tqdm arg on either to keep the bar or not
        :return
            scheduler -> torch.LongTensor(): the list of token to predict at each step
        """
        r = torch.linspace(1, 0, step+1)[1:]
        if mode == "root":              # root scheduler
            val_to_mask = 1 - (r ** .5)
        elif mode == "linear":          # linear scheduler
            val_to_mask = 1 - r
        elif mode == "square":          # square scheduler
            val_to_mask = 1 - (r ** 2)
        elif mode == "cosine":          # cosine scheduler
            val_to_mask = torch.cos(r * math.pi * 0.5)
        elif mode == "arccos":          # arc cosine scheduler
            val_to_mask = torch.arccos(r) / (math.pi * 0.5)
        else:
            return
        # fill the scheduler by the ratio of tokens to predict at each step
        sche = (val_to_mask / val_to_mask.sum()) * (self.shape[1]*self.shape[2])
        sche = sche.round()
        sche[sche == 0] = 1                                                  # add 1 to predict a least 1 token / step
        sche[-1] += (self.shape[1] * self.shape[2]) - sche.sum()         # need to sum up nb of code
        return tqdm(sche.int(), leave=leave)

    def downsample_feats(self, x):
        # Accepts shape [B*T,H,W,C]
        BT,H,W,C = x.shape # shape [8*5, 16, 32, 3072]
        emb_dim = self.feature_dim
        x = torch.movedim(x, -1, 1) # [B*T,C,H,W]
        features_list = [x[:,i*emb_dim:(i+1)*emb_dim] for i in range(self.d_num_layers)]
        features_list_down = [self.downsample_conv[i](x) for i, x in enumerate(features_list)]
        return torch.movedim(torch.cat(features_list_down, dim=1), 1, -1) # [B*T,H//2,W//2,C]

    def upsample_feats(self, x):
        # Accepts shape [B*T,H//2,W//2,C]
        BT,H,W,C = x.shape
        emb_dim = self.feature_dim
        x = torch.movedim(x, -1, 1) # [B*T,C//2,H//2,W]
        features_list = [x[:,i*emb_dim:(i+1)*emb_dim] for i in range(self.d_num_layers)]
        features_list_up = [self.upsample_conv[i](x) for i, x in enumerate(features_list)]
        return torch.movedim(torch.cat(features_list_up, dim=1), 1, -1) # [B*T,H,W,C]

    def pca_transform(self, x):
        BT, HW, C = x.shape
        mean = self.mean.to(device=x.device, dtype=x.dtype)
        std = self.std.to(device=x.device, dtype=x.dtype)
        pca_mean = self.pca_mean.to(device=x.device, dtype=x.dtype)
        pca_components = self.pca_components.to(device=x.device, dtype=x.dtype)
        x = (x - mean) / std
        x = x - pca_mean
        x_pca = torch.matmul(x, pca_components.T)
        if self.pca_whiten_alpha > 0.0:
            pca_var = self.pca_explained_variance.to(device=x.device, dtype=x.dtype).clamp(min=self.pca_whiten_eps)
            x_pca = x_pca * pca_var.pow(-0.5 * self.pca_whiten_alpha)
        return x_pca

    def pca_inverse_transform(self, x):
        B, T, H, W, C = x.shape
        mean = self.mean.to(device=x.device, dtype=x.dtype)
        std = self.std.to(device=x.device, dtype=x.dtype)
        pca_mean = self.pca_mean.to(device=x.device, dtype=x.dtype)
        pca_components = self.pca_components.to(device=x.device, dtype=x.dtype)
        if self.pca_whiten_alpha > 0.0:
            pca_var = self.pca_explained_variance.to(device=x.device, dtype=x.dtype).clamp(min=self.pca_whiten_eps)
            x = x * pca_var.pow(0.5 * self.pca_whiten_alpha)
        x = torch.matmul(x, pca_components) + pca_mean
        x = x * std + mean
        return x

    def _estimate_pairwise_distance(self, tokens):
        if tokens.shape[0] < 2:
            return None
        token_cap = min(tokens.shape[0], self.drift_temp_token_cap)
        if tokens.shape[0] > token_cap:
            idx = torch.randperm(tokens.shape[0], device=tokens.device)[:token_cap]
            tokens = tokens[idx]
        dists = torch.pdist(tokens, p=2)
        if dists.numel() == 0:
            return None
        return float(dists.mean().item())

    def _resolve_drift_temperatures(self, x_tokens_fp32, y_tokens_fp32):
        measured_dist = None
        if self.drift_adaptive_temp and (self.global_step % self.drift_temp_update_interval == 0):
            with torch.no_grad():
                measured_dist = self._estimate_pairwise_distance(
                    torch.cat([x_tokens_fp32.detach(), y_tokens_fp32.detach()], dim=0)
                )
                if measured_dist is not None and measured_dist > 0.0:
                    if self._drift_temp_ref_dist is None:
                        self._drift_temp_ref_dist = measured_dist
                    raw_scale = measured_dist / max(self._drift_temp_ref_dist, 1e-8)
                    raw_scale = min(self.drift_temp_max_scale, max(self.drift_temp_min_scale, raw_scale))
                    self._drift_temp_scale_ema = (
                        self.drift_temp_ema_decay * self._drift_temp_scale_ema
                        + (1.0 - self.drift_temp_ema_decay) * raw_scale
                    )
        if self.drift_adaptive_temp:
            temp_scale = min(self.drift_temp_max_scale, max(self.drift_temp_min_scale, self._drift_temp_scale_ema))
        else:
            temp_scale = 1.0
        temperatures = tuple(float(t) * temp_scale for t in self.drift_temperatures)
        return temperatures, temp_scale, measured_dist


    def preprocess(self, x):
        B, T, C, H, W = x.shape
        # DINOv2 accepts 4 dimensions [B,C,H,W]. 
        # We use flatten at batch and time dim of x.
        x = x.flatten(end_dim=1) # x.shape [B*T,C,H,W]
        x = self.extract_features(x) # [B*T,H*W,C]
        if self.args.pca_ckpt:
            x = self.pca_transform(x)
        x = einops.rearrange(x, 'b (h w) c -> b h w c',h=H//self.patch_size, w=W//self.patch_size)
        x = x.unflatten(dim=0, sizes=(B, self.sequence_length)) # [B,T,H,W,C]
        return x

    def postprocess(self, x):
        if self.args.pca_ckpt:
            x = self.pca_inverse_transform(x)
        return x

    def _is_dist_ready(self):
        return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1

    def _all_gather_variable_first_dim(self, tensor):
        if not self._is_dist_ready():
            return tensor

        world_size = dist.get_world_size()
        local_n = torch.tensor([tensor.shape[0]], device=tensor.device, dtype=torch.long)
        gathered_n = [torch.zeros_like(local_n) for _ in range(world_size)]
        dist.all_gather(gathered_n, local_n)
        sizes = [int(x.item()) for x in gathered_n]
        max_n = max(sizes)

        if tensor.shape[0] < max_n:
            pad_shape = (max_n - tensor.shape[0],) + tuple(tensor.shape[1:])
            pad = torch.zeros(pad_shape, device=tensor.device, dtype=tensor.dtype)
            tensor = torch.cat([tensor, pad], dim=0)

        gathered = [torch.empty_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered, tensor)
        chunks = [g[:sizes[i]] for i, g in enumerate(gathered) if sizes[i] > 0]
        return torch.cat(chunks, dim=0) if chunks else tensor[:0]

    def _predict_sequence(self, masked_x, text_tokens=None, text_mask=None):
        if self.args.vis_attn:
            x_pred, attn = self.maskvit(
                masked_x,
                return_attn=True,
                text_tokens=text_tokens,
                text_mask=text_mask,
            )
        else:
            x_pred, = self.maskvit(
                masked_x,
                text_tokens=text_tokens,
                text_mask=text_mask,
            )
            attn = None
        x_pred = einops.rearrange(x_pred, 'b (sl h w) c -> b sl h w c', sl=self.shape[0], h=self.shape[1], w=self.shape[2])
        x_pred = self.activation(x_pred)
        return x_pred, attn

    def _build_drift_input_tokens(self, x):
        B, T, H, W, C = x.shape
        noise = torch.randn(B, H, W, C, device=x.device, dtype=x.dtype)
        x_input = x.clone()
        x_input[:, -1] = noise
        if self.masking == "simple_replace":
            final_tokens = self.embed(x_input)
        elif self.masking in ("half_half", "half_half_previous"):
            embedded_tokens = self.embed(x_input)
            unmask_all = self.unmask_vector.expand(B, T, H, W, -1)
            final_tokens = torch.cat((embedded_tokens, unmask_all), dim=-1)
        else:
            raise ValueError(f"Unsupported masking mode for drifting: {self.masking}")
        return final_tokens

    def _drift_training_step(self, x, text_tokens=None, text_mask=None):
        """
        Drifting training step following Algorithm 1 of Deng et al. (2026).
        """
        B, _, H, W, C = x.shape
        final_tokens = self._build_drift_input_tokens(x)
        x_pred, _ = self._predict_sequence(
            final_tokens,
            text_tokens=text_tokens,
            text_mask=text_mask,
        )

        x_gen = x_pred[:, -1]
        y_pos = x[:, -1]
        n_tok = H * W
        x_tokens = x_gen.reshape(B * n_tok, C)
        y_pos_tokens = y_pos.reshape(B * n_tok, C)
        sample_ids = build_token_sample_ids(B, n_tok, x_tokens.device)
        token_subset_idx = None
        if self.drift_train_token_cap > 0 and x_tokens.shape[0] > self.drift_train_token_cap:
            token_subset_idx = torch.randperm(x_tokens.shape[0], device=x_tokens.device)[: self.drift_train_token_cap]
            x_tokens = x_tokens[token_subset_idx]
            y_pos_tokens = y_pos_tokens[token_subset_idx]
            sample_ids = sample_ids[token_subset_idx]

        # Drift kernels are numerically sensitive in mixed precision; force fp32 math.
        with torch.autocast(device_type=x.device.type, enabled=False):
            x_tokens_fp32 = torch.nan_to_num(x_tokens.float(), nan=0.0, posinf=1e4, neginf=-1e4)
            y_pos_tokens_fp32 = torch.nan_to_num(y_pos_tokens.float(), nan=0.0, posinf=1e4, neginf=-1e4)
            y_neg_tokens_fp32 = x_tokens_fp32

            y_neg_sample_ids = sample_ids
            if self.drift_all_gather_neg and self._is_dist_ready():
                sample_ids_for_neg = sample_ids + (int(dist.get_rank()) << 32)
                y_neg_tokens_fp32 = self._all_gather_variable_first_dim(x_tokens_fp32.detach())
                y_neg_sample_ids = self._all_gather_variable_first_dim(sample_ids_for_neg)

            if self._is_dist_ready():
                sample_ids = sample_ids + (int(dist.get_rank()) << 32)

            drift_temperatures, drift_temp_scale, drift_temp_dist = self._resolve_drift_temperatures(
                x_tokens_fp32, y_pos_tokens_fp32
            )
            V = compute_V(
                x=x_tokens_fp32,
                y_pos=y_pos_tokens_fp32,
                y_neg=y_neg_tokens_fp32,
                temperatures=drift_temperatures,
                x_sample_ids=sample_ids,
                y_pos_sample_ids=sample_ids,
                y_neg_sample_ids=y_neg_sample_ids,
            )
            V = torch.nan_to_num(V, nan=0.0, posinf=0.0, neginf=0.0)
            v_sq_norm_raw = V.detach().pow(2).sum(dim=-1).mean()
            if self.drift_v_clip > 0.0:
                v_norm = V.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                clip_scale = (self.drift_v_clip / v_norm).clamp(max=1.0)
                V = V * clip_scale

            x_drifted = (x_tokens_fp32 + V).detach()
            loss = F.mse_loss(x_tokens_fp32, x_drifted)
            if not torch.isfinite(loss):
                if self.global_rank == 0:
                    print(
                        "[DriftDebug] non-finite loss detected: "
                        f"x_absmax={x_tokens_fp32.abs().max().item():.4e} "
                        f"y_absmax={y_pos_tokens_fp32.abs().max().item():.4e} "
                        f"v_absmax={V.abs().max().item():.4e}"
                    )
                loss = x_tokens_fp32.sum() * 0.0

        metrics = {
            "Train/l_drift": loss.detach(),
            "Train/drift_v_sq_norm": V.detach().pow(2).sum(dim=-1).mean(),
            "Train/drift_v_sq_norm_raw": v_sq_norm_raw,
            "Train/drift_cos_to_gt": F.cosine_similarity(x_tokens_fp32.detach(), y_pos_tokens_fp32.detach(), dim=-1).mean(),
            "Train/drift_neg_pool_tokens": x_tokens_fp32.new_tensor(float(y_neg_tokens_fp32.shape[0])),
            "Train/drift_temp_scale": x_tokens_fp32.new_tensor(float(drift_temp_scale)),
            "Train/drift_temp_ref_dist": x_tokens_fp32.new_tensor(
                float(self._drift_temp_ref_dist) if self._drift_temp_ref_dist is not None else 0.0
            ),
            "Train/pred_batch_std": x_gen.detach().reshape(B, -1).std(dim=0, unbiased=False).mean(),
        }
        if drift_temp_dist is not None:
            metrics["Train/drift_temp_measured_dist"] = x_tokens_fp32.new_tensor(float(drift_temp_dist))
        if drift_temperatures:
            metrics["Train/drift_temp_min"] = x_tokens_fp32.new_tensor(float(min(drift_temperatures)))
            metrics["Train/drift_temp_max"] = x_tokens_fp32.new_tensor(float(max(drift_temperatures)))

        if self.global_step % self.drift_log_interval == 0:
            with torch.no_grad():
                samples = [x_tokens_fp32.detach()]
                for _ in range(self.drift_diversity_k - 1):
                    final_tokens_alt = self._build_drift_input_tokens(x)
                    pred_alt, _ = self._predict_sequence(
                        final_tokens_alt,
                        text_tokens=text_tokens,
                        text_mask=text_mask,
                    )
                    alt_tokens = torch.nan_to_num(
                        pred_alt[:, -1].reshape(B * n_tok, -1).float(),
                        nan=0.0,
                        posinf=1e4,
                        neginf=-1e4,
                    )
                    if token_subset_idx is not None:
                        alt_tokens = alt_tokens[token_subset_idx]
                    samples.append(alt_tokens)

                if len(samples) >= 2:
                    diversity_cos = F.cosine_similarity(samples[0], samples[1], dim=-1).mean()
                    metrics["Train/drift_diversity_cos"] = diversity_cos
                    metrics["Train/noise_sensitivity"] = diversity_cos
                pairwise_cos = []
                for i in range(len(samples)):
                    for j in range(i + 1, len(samples)):
                        pairwise_cos.append(F.cosine_similarity(samples[i], samples[j], dim=-1).mean())
                if pairwise_cos:
                    metrics["Train/diversity_pairwise_cos"] = torch.stack(pairwise_cos).mean()

                if self.global_step % self.drift_antisymmetry_interval == 0 and x_tokens.shape[0] >= 2:
                    token_cap = min(self.drift_metric_token_cap, x_tokens.shape[0])
                    idx = torch.randperm(x_tokens.shape[0], device=x.device)[:token_cap]
                    x_ref = x_tokens_fp32.detach()[idx]
                    y_pos_ref = y_pos_tokens_fp32.detach()[idx]
                    y_neg_ref = x_tokens_fp32.detach()[idx[torch.randperm(token_cap, device=x.device)]]
                    v_ab = compute_V(x=x_ref, y_pos=y_pos_ref, y_neg=y_neg_ref, temperatures=drift_temperatures)
                    v_ba = compute_V(x=x_ref, y_pos=y_neg_ref, y_neg=y_pos_ref, temperatures=drift_temperatures)
                    anti_num = (v_ab + v_ba).norm(dim=-1).mean()
                    anti_den = v_ab.norm(dim=-1).mean().clamp(min=1e-8)
                    metrics["Train/drift_antisymmetry_ratio"] = anti_num / anti_den

        return loss, metrics
    
    def calculate_loss(self, x_pred, x_target):
        if self.args.loss_type == "MSE":
            loss = F.mse_loss(x_pred, x_target)
        elif self.args.loss_type == "SmoothL1":
            loss = F.smooth_l1_loss(x_pred, x_target, beta=self.args.beta_smoothl1)
        elif self.args.loss_type == "L1":
            loss = F.l1_loss(x_pred, x_target)
        return loss

    def forward_loss(self, x_pred, x_target, mask):
        B, T, H, W, C = x_pred.shape
        x_target = x_target[mask] 
        x_pred = x_pred[mask] 
        return self.calculate_loss(x_pred, x_target)
        
    def sample(self, x, sched_mode="arccos", step=15, mask_frames=1):
        self.maskvit.eval()
        with torch.no_grad():
            x = self.preprocess(x)
            B, SL, H, W, C = x.shape
            if self.use_drifting_loss:
                if self.args.sliding_window_inference:
                    window_size = (16,32)
                    stride = (16,32)
                    x_wins = self.sliding_window(x, window_size, stride)
                    wins = []
                    for i in range(x_wins.shape[0]):
                        win = x_wins[i]
                        final_tokens = self._build_drift_input_tokens(win)
                        x_pred, _ = self._predict_sequence(final_tokens)
                        pred_win = win.clone()
                        pred_win[:, -1] = x_pred[:, -1]
                        wins.append(pred_win)
                    prediction = self.merge_windows(torch.stack(wins), (B, SL, H, W, C), window_size, stride).to(x.device)
                else:
                    if self.args.crop_feats:
                        x = self.crop_feats(x)
                        B, SL, H, W, C = x.shape
                    final_tokens = self._build_drift_input_tokens(x)
                    x_pred, _ = self._predict_sequence(final_tokens)
                    prediction = x.clone()
                    prediction[:, -1] = x_pred[:, -1]
                prediction = self.postprocess(prediction)
                loss = torch.tensor(0.0, device=x.device)
                return prediction, loss
            if not self.args.sliding_window_inference:
                if self.args.crop_feats:
                    x = self.crop_feats(x)
                masked_soft_tokens, mask = self.get_mask_tokens(x, mode="full_mask",mask_frames=mask_frames)
                mask = mask.to(x.device)
                if self.args.single_step_sample_train or step==1:
                    if self.args.vis_attn:
                        _, final_tokens, attn_weights = self.forward(x, masked_soft_tokens, mask)
                    else:
                        loss, final_tokens = self.forward(x, masked_soft_tokens, mask)
                else:
                    assert "Not implemented"
                prediction = self.postprocess(final_tokens)
            else:
                window_size = (16,32)
                stride = (16,32)
                x = self.sliding_window(x, window_size, stride)
                wins = []
                for i in range(x.shape[0]):
                    win = x[i]
                    masked_soft_tokens, mask = self.get_mask_tokens(win, mode="full_mask",mask_frames=mask_frames)
                    mask = mask.to(x.device)
                    if self.args.single_step_sample_train or step==1:
                        if self.args.vis_attn:
                            _, final_tokens, attn_weights = self.forward(x, masked_soft_tokens, mask)
                        else:
                            loss, final_tokens = self.forward(win, masked_soft_tokens, mask)
                    else:
                        # Instantiate scheduler
                        if isinstance(sched_mode, str):  # Standard ones
                            scheduler = self.adap_sche(step, mode=sched_mode)
                        else:  # Custom one
                            scheduler = sched_mode
                        final_tokens, loss = self.oracle_sample(x, masked_soft_tokens, mask, scheduler, step, mask_frames)
                    prediction = self.postprocess(final_tokens)
                    wins.append(prediction)
                prediction = self.merge_windows(torch.stack(wins), (B, SL, H, W, 3072), window_size, stride).to(x.device)
            return prediction, loss
            # return prediction, loss, final_tokens

    def sample_unroll(self, x, gt_feats, sched_mode="arccos", step=15, mask_frames=1, unroll_steps=3, ):
        self.maskvit.eval()
        with torch.no_grad():
            x = self.preprocess(x)
        B, SL, H, W, C = x.shape
        for i in range(unroll_steps):
            if self.use_drifting_loss:
                with torch.no_grad():
                    if not self.args.sliding_window_inference:
                        final_tokens = self._build_drift_input_tokens(x)
                        x_pred, _ = self._predict_sequence(final_tokens)
                        final_tokens = x.clone()
                        final_tokens[:, -1] = x_pred[:, -1]
                    else:
                        window_size = (16,32)
                        stride = (16,32)
                        x_s = self.sliding_window(x, window_size, stride)
                        wins = []
                        for j in range(x_s.shape[0]):
                            win = x_s[j]
                            drift_tokens = self._build_drift_input_tokens(win)
                            x_pred_win, _ = self._predict_sequence(drift_tokens)
                            pred_win = win.clone()
                            pred_win[:, -1] = x_pred_win[:, -1]
                            wins.append(pred_win)
                        final_tokens = self.merge_windows(torch.stack(wins), (B, SL, H, W, C), window_size, stride).to(x.device)
                x[:,-1] = final_tokens[:,-1]
                x = torch.cat((x[:,1:], x[:,-1].unsqueeze(1)), dim=1)
                continue
            if not self.args.sliding_window_inference:
                masked_soft_tokens, mask = self.get_mask_tokens(x, mode="full_mask",mask_frames=mask_frames)
                mask = mask.to(x.device)
                if self.args.single_step_sample_train or step==1:
                    if self.args.vis_attn:
                        _, final_tokens, attn_weights = self.forward(x, masked_soft_tokens, mask)
                    else:
                        loss, final_tokens= self.forward(x, masked_soft_tokens, mask)
                else:
                    assert "Not implemented"
                # x = self.postprocess(final_tokens)
            else:
                window_size = (16,32)
                stride = (16,32)
                x_s = self.sliding_window(x, window_size, stride)
                wins = []
                for i in range(x_s.shape[0]):
                    win = x_s[i]
                    masked_soft_tokens, mask = self.get_mask_tokens(win, mode="full_mask",mask_frames=mask_frames)
                    mask = mask.to(x.device)
                    if self.args.single_step_sample_train or step==1:
                        if self.args.vis_attn:
                            _, final_tokens, attn_weights = self.forward(x, masked_soft_tokens, mask)
                        else:
                            loss, final_tokens_win = self.forward(win, masked_soft_tokens, mask)
                    wins.append(final_tokens_win)
                final_tokens = self.merge_windows(torch.stack(wins), (B, SL, H, W, 1152), window_size, stride).to(x.device)
            x[:,-1] = final_tokens[:,-1]
            x = torch.cat((x[:,1:], x[:,-1].unsqueeze(1)), dim=1) # Mayve also try torch.zeros instead of x[:,-1]
        prediction = self.postprocess(x)
        loss = self.calculate_loss(prediction[:,-1].flatten(end_dim=-2), gt_feats.flatten(end_dim=-2))
        # return prediction, loss, x
        return prediction, loss

    def forward(self, x, masked_x, mask=None, text_tokens=None, text_mask=None):
        x_pred, attn = self._predict_sequence(masked_x, text_tokens=text_tokens, text_mask=text_mask)
        # if self.args.predict_residuals:
        #     x_prev = torch.cat([torch.zeros(B,1,H,W,C).to(x.device), x[:,:-1]], dim=1)
        #     x_pred = x_pred + x_prev
        loss = self.forward_loss(x_pred=x_pred, x_target=x, mask=mask)
        if self.args.vis_attn:
            return loss, x_pred, attn
        else:
            return loss, x_pred

    

    def training_step(self, x, batch_idx):
        text_tokens = None
        text_mask = None
        if isinstance(x, (list, tuple)):
            if self.use_language_condition:
                if self.use_precomputed_text:
                    if len(x) == 3:
                        frames, text_tokens, text_mask = x
                    elif len(x) == 5:
                        frames, _, _, text_tokens, text_mask = x
                    else:
                        raise ValueError("Expected batch to be (frames, text_tokens, text_mask) or (frames, cmd, blip, text_tokens, text_mask).")
                    text_tokens = self.text_proj(text_tokens.to(frames.device))
                    if text_mask is not None:
                        text_mask = text_mask.to(frames.device)
                else:
                    frames = x[0]
                    cmd = x[1]
                    captions = [self.cmd_to_caption(c) for c in cmd]
                    text_tokens, text_mask = self.encode_text(captions, frames.device)
            else:
                frames = x[0]
            x = frames
        B = x.shape[0]
        if self.use_precomputed_feats:
            if x.dim() != 5 or x.shape[-1] == 3:
                raise ValueError("use_precomputed_feats expects features with shape [B,T,H,W,C].")
            if self.args.pca_ckpt:
                t = x.shape[1]
                bt = x.shape[0] * t
                h = x.shape[2]
                w = x.shape[3]
                x = x.view(bt, h * w, x.shape[-1])
                x = self.pca_transform(x)
                x = x.view(B, t, h, w, -1)
        else:
            x = self.preprocess(x)
        # Mask the encoded tokens
        if self.args.crop_feats:
            x = self.crop_feats(x)
        if self.use_drifting_loss:
            loss, drift_metrics = self._drift_training_step(
                x,
                text_tokens=text_tokens,
                text_mask=text_mask,
            )
            for k, v in drift_metrics.items():
                self.log(
                    k,
                    v,
                    batch_size=B,
                    logger=True,
                    on_step=True,
                    on_epoch=False,
                    prog_bar=(k in {"Train/l_drift", "Train/drift_v_sq_norm"}),
                    rank_zero_only=True,
                )
        else:
            if self.sequence_length == 7:
                train_mask_frames = torch.randint(1, 4, (1,)) if self.training else 3
            else:
                train_mask_frames = self.train_mask_frames if self.training else 1
            masked_x, mask = self.get_mask_tokens(x, mode=self.train_mask_mode, mask_frames=train_mask_frames)
            # masked_x, mask = self.get_mask_tokens(x, mode="full_mask", mask_frames=train_mask_frames) # masked_x.shape [B,T,H,W,C], mask.shape [B,T,H,W,1]
            if self.args.vis_attn:
                loss, _, _ = self.forward(x, masked_x, mask, text_tokens=text_tokens, text_mask=text_mask)
            else:
                loss, _ = self.forward(x, masked_x, mask, text_tokens=text_tokens, text_mask=text_mask)
            self.log(
                "Train/mask_ratio",
                mask[:, -1].float().mean().item() * 100,
                logger=True,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                rank_zero_only=True,
            )
        self.log("Train/loss", loss, batch_size=B, logger=True, on_step=True, on_epoch=False, prog_bar=True, rank_zero_only=True)
        lr = self.optimizers().optimizer.param_groups[0]["lr"]
        self.log("Train/lr", lr, logger=True, on_step=True, on_epoch=False, prog_bar=True, rank_zero_only=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if self.args.eval_mode:
            return self.evaluation_step(batch, batch_idx)
        B = batch[0].shape[0]
        loss = self.training_step(batch, batch_idx)
        self.log('val/loss', loss, prog_bar=True, batch_size=B, logger=True, sync_dist=True, on_step=False, on_epoch=True)

    def baseline_evaluation_step(self, x, gt_feats):
        B = x.shape[0]
        with torch.inference_mode():
            x = self.preprocess(x)
            x = self.postprocess(x)
            loss = self.calculate_loss(x[:,-2].flatten(end_dim=-2), gt_feats.flatten(end_dim=-2))
            x[:,-1] = x[:,-2]
        return x, loss

    def evaluation_step(self, batch, batch_idx):
        gt_segm = None
        gt_depth = None
        gt_normals = None
        if torch.is_tensor(batch):
            data_tensor = batch
            gt_img = data_tensor[:, -1]
        elif isinstance(batch, (list, tuple)):
            if len(batch) == 0:
                raise ValueError("Empty batch received in evaluation_step.")
            if self.args.eval_modality is None:
                if len(batch) >= 2:
                    data_tensor, gt_img = batch[:2]
                else:
                    data_tensor = batch[0]
                    gt_img = data_tensor[:, -1]
            elif self.args.eval_modality == "segm":
                if len(batch) < 3:
                    raise ValueError(f"Expected at least 3 items for segm eval, got {len(batch)}.")
                data_tensor, gt_img, gt_segm = batch[:3]
            elif self.args.eval_modality == "depth":
                if len(batch) < 3:
                    raise ValueError(f"Expected at least 3 items for depth eval, got {len(batch)}.")
                data_tensor, gt_img, gt_depth = batch[:3]
            elif self.args.eval_modality == "surface_normals":
                if len(batch) < 3:
                    raise ValueError(f"Expected at least 3 items for surface_normals eval, got {len(batch)}.")
                data_tensor, gt_img, gt_normals = batch[:3]
            else:
                raise ValueError(f"Unsupported eval_modality: {self.args.eval_modality}")
        else:
            raise TypeError(f"Unsupported batch type in evaluation_step: {type(batch)}")
        B, sl, C, H, W = data_tensor.shape
        gt_feats = self.extract_features(gt_img)
        gt_feats = einops.rearrange(gt_feats, 'b (h w) c -> b h w c',h=H//self.patch_size, w=W//self.patch_size)
        if self.args.evaluate_baseline:
            samples, loss = self.baseline_evaluation_step(data_tensor,gt_feats=gt_feats)
        else:
            if self.args.eval_midterm:
                samples, loss = self.sample_unroll(data_tensor,gt_feats,sched_mode=self.train_mask_mode,step=self.args.step, unroll_steps=3)
            else:
                samples, loss = self.sample(data_tensor,sched_mode=self.train_mask_mode,step=self.args.step)
        # Evaluation
        pred_feats = samples[:,-1]
        gt_feats = gt_feats.unsqueeze(1)
        if self.args.crop_feats:
            gt_feats = self.crop_feats(gt_feats, use_crop_params=True)
        gt_feats = gt_feats.squeeze(1)
        pred_flat = pred_feats.reshape(B, -1, pred_feats.shape[-1])
        gt_flat = gt_feats.reshape(B, -1, gt_feats.shape[-1])
        feat_mse = F.mse_loss(pred_flat, gt_flat)
        feat_l1 = F.l1_loss(pred_flat, gt_flat)
        feat_cos = F.cosine_similarity(pred_flat, gt_flat, dim=-1).mean()
        pred_global = F.normalize(pred_flat.mean(dim=1), dim=-1)
        gt_global = F.normalize(gt_flat.mean(dim=1), dim=-1)
        sim = pred_global @ gt_global.T
        feat_recall1 = (sim.argmax(dim=1) == torch.arange(B, device=sim.device)).float().mean()
        self.log('val/feat_mse', feat_mse, prog_bar=False, batch_size=B, on_step=True, logger=True, rank_zero_only=True)
        self.log('val/feat_l1', feat_l1, prog_bar=False, batch_size=B, on_step=True, logger=True, rank_zero_only=True)
        self.log('val/feat_cos', feat_cos, prog_bar=False, batch_size=B, on_step=True, logger=True, rank_zero_only=True)
        self.log('val/feat_recall1', feat_recall1, prog_bar=False, batch_size=B, on_step=True, logger=True, rank_zero_only=True)
        self.mean_metric.update(loss)
        mean_loss = self.mean_metric.compute()
        self.log('val/mean_loss', mean_loss, prog_bar=True, batch_size=1, on_step=True, logger=True, rank_zero_only=True)
        print(f"Iteration {batch_idx}: Validation Loss: {mean_loss:.4f}")
        if self.args.eval_modality == "segm":
            pred_feats_list = [pred_feats[:,:,:,i*self.feature_dim:(i+1)*self.feature_dim] for i in range(self.d_num_layers)]
            pred_feats_list = [einops.rearrange(x, 'b h w c -> b (h w) c',h=H//self.patch_size, w=W//self.patch_size) for x in pred_feats_list]
            pred_segm = self.head(pred_feats_list,self.patch_h,self.patch_w)
            pred_segm = F.interpolate(pred_segm, size=(1024,2048), mode='bicubic', align_corners=False)
            self.iou_metric.update(pred_segm, gt_segm.squeeze(1))
            IoU = self.iou_metric.compute()
            mIoU = torch.mean(IoU)
            MO_mIoU = torch.mean(IoU[11:])
            self.log('val/mIoU', mIoU, prog_bar=True, batch_size=1, on_step=True, logger=True, rank_zero_only=True)
            self.log('val/MO_mIoU', MO_mIoU, prog_bar=True, batch_size=1, on_step=True, logger=True, rank_zero_only=True)
            print(f"Validation mIoU: {mIoU:.4f}, Validation MO_mIoU: {MO_mIoU:.4f}")
        elif self.args.eval_modality == "depth":
            pred_feats_list = [pred_feats[:,:,:,i*self.feature_dim:(i+1)*self.feature_dim] for i in range(self.d_num_layers)]
            pred_feats_list = [einops.rearrange(x, 'b h w c -> b (h w) c',h=H//self.patch_size, w=W//self.patch_size) for x in pred_feats_list]
            pred_depth = self.head(pred_feats_list,self.patch_h,self.patch_w)
            pred_depth = F.interpolate(pred_depth, size=(1024,2048), mode='bicubic', align_corners=False)
            pred_depth = pred_depth.argmax(dim=1)
            update_depth_metrics(pred_depth, gt_depth.squeeze(1), self.d1, self.d2, self.d3, self.abs_rel, self.rmse, self.log_10, self.rmse_log, self.silog, self.sq_rel)
            d1, d2, d3, abs_rel, rmse, log_10, rmse_log, silog, sq_rel = compute_depth_metrics(self.d1, self.d2, self.d3, self.abs_rel, self.rmse, self.log_10, self.rmse_log, self.silog, self.sq_rel)
            self.log('val/d1', d1, prog_bar=True, batch_size=1, on_step=True, logger=True, rank_zero_only=True)
            self.log('val/d2', d2, prog_bar=True, batch_size=1, on_step=True, logger=True, rank_zero_only=True)
            self.log('val/d3', d3, prog_bar=True, batch_size=1, on_step=True, logger=True, rank_zero_only=True)
            self.log('val/abs_rel', abs_rel, prog_bar=True, batch_size=1, on_step=True, logger=True, rank_zero_only=True)
            self.log('val/rmse', rmse, prog_bar=True, batch_size=1, on_step=True, logger=True, rank_zero_only=True)
            self.log('val/log_10', log_10, prog_bar=True, batch_size=1, on_step=True, logger=True, rank_zero_only=True)
            self.log('val/rmse_log', rmse_log, prog_bar=True, batch_size=1, on_step=True, logger=True, rank_zero_only=True)
            self.log('val/silog', silog, prog_bar=True, batch_size=1, on_step=True, logger=True, rank_zero_only=True)
            self.log('val/sq_rel', sq_rel, prog_bar=True, batch_size=1, on_step=True, logger=True, rank_zero_only=True)
        elif self.args.eval_modality == "surface_normals":
            pred_feats_list = [pred_feats[:,:,:,i*self.feature_dim:(i+1)*self.feature_dim] for i in range(self.d_num_layers)]
            pred_feats_list = [einops.rearrange(x, 'b h w c -> b (h w) c',h=H//self.patch_size, w=W//self.patch_size) for x in pred_feats_list]
            pred_normals = self.head(pred_feats_list,self.patch_h,self.patch_w)
            pred_normals = F.interpolate(pred_normals, size=(1024,2048), mode='bicubic', align_corners=False)
            update_normal_metrics(pred_normals, gt_normals, self.mean_ae, self.median_ae, self.rmse, self.a1, self.a2, self.a3, self.a4, self.a5)
            mean_ae, median_ae, rmse, a1, a2, a3, a4, a5 = compute_normal_metrics(self.mean_ae, self.median_ae, self.rmse, self.a1, self.a2, self.a3, self.a4, self.a5)
            self.log('val/mean_ae', mean_ae, prog_bar=True, batch_size=1, on_step=True, logger=True, rank_zero_only=True)
            self.log('val/median_ae', median_ae, prog_bar=True, batch_size=1, on_step=True, logger=True, rank_zero_only=True)
            self.log('val/rmse', rmse, prog_bar=True, batch_size=1, on_step=True, logger=True, rank_zero_only=True)
            self.log('val/a1', a1, prog_bar=True, batch_size=1, on_step=True, logger=True, rank_zero_only=True)
            self.log('val/a2', a2, prog_bar=True, batch_size=1, on_step=True, logger=True, rank_zero_only=True)
            self.log('val/a3', a3, prog_bar=True, batch_size=1, on_step=True, logger=True, rank_zero_only=True)
            self.log('val/a4', a4, prog_bar=True, batch_size=1, on_step=True, logger=True, rank_zero_only=True)
            self.log('val/a5', a5, prog_bar=True, batch_size=1, on_step=True, logger=True, rank_zero_only=True)
        self.log('val/loss', loss, prog_bar=True, batch_size=data_tensor.shape[0], sync_dist=True, on_step=False, on_epoch=True, logger=True)
        
        
    def on_validation_epoch_end(self):
        if self.args.eval_mode:
            mean_loss = self.mean_metric.compute()
            print(f"Validation Loss: {mean_loss:.4f}")
            self.log_dict({'val/mean_loss': mean_loss}, prog_bar=True, logger=True)
            self.mean_metric.reset()
            if self.args.eval_modality == "segm":
                IoU = self.iou_metric.compute()
                mIoU = torch.mean(IoU)
                MO_mIoU = torch.mean(IoU[11:])
                print("mIoU = %10f" % (mIoU*100))
                print("MO_mIoU = %10f" % (MO_mIoU*100))
                self.log_dict({"val/mIoU": mIoU * 100, "val/MO_mIoU": MO_mIoU * 100}, logger=True, prog_bar=True)
                self.iou_metric.reset()
            elif self.args.eval_modality == "depth":
                d1, d2, d3, abs_rel, rmse, log_10, rmse_log, silog, sq_rel = compute_depth_metrics(self.d1, self.d2, self.d3, self.abs_rel, self.rmse, self.log_10, self.rmse_log, self.silog, self.sq_rel)
                print("d1 =%10f" % (d1), "d2 =%10f" % (d2), "d3 =%10f" % (d3), "abs_rel =%10f" % (abs_rel), "rmse =%10f" % (rmse), "log_10 =%10f" % (log_10), "rmse_log =%10f" % (rmse_log), "silog =%10f" % (silog), "sq_rel =%10f" % (sq_rel))
                self.log_dict({"d1":d1, "d2":d2, "d3":d3, "abs_rel":abs_rel, "rmse":rmse, "log_10":log_10, "rmse_log":rmse_log, "silog":silog, "sq_rel":sq_rel}, logger=True, prog_bar=True)
                reset_depth_metrics(self.d1, self.d2, self.d3, self.abs_rel, self.rmse, self.log_10, self.rmse_log, self.silog, self.sq_rel)
            elif self.args.eval_modality == "surface_normals":
                mean_ae, median_ae, rmse, a1, a2, a3, a4, a5 = compute_normal_metrics(self.mean_ae, self.median_ae, self.rmse, self.a1, self.a2, self.a3, self.a4, self.a5)
                print("mean_ae =%10f" % (mean_ae), "median_ae =%10f" % (median_ae), "rmse =%10f" % (rmse), "a1 =%10f" % (a1), "a2 =%10f" % (a2), "a3 =%10f" % (a3), "a4 =%10f" % (a4), "a5 =%10f" % (a5))
                self.log_dict({"mean_ae":mean_ae, "median_ae":median_ae, "rmse":rmse, "a1":a1, "a2":a2, "a3":a3, "a4":a4, "a5":a5}, logger=True, prog_bar=True)
                reset_normal_metrics(self.mean_ae, self.median_ae, self.rmse, self.a1, self.a2, self.a3, self.a4, self.a5)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr, betas=(0.9, 0.999))
        assert hasattr(self.args, 'max_steps') and self.args.max_steps is not None, f"Must set max_steps argument"
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, self.args.max_steps)
        return [optimizer], [dict(scheduler=scheduler, interval='step', frequency=1)]
