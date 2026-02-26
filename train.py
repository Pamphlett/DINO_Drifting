from src.data import CS_VideoData, OpenDV_VideoData
from src.dino_f import Dino_f
import pytorch_lightning as pl
import torch 
import argparse
import os
import time
import yaml
from pytorch_lightning.strategies import DDPStrategy
import numpy as np
from pytorch_lightning.loggers import CSVLogger, WandbLogger, TensorBoardLogger
from datetime import timedelta


def parse_tuple(x):
    return tuple(map(int, x.split(',')))

def parse_list(x):
    return list(map(int, x.split(',')))


def parse_float_list(x):
    return list(map(float, x.split(',')))

def parse_val_check_interval(x):
    if x is None:
        return None
    try:
        if "." in x:
            return float(x)
        return int(x)
    except ValueError:
        return float(x)

def _log_stage(log_path, message):
    if not log_path:
        return
    try:
        log_dir = os.path.dirname(log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))
        pid = os.getpid()
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(log_path, "a") as f:
            f.write(f"{ts} rank={rank} pid={pid} {message}\n")
    except Exception:
        pass

def _configure_language_args(args):
    if args.dataset != "opendv":
        if args.use_language_condition:
            print("[Info] --use_language_condition is enabled with a non-OpenDV dataset. Language inputs may be unavailable.")
        return

    # Use language annotations for OpenDV clip indexing whenever annotation root is available.
    if args.opendv_lang_root and not args.opendv_use_lang_annos:
        print("[Info] Enabling --opendv_use_lang_annos because --opendv_lang_root is provided.")
        args.opendv_use_lang_annos = True

    if not args.use_language_condition:
        # Keep clip indexing from annos, but avoid feeding any language condition into training.
        if args.opendv_return_language:
            print("[Info] Disabling --opendv_return_language because --use_language_condition is not set.")
            args.opendv_return_language = False
        if args.opendv_use_lang_features:
            print("[Info] Disabling --opendv_use_lang_features because --use_language_condition is not set.")
            args.opendv_use_lang_features = False
        return

    if not args.opendv_lang_root:
        raise ValueError("--use_language_condition for OpenDV requires --opendv_lang_root.")
    if not args.opendv_use_lang_annos:
        print("[Info] Enabling --opendv_use_lang_annos because --use_language_condition is set.")
        args.opendv_use_lang_annos = True
    if args.use_precomputed_text:
        if not args.opendv_use_lang_features:
            print("[Info] Enabling --opendv_use_lang_features because --use_precomputed_text is set.")
            args.opendv_use_lang_features = True
    else:
        if not args.opendv_return_language:
            print("[Info] Enabling --opendv_return_language for online text encoding.")
            args.opendv_return_language = True

def main():
    # Use Tensor Cores efficiently on Ampere+ GPUs while keeping good stability.
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    parser = argparse.ArgumentParser()
    # Data Parameters
    parser.add_argument('--dataset', type=str, default='cityscapes', choices=['cityscapes', 'opendv'])
    parser.add_argument('--data_path', type=str, default='/home/ubuntu/cityscapes')
    parser.add_argument('--opendv_root', type=str, default=None, help='Root containing full_images/val_images for OpenDV-YouTube.')
    parser.add_argument('--opendv_train_root', type=str, default=None, help='Override OpenDV-YouTube train image root.')
    parser.add_argument('--opendv_val_root', type=str, default=None, help='Override OpenDV-YouTube val image root.')
    parser.add_argument('--opendv_meta_path', type=str, default=None, help='Optional OpenDV-YouTube meta JSON for faster indexing.')
    parser.add_argument('--opendv_lang_root', type=str, default=None, help='Path to OpenDV-YouTube-Language annotations.')
    parser.add_argument('--opendv_use_lang_annos', action='store_true', default=False, help='Use language annotation clips to index OpenDV-YouTube.')
    parser.add_argument('--opendv_filter_folder', type=str, default=None, help='Only keep clips whose folder contains this substring.')
    parser.add_argument('--opendv_max_clips', type=int, default=None, help='Optional cap on number of clips loaded from annotations.')
    parser.add_argument('--opendv_video_dir', type=str, default=None, help='Restrict OpenDV-YouTube to a single video directory.')
    parser.add_argument('--opendv_return_language', action='store_true', default=False, help='Return language annotations (cmd, blip) with clips.')
    parser.add_argument('--opendv_lang_cache_path', type=str, default=None, help='Optional JSON cache for language clips to speed startup.')
    parser.add_argument('--opendv_lang_cache_train', type=str, default=None, help='Optional train cache JSON for language clips.')
    parser.add_argument('--opendv_lang_cache_val', type=str, default=None, help='Optional val cache JSON for language clips.')
    parser.add_argument('--opendv_feat_ext', type=str, default='.dinov2.pt', help='Extension for precomputed feature files (replaces image extension).')
    parser.add_argument('--opendv_use_lang_features', action='store_true', default=False, help='Load precomputed CLIP text tokens from clip folders.')
    parser.add_argument('--opendv_lang_feat_name', type=str, default='lang_clip_{start}_{end}.pt', help='Filename pattern for precomputed text tokens inside each clip folder.')
    parser.add_argument('--opendv_lang_feat_key', type=str, default='text_tokens', help='Key for text tokens in precomputed feature files.')
    parser.add_argument('--opendv_lang_mask_key', type=str, default='attention_mask', help='Key for attention mask in precomputed feature files.')
    parser.add_argument('--opendv_return_future_gt', action='store_true', default=False, help='Return additional future GT frames in OpenDV eval mode.')
    parser.add_argument('--opendv_eval_future_steps', type=int, default=0, help='Number of future steps (stride=3) to return as GT in OpenDV eval mode.')
    parser.add_argument('--dst_path', type=str, default=None)
    parser.add_argument('--img_size', type=parse_tuple, default=(224,448))
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--num_workers_val', type=int, default=None, 
            help='(Optional) number of workers for the validation set dataloader. If None (default) it is the same as num_workers.')
    parser.add_argument('--dataloader_timeout', type=int, default=0,
            help='Seconds to wait for a dataloader worker batch (0 disables timeout).')
    parser.add_argument('--dataloader_log_path', type=str, default=None,
            help='Optional path to append DataLoader debug logs.')
    parser.add_argument('--dataloader_log_every', type=int, default=0,
            help='Log every N samples in the dataset (0 disables).')
    parser.add_argument('--ddp_stage_log_path', type=str, default=None,
            help='Optional path to append DDP stage logs.')
    parser.add_argument('--sequence_length', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--n_cond_frames', type=int, default=0)
    parser.add_argument('--random_crop', action='store_true', default=False)
    parser.add_argument('--random_horizontal_flip', action='store_true', default=False)
    parser.add_argument('--random_time_flip', action='store_true', default=False)
    parser.add_argument('--timestep_augm', type=list, default=None, help='Probabilities for each timestep to be selected for augmentation starting from timestep 2 to length of prob list e.g. [0.1,0.6,0.1,0.1,0.1] for timesteps [2,3,4,5,6]. If None, timestep [2,3,4] are selected with equal probability')
    parser.add_argument('--no_timestep_augm', action='store_true', help='If True, no timestep augmentation is used (i.e., the num_frames_skip is always equal to 2 during training.)')
    parser.add_argument('--use_fc_bias', action='store_true', help='Use bias for the fc_in and fc_out layers.')
    parser.add_argument('--eval_modality', type=str, default=None, choices=[None, 'segm', 'depth', 'surface_normals'], help='Modality to be used for evaluation. If None, the input modality is used.')
    # Trasformer Parameters
    parser.add_argument('--feature_extractor', type=str, default='dino', choices=['dino', 'eva2-clip', 'sam'])
    parser.add_argument('--dinov2_variant', type=str, default='vitb14_reg', choices=['vits14_reg','vitb14_reg'])
    parser.add_argument('--d_layers', type=parse_list, default=[2,5,8,11])
    parser.add_argument('--hidden_dim', type=int, default=768)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--layers', type=int, default=16)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--loss_type', type=str, default='SmoothL1',choices=['SmoothL1', 'MSE','L1'])
    parser.add_argument('--beta_smoothl1', type=float, default=0.1)
    parser.add_argument('--use_drifting_loss', action='store_true', default=False)
    parser.add_argument('--drift_temperatures', type=parse_float_list, default=[0.02, 0.05, 0.2])
    parser.add_argument('--drift_v_clip', type=float, default=50.0, help='Per-token L2 clip on drift vectors V (0 disables clipping).')
    parser.add_argument('--drift_all_gather_neg', action='store_true', default=False,
                        help='All-gather prediction tokens across DDP ranks to enlarge the negative pool for drifting.')
    parser.add_argument('--drift_adaptive_temp', action='store_true', default=False,
                        help='Adapt drift temperatures online using token pairwise-distance scale (EMA).')
    parser.add_argument('--drift_temp_ema_decay', type=float, default=0.95,
                        help='EMA decay for adaptive drift temperature scaling.')
    parser.add_argument('--drift_temp_update_interval', type=int, default=20,
                        help='Update interval (steps) for adaptive drift temperature scaling.')
    parser.add_argument('--drift_temp_token_cap', type=int, default=1024,
                        help='Max token count used to estimate pairwise distance for adaptive temperatures.')
    parser.add_argument('--drift_temp_min_scale', type=float, default=0.5,
                        help='Lower clamp for adaptive drift temperature scale.')
    parser.add_argument('--drift_temp_max_scale', type=float, default=2.0,
                        help='Upper clamp for adaptive drift temperature scale.')
    parser.add_argument('--drift_temp_ref_dist', type=float, default=0.0,
                        help='Reference pairwise distance for adaptive temperatures. <=0 means auto from first update.')
    parser.add_argument('--drift_log_interval', type=int, default=20)
    parser.add_argument('--drift_antisymmetry_interval', type=int, default=200)
    parser.add_argument('--drift_metric_token_cap', type=int, default=512)
    parser.add_argument('--drift_train_token_cap', type=int, default=0,
                        help='Optional cap on token count used by drifting loss cdist/softmax (0 disables subsampling).')
    parser.add_argument('--drift_diversity_k', type=int, default=3)
    parser.add_argument('--drift_noise_dim', type=int, default=256,
                        help='Dimension of random style noise z for drifting conditioning.')
    parser.add_argument('--attn_dropout', type=float, default=0.3)
    parser.add_argument('--step', type=int, default=2)
    parser.add_argument('--masking', type=str, default='half_half', choices=['half_half', 'simple_replace', 'half_half_previous'])
    parser.add_argument('--train_mask_mode', type=str, default='arccos', choices=['full_mask', 'arccos', 'linear', 'cosine', 'square'])
    parser.add_argument('--seperable_attention', action='store_true', default=False)
    parser.add_argument('--seperable_window_size', type=int, default=1)
    parser.add_argument('--train_mask_frames', type=int, default=1)
    parser.add_argument('--output_activation', type=str, default='none', choices=['none', 'sigmoid'])
    parser.add_argument('--use_first_last', action='store_true', default=False)
    parser.add_argument('--down_up_sample', action='store_true', default=False)
    parser.add_argument('--pca_ckpt', type=str, default=None)
    parser.add_argument('--pca_whiten_alpha', type=float, default=0.0,
                        help='Alpha-whitening strength in PCA space: 0=no whitening, 1=full whitening.')
    parser.add_argument('--pca_whiten_eps', type=float, default=1e-8,
                        help='Numerical epsilon for PCA alpha-whitening.')
    parser.add_argument('--crop_feats', action='store_true', default=False)
    parser.add_argument('--sliding_window_inference', action='store_true', default=False)
    parser.add_argument('--high_res_adapt', action='store_true', default=False, help='If True, the input images are resized to 448x896 instead of 224x448')
    # DPT Head Parameters
    parser.add_argument('--num_classes', type=int, default=19, choices=[19, 256, 3], help="19 Classes for segmentation, 256(classification) Depth or 3(regression) for surface normals")
    parser.add_argument('--use_bn', action='store_true', default=False)
    parser.add_argument('--use_cls', action='store_true', default=False)
    parser.add_argument('--nfeats', type=int, default=256)
    parser.add_argument('--dpt_out_channels', type=parse_list, default=[128, 256, 512, 512])
    parser.add_argument('--head_ckpt', type=str, default=None)
    # training parameters
    parser.add_argument('--max_epochs', type=int, default=800) # 50220//372 == 135 epochs
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--single_step_sample_train', action='store_true', default=False)
    parser.add_argument('--precision', type=str, default='32-true',choices=['16-true','16-mixed','32-true', '32'])
    parser.add_argument('--ckpt', type=str, default=None, help='Path of a checkpoint to resume training')
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--accum_iter', type=int, default=1)
    parser.add_argument('--warmup_p', type=float, default=0.0)
    parser.add_argument('--lr_base', type=float, default=1e-3)
    parser.add_argument('--gclip', type=float, default=2.0)
    parser.add_argument('--evaluate', action='store_true', default=False)
    parser.add_argument('--vis_attn', action='store_true', default=False)
    parser.add_argument('--eval_last', action='store_true', default=False)
    parser.add_argument('--eval_ckpt_only', action='store_true', default=False)
    parser.add_argument('--eval_mode_during_training', action='store_true', help='if activated (True) it uses the evaluation mode (i.e., step-by-step prediction and mIoU computation) during the training loop')
    parser.add_argument('--eval_freq', type=int, default=1)
    parser.add_argument('--use_val_to_train', action='store_true', default=False)
    parser.add_argument('--use_train_to_val', action='store_true', default=False)
    parser.add_argument('--evaluate_baseline', action='store_true', default=False)
    parser.add_argument('--eval_midterm', action='store_true', default=False)
    parser.add_argument('--wandb_project', type=str, default='dino-foresight')
    parser.add_argument('--wandb_entity', type=str, default=None)
    parser.add_argument('--wandb_name', type=str, default=None)
    parser.add_argument('--wandb_mode', type=str, default='online', choices=['online', 'offline', 'disabled'])
    parser.add_argument('--use_tensorboard', action='store_true', default=False, help='Enable TensorBoard logger.')
    parser.add_argument('--tensorboard_name', type=str, default='tensorboard', help='Subfolder name for TensorBoard logs.')
    parser.add_argument('--use_csv_logger', action='store_true', default=False, help='Enable CSV logger.')
    parser.add_argument('--csv_logger_name', type=str, default='csv', help='Subfolder name for CSV logs.')
    parser.add_argument('--strategy', type=str, default='auto', choices=['auto', 'ddp', 'ddp_spawn'])
    parser.add_argument('--find_unused_parameters', action='store_true', default=False,
                        help='Enable DDP find_unused_parameters to avoid hangs when some params are unused.')
    parser.add_argument('--ddp_timeout', type=int, default=1800, help='DDP timeout in seconds.')
    parser.add_argument('--use_language_condition', action='store_true', default=False)
    parser.add_argument('--use_precomputed_text', action='store_true', default=False)
    parser.add_argument('--use_precomputed_feats', action='store_true', default=False)
    parser.add_argument('--clip_model_name', type=str, default='openai/clip-vit-base-patch32')
    parser.add_argument('--clip_max_length', type=int, default=77)
    parser.add_argument('--clip_cache_dir', type=str, default=None)
    parser.add_argument('--clip_local_files_only', action='store_true', default=False)
    parser.add_argument('--clip_text_dim', type=int, default=512)
    parser.add_argument('--num_sanity_val_steps', type=int, default=2)
    parser.add_argument('--limit_val_batches', type=float, default=1.0)
    parser.add_argument('--save_every_n_steps', type=int, default=0, help='Save checkpoints every N training steps (0 disables).')
    parser.add_argument('--val_check_interval', type=parse_val_check_interval, default=None, help='Run val every N train batches (int) or fraction of epoch (float).')

    args = parser.parse_args()
    _configure_language_args(args)
    _log_stage(args.ddp_stage_log_path, "args_parsed")

    args.eval_mode = args.eval_mode_during_training
    pl.seed_everything(args.seed, workers=True)

    if args.dataset == 'opendv':
        data = OpenDV_VideoData(arguments=args, subset='train', batch_size=args.batch_size)
    else:
        data = CS_VideoData(arguments=args, subset='train', batch_size=args.batch_size)
    _log_stage(args.ddp_stage_log_path, "datamodule_ready")


    if args.precision == '32':
        args.precision = 32

    if args.num_gpus > 1:
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            args.rank = int(os.environ['RANK'])
            args.world_size = int(os.environ['WORLD_SIZE'])
            args.gpu = int(os.environ['LOCAL_RANK'])
        elif 'SLURM_PROCID' in os.environ:
            args.rank = int(os.environ['SLURM_PROCID'])
            args.world_size = int(os.environ['SLURM_NTASKS'])
            args.gpu = int(os.environ['SLURM_LOCALID'])
            gpus_per_node = int(os.environ['SLURM_GPUS_ON_NODE'])
            assert gpus_per_node == torch.cuda.device_count()
            args.node = args.rank // gpus_per_node
        else:
            args.rank = 0
            args.world_size = args.num_gpus
            args.gpu = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        args.device = torch.device(args.gpu)
    else:
        args.rank = 0
        args.world_size = 1
        args.gpu = 0
        args.node = 0
        args.device = torch.device('cuda:'+str(args.gpu))

    print(f'rank={args.rank} - world_size={args.world_size} - gpu={args.gpu} - device={args.device}')
    args.max_steps = (args.max_epochs * (len(data.train_dataloader()) // (args.num_gpus * args.accum_iter)))
    args.warmup_steps = int(args.warmup_p * args.max_steps)
    args.effective_batch_size = args.batch_size * args.world_size * args.accum_iter
    args.lr = (args.lr_base * args.effective_batch_size) / 8 # args.lr_base is specified for an effective batch-size of 8
    print(f'Effective batch size:{args.effective_batch_size} lr_base={args.lr_base} lr={args.lr} max_epochs={args.max_epochs} - max_steps={args.max_steps}')

    if not args.high_res_adapt:
        Dino_foresight = Dino_f(args)
    else:
        Dino_foresight = Dino_f.load_from_checkpoint(args.ckpt,args=args,strict=False, map_location="cpu")
    _log_stage(args.ddp_stage_log_path, "model_ready")

    if args.dst_path is None:
        args.dst_path = os.getcwd()
    ckpt_dir = os.path.join(args.dst_path, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    callbacks = []
    epoch_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=ckpt_dir,
        save_top_k=-1,
        every_n_epochs=1,
        save_on_train_epoch_end=True,
        filename="epoch-{epoch:04d}-step-{step}",
        auto_insert_metric_name=False,
    )
    callbacks.append(epoch_checkpoint_callback)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=ckpt_dir,
        monitor='val/loss',
        mode='min',
        save_top_k=1,
        save_last=True,
        filename="best-epoch-{epoch:04d}-step-{step}",
        auto_insert_metric_name=False,
    )
    callbacks.append(checkpoint_callback)
    if args.save_every_n_steps and args.save_every_n_steps > 0:
        step_ckpt = pl.callbacks.ModelCheckpoint(
            dirpath=ckpt_dir,
            save_top_k=-1,
            every_n_train_steps=args.save_every_n_steps,
            filename="step-{step}",
        )
        callbacks.append(step_ckpt)
    if args.max_epochs < args.eval_freq:
        args.eval_freq = 1
    logger_list = []
    if args.rank == 0 and args.use_tensorboard:
        tb_logger = TensorBoardLogger(
            save_dir=args.dst_path,
            name=args.tensorboard_name,
            default_hp_metric=False,
        )
        logger_list.append(tb_logger)
    if args.rank == 0 and args.use_csv_logger:
        csv_logger = CSVLogger(
            save_dir=args.dst_path,
            name=args.csv_logger_name,
        )
        logger_list.append(csv_logger)
    if args.wandb_mode != "disabled" and args.rank == 0:
        wandb_logger = WandbLogger(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            mode=args.wandb_mode,
        )
        logger_list.append(wandb_logger)
    if len(logger_list) == 0:
        active_logger = False
    elif len(logger_list) == 1:
        active_logger = logger_list[0]
    else:
        active_logger = logger_list
    ddp_timeout = timedelta(seconds=args.ddp_timeout)
    if args.strategy == "auto":
        strategy = (
            DDPStrategy(find_unused_parameters=args.find_unused_parameters, timeout=ddp_timeout)
            if args.num_gpus > 1
            else "auto"
        )
    elif args.strategy == "ddp":
        strategy = DDPStrategy(find_unused_parameters=args.find_unused_parameters, timeout=ddp_timeout)
    else:
        strategy = "ddp_spawn"
    trainer_kwargs = dict(
        accelerator='gpu',
        strategy=strategy,
        devices=args.num_gpus,
        callbacks=callbacks,
        max_epochs=args.max_epochs,
        gradient_clip_val=args.gclip,
        default_root_dir=args.dst_path,
        precision=args.precision,
        logger=active_logger,
        log_every_n_steps=5,
        check_val_every_n_epoch=args.eval_freq,
        num_sanity_val_steps=args.num_sanity_val_steps,
        limit_val_batches=args.limit_val_batches,
        accumulate_grad_batches=args.accum_iter,
    )
    if args.val_check_interval is not None:
        trainer_kwargs["val_check_interval"] = args.val_check_interval
    trainer = pl.Trainer(**trainer_kwargs)
    _log_stage(args.ddp_stage_log_path, "trainer_created")

    if not args.eval_ckpt_only:
       if args.ckpt and not args.high_res_adapt:
           _log_stage(args.ddp_stage_log_path, "fit_start")
           trainer.fit(Dino_foresight,data,ckpt_path=args.ckpt)
       else:
           _log_stage(args.ddp_stage_log_path, "fit_start")
           trainer.fit(Dino_foresight,data)
    else:
        args.evaluate = True
        args.eval_last = True
        args.eval_mode = True
        checkpoint_callback.last_model_path = args.ckpt
        if args.dataset == 'opendv':
            data = OpenDV_VideoData(arguments=args, subset='train', batch_size=args.batch_size)
        else:
            data = CS_VideoData(arguments=args, subset='train', batch_size=args.batch_size)

    # Evaluation
    if args.evaluate:
        args.eval_mode = True
        if args.dataset == 'opendv':
            data = OpenDV_VideoData(arguments=args, subset='train', batch_size=args.batch_size)
        else:
            data = CS_VideoData(arguments=args, subset='train', batch_size=args.batch_size)
        if not args.eval_last:
            print('Loading best model')
            checkpoint_path = checkpoint_callback.best_model_path
            if not checkpoint_path:
                print("Best checkpoint not found. Falling back to last checkpoint.")
                checkpoint_path = checkpoint_callback.last_model_path
        else:
            print('Loading last model')
            checkpoint_path = checkpoint_callback.last_model_path

        print(f'checkpoint_path = {checkpoint_path}')
        Dino_foresight = Dino_f.load_from_checkpoint(checkpoint_path,args=args,strict=False, map_location="cpu")
        print('-----------Dino_foresight.eval_mode = ',Dino_foresight.args.eval_mode)
        Dino_foresight.to(args.device)
        Dino_foresight.eval()

        val_data_loader = data.val_dataloader()
        out_metrics = trainer.validate(model=Dino_foresight, dataloaders=val_data_loader)
        loss = out_metrics[0]['val/mean_loss']
        if args.eval_modality is None:
            feat_mse = out_metrics[0].get("val/feat_mse", None)
            feat_l1 = out_metrics[0].get("val/feat_l1", None)
            feat_cos = out_metrics[0].get("val/feat_cos", None)
            feat_recall1 = out_metrics[0].get("val/feat_recall1", None)
            if args.rank==0:
                result_path = os.path.join(trainer.log_dir,'results.txt')
                with open(result_path,'w') as f:
                    f.write(f'Mean Loss: {loss}\n')
                    if feat_mse is not None:
                        f.write(f'feat_mse: {feat_mse}\n')
                    if feat_l1 is not None:
                        f.write(f'feat_l1: {feat_l1}\n')
                    if feat_cos is not None:
                        f.write(f'feat_cos: {feat_cos}\n')
                    if feat_recall1 is not None:
                        f.write(f'feat_recall1: {feat_recall1}\n')
        if args.eval_modality=='segm':
            mIoU = out_metrics[0]['val/mIoU']
            MO_mIoU = out_metrics[0]['val/MO_mIoU']
            if args.rank==0:
                result_path = os.path.join(trainer.log_dir,'results.txt')
                with open(result_path,'w') as f:
                    f.write(f'Mean Loss: {loss}\n')
                    f.write(f'mIoU: {mIoU}\n')
                    f.write(f'MO_mIoU: {MO_mIoU}\n')
                print(f'Results saved in {result_path}')
        elif args.eval_modality == 'depth':
            d1 = out_metrics[0]["d1"]
            d2 = out_metrics[0]["d2"]
            d3 = out_metrics[0]["d3"]
            abs_rel = out_metrics[0]["abs_rel"]
            rmse = out_metrics[0]["rmse"]
            rmse_log = out_metrics[0]["rmse_log"]
            silog = out_metrics[0]["silog"]
            sq_rel = out_metrics[0]["sq_rel"]
            log_10 = out_metrics[0]["log_10"]
            if args.rank == 0:
                # Save d1 to a text file
                result_path = os.path.join(trainer.log_dir, 'results.txt')
                with open(result_path, 'w') as f:
                    f.write(f'Mean Loss: {loss}\n')
                    f.write(f'd1: {d1}\n')
                    f.write(f'd2: {d2}\n')
                    f.write(f'd3: {d3}\n')
                    f.write(f'abs_rel: {abs_rel}\n')
                    f.write(f'rmse: {rmse}\n')
                    f.write(f'rmse_log: {rmse_log}\n')
                    f.write(f'sq_rel: {sq_rel}\n')
                    f.write(f'log_10: {log_10}\n')
                    f.write(f'silog: {silog}\n')
                print(f'Results saved at: {result_path}')
        elif args.eval_modality == 'surface_normals':
            mean_ae = out_metrics[0]["mean_ae"]
            median_ae = out_metrics[0]["median_ae"]
            rmse = out_metrics[0]["rmse"]
            a1 = out_metrics[0]["a1"]
            a2 = out_metrics[0]["a2"]
            a3 = out_metrics[0]["a3"]
            a4 = out_metrics[0]["a4"]
            a5 = out_metrics[0]["a5"]
            if args.rank == 0:
                # Save d1 to a text file
                result_path = os.path.join(trainer.log_dir, 'results.txt')
                with open(result_path, 'w') as f:
                    f.write(f'Mean Loss: {loss}\n')
                    f.write(f'mean_ae: {mean_ae}\n')
                    f.write(f'median_ae: {median_ae}\n')
                    f.write(f'rmse: {rmse}\n')
                    f.write(f'a1: {a1}\n')
                    f.write(f'a2: {a2}\n')
                    f.write(f'a3: {a3}\n')
                    f.write(f'a4: {a4}\n')
                    f.write(f'a5: {a5}\n')
                print(f'Results saved at: {result_path}')


if __name__ == '__main__':
    main()
