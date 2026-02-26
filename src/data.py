import os
import os.path as osp
import json
import glob
import time
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from torch.utils.data._utils.collate import collate, default_collate_fn_map
import pytorch_lightning as pl
from PIL import Image

IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def _stack_tensor_no_shared(batch, collate_fn_map=None):
    return torch.stack(batch, 0)


_OPENDV_COLLATE_FN_MAP = dict(default_collate_fn_map)
_OPENDV_COLLATE_FN_MAP[torch.Tensor] = _stack_tensor_no_shared


def opendv_collate(batch):
    return collate(batch, collate_fn_map=_OPENDV_COLLATE_FN_MAP)

def _is_image_file(filename):
    return filename.lower().endswith(IMAGE_EXTS)

def _youtuber_formatize(youtuber):
    return youtuber.replace(" ", "_")

def _iter_json_array(path, chunk_size=1024 * 1024):
    decoder = json.JSONDecoder()
    with open(path, "r") as f:
        buf = ""
        pos = 0
        while True:
            if pos >= len(buf):
                chunk = f.read(chunk_size)
                if not chunk:
                    return
                buf += chunk
            while pos < len(buf) and buf[pos].isspace():
                pos += 1
            if pos < len(buf):
                if buf[pos] != "[":
                    raise ValueError(f"Expected '[' at start of JSON array in {path}")
                pos += 1
                break
        while True:
            while True:
                if pos >= len(buf):
                    chunk = f.read(chunk_size)
                    if not chunk:
                        return
                    buf = buf[pos:] + chunk
                    pos = 0
                if buf[pos].isspace() or buf[pos] == ",":
                    pos += 1
                elif buf[pos] == "]":
                    return
                else:
                    break
            while True:
                try:
                    obj, next_pos = decoder.raw_decode(buf, pos)
                    pos = next_pos
                    yield obj
                    break
                except json.JSONDecodeError:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        raise
                    buf += chunk


def _select_train_frame_paths(frames_path, subset, augmentations, sequence_length=5, num_frames_skip=0):
    if subset == "val":
        num_frames_skip = 2
        step = num_frames_skip + 1
        start_idx = 20 - step * sequence_length + num_frames_skip
        max_start = len(frames_path) - step * sequence_length
        if max_start < 0:
            max_start = 0
        if start_idx < 0:
            start_idx = 0
        elif start_idx > max_start:
            start_idx = max_start
    else:
        if augmentations["no_timestep_augm"] is True:
            num_frames_skip = 2
        elif augmentations["timestep_augm"] is not None:
            num_frames_skip = np.random.choice(
                list(range(1, len(augmentations["timestep_augm"]) + 1)),
                p=augmentations["timestep_augm"],
            )
        else:
            num_frames_skip = np.random.randint(1, 4)  # [1,3] with equal probabilities 4 is excluded
        step = num_frames_skip + 1
        start_idx = np.random.randint(0, len(frames_path) - step * sequence_length + num_frames_skip + 1)
    sequence_frames_path = frames_path[start_idx:start_idx + step * sequence_length:step]
    if augmentations["random_time_flip"] is True and subset == "train":
        sequence_frames_path = sequence_frames_path[::-1] if np.random.rand() > 0.5 else sequence_frames_path
    return sequence_frames_path

class CityScapesRGBDataset(data.Dataset):
    """
    A custom dataset class for CityScapes segmentation dataset.
    Args:

        data_path (str): The path to the data folder.
        sequence_length (int): The length of each sequence.
        shape (tuple): The desired shape of the frames.
        downsample_factor (int): The factor by which the frames should be downsampled.
        subset (str, optional): The subset of the dataset to use (train, val, test). Defaults to "train".

    Attributes:
        data_path (str): The path to the data folder.
        sequence_length (int): The length of each sequence.
        shape (tuple): The desired shape of the frames.
        downsample_factor (int): The factor by which the frames should be downsampled.
        subset (str): The subset of the dataset being used.
        num_frames (int): The total number of frames in the dataset.
        sequences (list): The list of unique sequence names in the dataset.
        
    Methods:
        __len__(): Returns the number of sequences in the dataset.
        __getitem__(idx): Retrieves the frames and their corresponding file paths for a given index.
    """

    def __init__(self, data_path, args, sequence_length, img_size, subset="train", eval_mode=False, eval_midterm=False, eval_modality=None, feature_extractor="dino"):
        super().__init__()
        self.data_path = data_path
        self.sequence_length = sequence_length
        self.img_size = img_size
        self.subset = subset 
        self.eval_mode = eval_mode
        self.eval_midterm = eval_midterm
        self.eval_modality = args.eval_modality
        self.feature_extractor = feature_extractor
        self.return_rgb_path = getattr(args, "return_rgb_path", False)
        self.num_frames = len(glob.glob(osp.join(self.data_path, subset, '**',"*.png")))
        self.sequences = set() # Each sequence name consists of city name and sequence id i.e. aachen_00001, hanover_00013
        self.augmentations = {
                "random_crop" : args.random_crop,
                "random_horizontal_flip" : args.random_horizontal_flip,
                "random_time_flip" : args.random_time_flip,
                "timestep_augm" : args.timestep_augm,
                "no_timestep_augm" : args.no_timestep_augm}
        for city_folder in glob.glob(osp.join(self.data_path, subset, '*')):
            city_name = osp.basename(city_folder)
            frames_in_city = glob.glob(osp.join(city_folder, '*'))
            city_seqs = set([f"{city_name}_{osp.basename(frame).split('_')[1]}" for frame in frames_in_city])
            if len(city_seqs)<10: # Note that in some cities very few, though very long sequences were recorded
                for seq in city_seqs:
                    sub_seqs = sorted(glob.glob(osp.join(self.data_path, subset, city_name, seq+'*.png')))
                    sub_seq_startframe_ids = [osp.basename(sub_seqs[i])[:-16] for i in range(len(sub_seqs)) if i%30==0]
                    self.sequences.update(sub_seq_startframe_ids)
            else:
                self.sequences.update(city_seqs)
                continue
        self.sequences = sorted(list(self.sequences))

    def __len__(self):
        """
        Returns the number of sequences in the dataset.

        Returns:
            int: The number of sequences.
        """
        return len(self.sequences)

    def __getitem__(self, idx):
        """
        Retrieves the frames and their corresponding file paths for a given index.

        Args:
            idx (int): The index of the sequence.

        Returns:
            tuple: A tuple containing the frames and their corresponding file paths.
        """
        sequence_name = self.sequences[idx]
        splits = sequence_name.split("_")
        if len(splits)==2: # Sample from Short sequences
            frames_filepaths = sorted(glob.glob(osp.join(self.data_path, self.subset, splits[0], sequence_name+'*.png'))) # Sequence of 30 frames    
        elif len(splits)==3: # Sample from Long sequences
            frames_filepaths = [osp.join(self.data_path, self.subset,splits[0], splits[0]+"_"+splits[1]+"_"+'{:06d}'.format(int(splits[2])+i)+'_leftImg8bit.png') for i in range(30)]
        # Load, process and Stack to Tensor
        if self.eval_mode:
            if self.eval_modality is None:
                if self.return_rgb_path:
                    frames, gt, gt_path = process_evalmode(
                        frames_filepaths,
                        self.img_size,
                        self.subset,
                        self.sequence_length,
                        self.eval_midterm,
                        feature_extractor=self.feature_extractor,
                        return_paths=True,
                    )
                    return frames, gt, gt_path
                frames, gt = process_evalmode(
                    frames_filepaths,
                    self.img_size,
                    self.subset,
                    self.sequence_length,
                    self.eval_midterm,
                    feature_extractor=self.feature_extractor,
                )
                return frames, gt
            elif self.eval_modality=="segm":
                if self.return_rgb_path:
                    frames, gt, gt_segm, gt_path = process_evalmode(
                        frames_filepaths,
                        self.img_size,
                        self.subset,
                        self.sequence_length,
                        self.eval_midterm,
                        self.eval_modality,
                        feature_extractor=self.feature_extractor,
                        return_paths=True,
                    )
                    return frames, gt, gt_segm, gt_path
                frames, gt, gt_segm = process_evalmode(
                    frames_filepaths,
                    self.img_size,
                    self.subset,
                    self.sequence_length,
                    self.eval_midterm,
                    self.eval_modality,
                    feature_extractor=self.feature_extractor,
                )
                return frames, gt, gt_segm
            elif self.eval_modality=="depth":
                if self.return_rgb_path:
                    frames, gt, gt_depth, gt_path = process_evalmode(
                        frames_filepaths,
                        self.img_size,
                        self.subset,
                        self.sequence_length,
                        self.eval_midterm,
                        self.eval_modality,
                        feature_extractor=self.feature_extractor,
                        return_paths=True,
                    )
                    return frames, gt, gt_depth, gt_path
                frames, gt, gt_depth = process_evalmode(
                    frames_filepaths,
                    self.img_size,
                    self.subset,
                    self.sequence_length,
                    self.eval_midterm,
                    self.eval_modality,
                    feature_extractor=self.feature_extractor,
                )
                return frames, gt, gt_depth
            elif self.eval_modality=="surface_normals":
                if self.return_rgb_path:
                    frames, gt, gt_normals, gt_path = process_evalmode(
                        frames_filepaths,
                        self.img_size,
                        self.subset,
                        self.sequence_length,
                        self.eval_midterm,
                        self.eval_modality,
                        feature_extractor=self.feature_extractor,
                        return_paths=True,
                    )
                    return frames, gt, gt_normals, gt_path
                frames, gt, gt_normals = process_evalmode(
                    frames_filepaths,
                    self.img_size,
                    self.subset,
                    self.sequence_length,
                    self.eval_midterm,
                    self.eval_modality,
                    feature_extractor=self.feature_extractor,
                )
                return frames, gt, gt_normals
        else:
            if self.return_rgb_path:
                frames, frame_paths = process_trainmode(
                    frames_filepaths,
                    self.img_size,
                    self.subset,
                    self.augmentations,
                    self.sequence_length,
                    feature_extractor=self.feature_extractor,
                    return_paths=True,
                )
                return frames, frame_paths[-1]
            frames = process_trainmode(
                frames_filepaths,
                self.img_size,
                self.subset,
                self.augmentations,
                self.sequence_length,
                feature_extractor=self.feature_extractor,
            )
            return frames

       
def process_trainmode(frames_path, img_size, subset, augmentations, sequence_length=5, num_frames_skip=0, feature_extractor="dino", return_paths=False):
    sequence_frames_path = _select_train_frame_paths(
        frames_path,
        subset,
        augmentations,
        sequence_length=sequence_length,
        num_frames_skip=num_frames_skip,
    )
    # Load frames as tensors and apply transformations
    sequence_frames = [Image.open(frame).convert('RGB') for frame in sequence_frames_path]
    W, H = sequence_frames[0].size # PIL IMAGE
    if augmentations["random_crop"] == True and subset=="train":
        s_f = np.random.rand()/2 + 0.5 # [0.5, 1]
        size = (int(H*s_f),int(W*s_f))
        i, j, h, w = T.RandomCrop(size).get_params(sequence_frames[0], output_size=size)
        sequence_frames = [TF.crop(frame,i,j,h,w) for frame in sequence_frames]
    if augmentations["random_horizontal_flip"] == True and subset=="train":
        sequence_frames = [TF.hflip(frame) for frame in sequence_frames] if np.random.rand()>0.5 else sequence_frames
    if feature_extractor in ['dino', 'sam']:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    elif feature_extractor == 'eva2-clip':
        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)
    transform = T.Compose([T.Resize(img_size),T.ToTensor(),T.Normalize(mean=mean, std=std)])
    sequence_tensors = [transform(frame) for frame in sequence_frames]
    sequence_tensor = torch.stack(sequence_tensors, dim=0)
    if return_paths:
        return sequence_tensor, sequence_frames_path
    return sequence_tensor

def process_evalmode(frames_path, img_size, subset, sequence_length=5, eval_midterm=False, eval_modality=None, feature_extractor="dino", return_paths=False):
    num_frames_skip = 2 
    step = num_frames_skip + 1  
    if eval_midterm and sequence_length<7:
        start_idx = 20 - step*sequence_length + num_frames_skip - 6
    else:
        start_idx = 20 - step*sequence_length + num_frames_skip
    sequence_frames_path = frames_path[start_idx : start_idx + step*sequence_length : step]
    sequence_frames = [Image.open(frame).convert('RGB') for frame in sequence_frames_path]
    if feature_extractor in ['dino', 'sam']:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    elif feature_extractor == 'eva2-clip':
        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)
    transform = T.Compose([T.Resize(img_size),T.ToTensor(),T.Normalize(mean=mean, std=std)])
    sequence_tensors = [transform(frame) for frame in sequence_frames]
    sequence_tensor = torch.stack(sequence_tensors, dim=0).clone()
    gt_path = frames_path[19]
    gt_img = transform(Image.open(gt_path)).clone()
    if eval_modality is None:
        if return_paths:
            return sequence_tensor, gt_img, gt_path
        return sequence_tensor, gt_img
    elif eval_modality=="segm":
        gt_segm_path = gt_path.replace("leftImg8bit_sequence","gtFine").replace("leftImg8bit","gtFine_labelTrainIds")
        gt_segm_img = Image.open(gt_segm_path)
        transform_segmap = T.PILToTensor()
        gt_segm_img = transform_segmap(gt_segm_img).clone()
        if return_paths:
            return sequence_tensor, gt_img, gt_segm_img, gt_path
        return sequence_tensor, gt_img, gt_segm_img
    elif eval_modality=="depth":
        gt_depth_path = gt_path.replace("leftImg8bit_sequence","leftImg8bit_sequence_depthv2").replace("leftImg8bit.png","leftImg8bit_depth.png")
        gt_depth_img = Image.open(gt_depth_path)
        transform_depthmap = T.PILToTensor()
        gt_depth_img = transform_depthmap(gt_depth_img).clone()
        if return_paths:
            return sequence_tensor, gt_img, gt_depth_img, gt_path
        return sequence_tensor, gt_img, gt_depth_img
    elif eval_modality=="surface_normals":
        gt_normals_path = gt_path.replace("leftImg8bit_sequence","leftImg8bit_normals")
        gt_normals_img = np.load(gt_normals_path.replace("png","npy"))
        gt_normals_img = torch.from_numpy(gt_normals_img).permute(2, 0, 1).clone()
        if return_paths:
            return sequence_tensor, gt_img, gt_normals_img, gt_path
        return sequence_tensor, gt_img, gt_normals_img


def process_opendv_evalmode(
    frames_path,
    img_size,
    sequence_length=5,
    feature_extractor="dino",
    return_paths=False,
    return_future_gt=False,
    future_steps=0,
):
    num_frames_skip = 2
    step = num_frames_skip + 1
    future_steps = max(0, int(future_steps))
    extra_steps = future_steps if return_future_gt else 0
    min_frames = step * (sequence_length + extra_steps)
    if len(frames_path) < min_frames:
        raise ValueError("Not enough frames to build an evaluation clip.")
    start_idx = len(frames_path) - min_frames
    sequence_frames_path = frames_path[start_idx:start_idx + step * sequence_length:step]
    sequence_frames = [Image.open(frame).convert('RGB') for frame in sequence_frames_path]
    if feature_extractor in ['dino', 'sam']:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    elif feature_extractor == 'eva2-clip':
        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)
    transform = T.Compose([T.Resize(img_size), T.ToTensor(), T.Normalize(mean=mean, std=std)])
    sequence_tensors = [transform(frame) for frame in sequence_frames]
    sequence_tensor = torch.stack(sequence_tensors, dim=0)
    gt_path = sequence_frames_path[-1]
    gt_img = transform(Image.open(gt_path))
    if return_future_gt and future_steps > 0:
        future_start = start_idx + step * sequence_length
        future_end = start_idx + step * (sequence_length + future_steps)
        future_paths = frames_path[future_start:future_end:step]
        if len(future_paths) != future_steps:
            raise ValueError(
                f"Insufficient future frames for rollout GT: expected {future_steps}, got {len(future_paths)}."
            )
        future_imgs = [transform(Image.open(frame).convert("RGB")) for frame in future_paths]
        future_gt = torch.stack(future_imgs, dim=0)
        if return_paths:
            return sequence_tensor, gt_img, future_gt, gt_path
        return sequence_tensor, gt_img, future_gt
    if return_paths:
        return sequence_tensor, gt_img, gt_path
    return sequence_tensor, gt_img
        

class CS_VideoData(pl.LightningDataModule):
    """
    LightningDataModule for loading CityScapes video data.

    Args:
        arguments: An object containing the required arguments for data loading.
        subset (str): The subset of the data to load. Default is "train".

    Attributes:
        data_path (str): The path to the data folder.
        subset (str): The subset of the data being loaded.
        sequence_length (int): The length of the video sequence.
        batch_size (int): The batch size for data loading.
        shape (tuple): The shape of the video frames.
        downsample_factor (int): The factor by which to downsample the frames.
    """

    def __init__(self, arguments, subset="train", batch_size=8):
        super().__init__()
        self.data_path = arguments.data_path
        self.subset = subset  # ["train","val","test"]
        self.sequence_length = arguments.sequence_length
        self.batch_size = batch_size
        self.img_size = arguments.img_size
        # assert self.img_size[0]%14==0 and self.img_size[1]%14==0, "Image size should be divisible by 14"
        self.arguments = arguments
        self.eval_midterm = arguments.eval_midterm
        self.eval_modality = arguments.eval_modality
        self.num_workers = arguments.num_workers
        self.num_workers_val = arguments.num_workers if arguments.num_workers_val is None else arguments.num_workers_val
        self.dataloader_timeout = getattr(arguments, "dataloader_timeout", 0)
        self.eval_mode = arguments.eval_mode
        self.use_val_to_train = arguments.use_val_to_train
        self.use_train_to_val = arguments.use_train_to_val
        self.feature_extractor = arguments.feature_extractor


    def _dataset(self, subset, eval_mode):
        """
        Private method to create and return the CityScapesDataset object.

        Args:
            subset (str): The subset of the data to load.

        Returns:
            CityScapesDataset: The dataset object.
        """
        dataset = CityScapesRGBDataset(self.data_path, self.arguments, self.sequence_length, self.img_size, subset, eval_mode, self.eval_midterm, self.eval_modality, self.feature_extractor)
        return dataset

    def _dataloader(self, subset, shuffle=True, drop_last=False, eval_mode=False):
        """
        Private method to create and return the DataLoader object.

        Args:
            subset (str): The subset of the data to load.
            shuffle (bool): Whether to shuffle the data. Default is True.

        Returns:
            DataLoader: The dataloader object.
        """
        dataset = self._dataset(subset, eval_mode)
        dataloader = data.DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=shuffle, drop_last=drop_last)
        return dataloader

    def train_dataloader(self):
        """
        Method to return the dataloader for the training subset.

        Returns:
            DataLoader: The dataloader object for training data.
        """
        train_subset = "val" if self.use_val_to_train else "train"
        dataset = self._dataset(train_subset, eval_mode=False)
        world_size = int(getattr(self.arguments, "num_gpus", 1) or 1)
        drop_last = len(dataset) >= self.batch_size * world_size
        return data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            timeout=self.dataloader_timeout,
            shuffle=True,
            drop_last=drop_last,
        )

    def val_dataloader(self):
        """
        Method to return the dataloader for the validation subset.

        Returns:
            DataLoader: The dataloader object for validation data.
        """
        val_subset = "train" if self.use_train_to_val else "val"
        dataset = self._dataset(val_subset, self.eval_mode)
        dataloader = data.DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers_val, pin_memory=True, shuffle=False, drop_last=False)
        return dataloader

    def test_dataloader(self):
        """
        Method to return the dataloader for the test subset.

        Returns:
            DataLoader: The dataloader object for test data.
        """
        return self._dataloader(subset="test")


class OpenDVYouTubeDataset(data.Dataset):
    def __init__(self, train_root, val_root, args, sequence_length, img_size, subset="train", eval_mode=False, feature_extractor="dino", meta_path=None):
        super().__init__()
        self.train_root = train_root
        self.val_root = val_root
        self.sequence_length = sequence_length
        self.img_size = img_size
        self.subset = subset
        self.eval_mode = eval_mode
        self.feature_extractor = feature_extractor
        self.eval_modality = args.eval_modality
        self.augmentations = {
            "random_crop": args.random_crop,
            "random_horizontal_flip": args.random_horizontal_flip,
            "random_time_flip": args.random_time_flip,
            "timestep_augm": args.timestep_augm,
            "no_timestep_augm": args.no_timestep_augm,
        }
        self.meta_path = meta_path
        self.lang_root = args.opendv_lang_root
        self.lang_use_annos = args.opendv_use_lang_annos
        self.lang_filter = args.opendv_filter_folder
        self.max_clips = args.opendv_max_clips
        self.video_dir = args.opendv_video_dir
        self.return_language = args.opendv_return_language
        self.lang_cache_path = args.opendv_lang_cache_path
        self.return_rgb_path = getattr(args, "return_rgb_path", False)
        self.use_precomputed_feats = getattr(args, "use_precomputed_feats", False)
        self.feat_ext = getattr(args, "opendv_feat_ext", ".dinov2.pt")
        self.use_lang_features = getattr(args, "opendv_use_lang_features", False)
        self.lang_feat_name = getattr(args, "opendv_lang_feat_name", "lang_clip_{start}_{end}.pt")
        self.lang_feat_key = getattr(args, "opendv_lang_feat_key", "text_tokens")
        self.lang_mask_key = getattr(args, "opendv_lang_mask_key", "attention_mask")
        self.clip_max_length = getattr(args, "clip_max_length", None)
        self.dataloader_log_path = getattr(args, "dataloader_log_path", None)
        self.dataloader_log_every = getattr(args, "dataloader_log_every", 0)
        self.return_future_gt = bool(getattr(args, "opendv_return_future_gt", False))
        self.eval_future_steps = max(0, int(getattr(args, "opendv_eval_future_steps", 0) or 0))
        if not self.return_future_gt:
            self.eval_future_steps = 0
        if self.subset == "train" and args.opendv_lang_cache_train:
            self.lang_cache_path = args.opendv_lang_cache_train
        if self.subset == "val" and args.opendv_lang_cache_val:
            self.lang_cache_path = args.opendv_lang_cache_val
        if self.use_lang_features and (not self.lang_use_annos or not self.lang_root):
            raise ValueError("opendv_use_lang_features requires opendv_use_lang_annos and opendv_lang_root.")
        if self.use_precomputed_feats and self.eval_mode:
            raise ValueError("Precomputed features are not supported with eval_mode; disable eval_mode_during_training.")
        self.single_video_frames = None
        self.use_annotations = False
        self.video_dirs = []
        self.clips = []
        self._init_sources()

    def _init_sources(self):
        if self.video_dir:
            if not osp.isdir(self.video_dir):
                raise FileNotFoundError(f"OpenDV-YouTube video dir not found: {self.video_dir}")
            if self.return_language:
                raise ValueError("opendv_return_language requires language annotations.")
            self.video_dirs = [self.video_dir]
            self.single_video_frames = sorted(
                [osp.join(self.video_dir, name) for name in os.listdir(self.video_dir) if _is_image_file(name)]
            )
            return
        if self.lang_root and self.lang_use_annos:
            self.use_annotations = True
            self.clips = self._load_language_clips()
            if not self.clips:
                raise ValueError("No OpenDV-YouTube clips found in language annotations.")
            return
        self.video_dirs = self._collect_videos()

    def _log_line(self, message):
        log_path = self.dataloader_log_path
        if not log_path:
            return
        try:
            log_dir = osp.dirname(log_path)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            worker = data.get_worker_info()
            worker_id = worker.id if worker is not None else "main"
            rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            line = f"{ts} rank={rank} worker={worker_id} pid={os.getpid()} {message}\n"
            with open(log_path, "a") as f:
                f.write(line)
        except Exception:
            pass

    def _min_frames_required(self):
        if self.eval_mode:
            num_frames_skip = 2
            step = num_frames_skip + 1
            return step * (self.sequence_length + self.eval_future_steps)
        if self.augmentations["no_timestep_augm"]:
            max_skip = 2
        elif self.augmentations["timestep_augm"] is not None:
            max_skip = max(1, len(self.augmentations["timestep_augm"]))
        else:
            max_skip = 3
        step = max_skip + 1
        return step * self.sequence_length - max_skip

    def _collect_videos(self):
        root = self.train_root if self.subset == "train" else self.val_root
        if not osp.isdir(root):
            raise FileNotFoundError(f"OpenDV-YouTube root not found: {root}")
        video_dirs = []
        if self.meta_path and osp.isfile(self.meta_path):
            meta_infos = json.load(open(self.meta_path, "r"))
            for video_meta in meta_infos:
                split = str(video_meta.get("split", "")).lower()
                if split and split != self.subset:
                    continue
                youtuber = video_meta.get("youtuber")
                video_id = video_meta.get("videoid") or video_meta.get("video_id") or video_meta.get("id")
                if not youtuber or not video_id:
                    continue
                dir_path = osp.join(root, _youtuber_formatize(youtuber), str(video_id))
                if osp.isdir(dir_path):
                    video_dirs.append(dir_path)
        else:
            for current_root, dirnames, filenames in os.walk(root):
                if any(_is_image_file(f) for f in filenames):
                    video_dirs.append(current_root)
                    dirnames[:] = []
        min_frames = self._min_frames_required()
        valid_dirs = []
        for dir_path in sorted(video_dirs):
            frame_count = sum(1 for name in os.listdir(dir_path) if _is_image_file(name))
            if frame_count >= min_frames:
                valid_dirs.append(dir_path)
        if not valid_dirs:
            raise ValueError("No valid OpenDV-YouTube clips found after filtering for minimum frames.")
        return valid_dirs

    def _entry_to_clip(self, entry, root, base_root, min_frames):
        folder = entry.get("folder")
        first_frame = entry.get("first_frame")
        last_frame = entry.get("last_frame")
        if not folder or not first_frame or not last_frame:
            return None
        if self.lang_filter and self.lang_filter not in folder:
            return None
        folder_norm = folder.replace("\\", "/").lstrip("/")
        if folder_norm.startswith("full_images/") or folder_norm == "full_images":
            rel_folder = folder_norm[len("full_images/"):] if folder_norm != "full_images" else ""
            clip_dir = osp.join(base_root, "full_images", rel_folder)
        elif folder_norm.startswith("val_images/") or folder_norm == "val_images":
            rel_folder = folder_norm[len("val_images/"):] if folder_norm != "val_images" else ""
            clip_dir = osp.join(base_root, "val_images", rel_folder)
        else:
            clip_dir = osp.join(root, folder_norm)
        if not osp.isdir(clip_dir):
            return None
        start_id = int(osp.splitext(first_frame)[0])
        end_id = int(osp.splitext(last_frame)[0])
        if end_id < start_id:
            return None
        frame_count = end_id - start_id + 1
        if frame_count < min_frames:
            return None
        return {
            "dir": clip_dir,
            "start": start_id,
            "end": end_id,
            "pad": len(osp.splitext(first_frame)[0]),
            "ext": osp.splitext(first_frame)[1],
            "cmd": entry.get("cmd"),
            "blip": entry.get("blip"),
        }

    def _load_language_clips(self):
        ann_files = []
        if self.subset == "val":
            ann_files.append(osp.join(self.lang_root, "10hz_YouTube_val.json"))
        else:
            for split_id in range(10):
                ann_files.append(osp.join(self.lang_root, f"10hz_YouTube_train_split{split_id}.json"))
        root = self.train_root if self.subset == "train" else self.val_root
        base_root = osp.dirname(root)
        min_frames = self._min_frames_required()
        clips = []
        if self.lang_cache_path and osp.isfile(self.lang_cache_path):
            cached = json.load(open(self.lang_cache_path, "r"))
            for entry in cached:
                clip = self._entry_to_clip(entry, root, base_root, min_frames)
                if clip is None:
                    continue
                clips.append(clip)
                if self.max_clips and len(clips) >= self.max_clips:
                    return clips
            return clips
        for ann_path in ann_files:
            if not osp.isfile(ann_path):
                continue
            for entry in _iter_json_array(ann_path):
                clip = self._entry_to_clip(entry, root, base_root, min_frames)
                if clip is None:
                    continue
                clips.append(clip)
                if self.max_clips and len(clips) >= self.max_clips:
                    return clips
        return clips

    def _clip_frame_paths(self, clip, min_frames=None):
        frame_items = []
        for name in os.listdir(clip["dir"]):
            if not _is_image_file(name):
                continue
            stem = osp.splitext(name)[0]
            if not stem.isdigit():
                continue
            frame_id = int(stem)
            if frame_id < clip["start"] or frame_id > clip["end"]:
                continue
            frame_items.append((frame_id, name))
        frame_items.sort()
        frames = [osp.join(clip["dir"], name) for _, name in frame_items]
        if min_frames is not None and len(frames) < min_frames:
            return None
        return frames

    def _feature_hw(self):
        patch = 14 if self.feature_extractor in ("dino", "eva2-clip") else 16
        return self.img_size[0] // patch, self.img_size[1] // patch

    def _frame_feature_path(self, frame_path):
        root, _ = osp.splitext(frame_path)
        return root + self.feat_ext

    def _load_feature_sequence(self, sequence_frames_path):
        target_h, target_w = self._feature_hw()
        feats = []
        for frame_path in sequence_frames_path:
            feat_path = self._frame_feature_path(frame_path)
            if not osp.isfile(feat_path):
                raise FileNotFoundError(f"OpenDV feature not found: {feat_path}")
            payload = torch.load(feat_path, map_location="cpu")
            if isinstance(payload, dict):
                feat = payload.get("features")
                if feat is None:
                    feat = payload.get("feat")
            else:
                feat = payload
            if feat is None:
                raise ValueError(f"Invalid feature payload: {feat_path}")
            if feat.dim() == 2:
                src_h = None
                src_w = None
                if isinstance(payload, dict):
                    img_size = payload.get("img_size")
                    patch_size = payload.get("patch_size")
                    if img_size and patch_size:
                        src_h = int(img_size[0]) // int(patch_size)
                        src_w = int(img_size[1]) // int(patch_size)
                if src_h is None or src_w is None:
                    src_h, src_w = target_h, target_w
                if feat.shape[0] != src_h * src_w:
                    raise ValueError(f"Unexpected feature size in {feat_path}")
                feat = feat.view(src_h, src_w, -1)
            feats.append(feat)
        return torch.stack(feats, dim=0)

    def _random_crop_features(self, frames, target_h, target_w):
        src_h = frames.shape[1]
        src_w = frames.shape[2]
        if src_h == 0 or src_w == 0:
            raise ValueError("Invalid feature map size for cropping.")
        s_f = np.random.rand() / 2.0 + 0.5
        crop_h = max(1, int(src_h * s_f))
        crop_w = max(1, int(src_w * s_f))
        crop_h = min(crop_h, src_h)
        crop_w = min(crop_w, src_w)
        top = 0 if src_h == crop_h else np.random.randint(0, src_h - crop_h + 1)
        left = 0 if src_w == crop_w else np.random.randint(0, src_w - crop_w + 1)
        cropped = frames[:, top:top + crop_h, left:left + crop_w, :]
        if crop_h == target_h and crop_w == target_w:
            return cropped
        cropped = cropped.permute(0, 3, 1, 2)
        resized = F.interpolate(cropped, size=(target_h, target_w), mode="bilinear", align_corners=False)
        return resized.permute(0, 2, 3, 1)

    def _clip_lang_feature_path(self, clip):
        name = self.lang_feat_name or "lang_clip_{start}_{end}.pt"
        start = str(clip["start"]).zfill(clip["pad"])
        end = str(clip["end"]).zfill(clip["pad"])
        if "{start}" in name or "{end}" in name:
            name = name.replace("{start}", start).replace("{end}", end)
        if "{start" in name:
            name = name.replace("{start", start)
        if "{end" in name:
            name = name.replace("{end", end)
        return osp.join(clip["dir"], name)

    def _load_lang_features(self, clip):
        feat_path = self._clip_lang_feature_path(clip)
        if not osp.isfile(feat_path):
            raise FileNotFoundError(f"OpenDV language feature not found: {feat_path}")
        payload = torch.load(feat_path, map_location="cpu")
        text_tokens = None
        text_mask = None
        if isinstance(payload, dict):
            text_tokens = payload.get(self.lang_feat_key)
            if text_tokens is None:
                text_tokens = payload.get("text_tokens")
            attention_mask = payload.get(self.lang_mask_key)
            if attention_mask is None:
                attention_mask = payload.get("attention_mask")
            if attention_mask is not None:
                attention_mask = torch.as_tensor(attention_mask)
                if attention_mask.dim() == 1 and self.clip_max_length:
                    seq_len = attention_mask.shape[0]
                    if seq_len < self.clip_max_length:
                        pad = torch.zeros(self.clip_max_length - seq_len, dtype=attention_mask.dtype)
                        attention_mask = torch.cat([attention_mask, pad], dim=0)
                    elif seq_len > self.clip_max_length:
                        attention_mask = attention_mask[:self.clip_max_length]
                text_mask = attention_mask == 0
        else:
            text_tokens = payload
        if text_tokens is not None:
            text_tokens = torch.as_tensor(text_tokens)
            if text_tokens.dim() == 2 and self.clip_max_length:
                seq_len = text_tokens.shape[0]
                if seq_len < self.clip_max_length:
                    pad = torch.zeros(
                        (self.clip_max_length - seq_len, text_tokens.shape[1]),
                        dtype=text_tokens.dtype,
                    )
                    text_tokens = torch.cat([text_tokens, pad], dim=0)
                elif seq_len > self.clip_max_length:
                    text_tokens = text_tokens[:self.clip_max_length]
            elif text_tokens.dim() != 2:
                raise ValueError(f"Invalid text token shape in {feat_path}: {tuple(text_tokens.shape)}")
        if text_tokens is None:
            raise ValueError(f"Missing text tokens in {feat_path}")
        if text_tokens is not None:
            text_tokens = text_tokens.clone()
        if text_mask is not None:
            text_mask = text_mask.clone()
        return text_tokens, text_mask

    def __len__(self):
        if self.use_annotations:
            return len(self.clips)
        return len(self.video_dirs)

    def __getitem__(self, idx):
        if self.eval_mode and self.eval_modality is not None:
            raise ValueError("OpenDV-YouTube loader does not provide labels for eval modalities.")
        orig_idx = idx
        log_every = int(self.dataloader_log_every) if self.dataloader_log_every else 0
        log_this = log_every > 0 and (orig_idx % log_every == 0)
        start_time = time.time() if log_this else None
        if log_this:
            self._log_line(
                "fetch_start idx=%s subset=%s eval=%s annotations=%s"
                % (orig_idx, self.subset, self.eval_mode, self.use_annotations)
            )
        min_frames = self._min_frames_required()
        attempts = 0
        last_exc = None
        last_lang_exc = None
        last_info = None
        rgb_path = None

        def _pack(*items):
            if self.return_rgb_path:
                return (*items, rgb_path)
            return items
        while attempts < 10:
            clip = None
            frames_filepaths = None
            text_tokens = None
            text_mask = None
            rgb_path = None
            if self.use_annotations:
                clip = self.clips[idx]
                frames_filepaths = self._clip_frame_paths(clip, min_frames=min_frames)
            elif self.single_video_frames is not None:
                frames_filepaths = self.single_video_frames
            else:
                video_dir = self.video_dirs[idx]
                frames_filepaths = sorted(
                    [osp.join(video_dir, name) for name in os.listdir(video_dir) if _is_image_file(name)]
                )
            if self.use_annotations and clip is not None:
                last_info = {
                    "orig_idx": orig_idx,
                    "idx": idx,
                    "dir": clip.get("dir"),
                    "start": clip.get("start"),
                    "end": clip.get("end"),
                    "pad": clip.get("pad"),
                    "ext": clip.get("ext"),
                }
            elif self.single_video_frames is not None:
                last_info = {
                    "orig_idx": orig_idx,
                    "idx": idx,
                    "dir": self.video_dir,
                    "frames": len(frames_filepaths or []),
                }
            else:
                last_info = {
                    "orig_idx": orig_idx,
                    "idx": idx,
                    "dir": video_dir,
                    "frames": len(frames_filepaths or []),
                }
            if frames_filepaths is None or len(frames_filepaths) < min_frames:
                if log_this:
                    self._log_line(
                        "insufficient_frames idx=%s frames=%s min=%s info=%s"
                        % (idx, len(frames_filepaths or []), min_frames, last_info)
                    )
                attempts += 1
                if self.use_annotations:
                    idx = np.random.randint(0, len(self.clips))
                elif self.single_video_frames is not None:
                    break
                else:
                    idx = np.random.randint(0, len(self.video_dirs))
                continue
            if self.use_lang_features and self.use_annotations:
                try:
                    text_tokens, text_mask = self._load_lang_features(clip)
                except (FileNotFoundError, ValueError) as exc:
                    last_lang_exc = exc
                    if last_info is not None:
                        last_info["lang_feat"] = self._clip_lang_feature_path(clip)
                    self._log_line(
                        "lang_feat_error idx=%s error=%s info=%s"
                        % (idx, repr(exc), last_info)
                    )
                    attempts += 1
                    idx = np.random.randint(0, len(self.clips))
                    continue
            try:
                if self.eval_mode:
                    future_gt = None
                    if self.return_rgb_path:
                        if self.return_future_gt:
                            frames, gt, future_gt, rgb_path = process_opendv_evalmode(
                                frames_filepaths,
                                self.img_size,
                                sequence_length=self.sequence_length,
                                feature_extractor=self.feature_extractor,
                                return_paths=True,
                                return_future_gt=True,
                                future_steps=self.eval_future_steps,
                            )
                        else:
                            frames, gt, rgb_path = process_opendv_evalmode(
                                frames_filepaths,
                                self.img_size,
                                sequence_length=self.sequence_length,
                                feature_extractor=self.feature_extractor,
                                return_paths=True,
                            )
                    else:
                        if self.return_future_gt:
                            frames, gt, future_gt = process_opendv_evalmode(
                                frames_filepaths,
                                self.img_size,
                                sequence_length=self.sequence_length,
                                feature_extractor=self.feature_extractor,
                                return_future_gt=True,
                                future_steps=self.eval_future_steps,
                            )
                        else:
                            frames, gt = process_opendv_evalmode(
                                frames_filepaths,
                                self.img_size,
                                sequence_length=self.sequence_length,
                                feature_extractor=self.feature_extractor,
                            )
                    frames = frames.clone()
                    gt = gt.clone()
                    payload = [frames, gt]
                    if future_gt is not None:
                        payload.append(future_gt.clone())
                    if self.use_lang_features and self.use_annotations:
                        if self.return_language:
                            if log_this:
                                elapsed = time.time() - start_time if start_time is not None else 0.0
                                self._log_line(
                                    "fetch_ok idx=%s attempts=%s elapsed=%.3f frames=%s"
                                    % (idx, attempts, elapsed, len(frames_filepaths or []))
                                )
                            return _pack(*payload, clip.get("cmd"), clip.get("blip"), text_tokens, text_mask)
                        if log_this:
                            elapsed = time.time() - start_time if start_time is not None else 0.0
                            self._log_line(
                                "fetch_ok idx=%s attempts=%s elapsed=%.3f frames=%s"
                                % (idx, attempts, elapsed, len(frames_filepaths or []))
                            )
                        return _pack(*payload, text_tokens, text_mask)
                    if self.return_language and self.use_annotations:
                        if log_this:
                            elapsed = time.time() - start_time if start_time is not None else 0.0
                            self._log_line(
                                "fetch_ok idx=%s attempts=%s elapsed=%.3f frames=%s"
                                % (idx, attempts, elapsed, len(frames_filepaths or []))
                            )
                        return _pack(*payload, clip.get("cmd"), clip.get("blip"))
                    if log_this:
                        elapsed = time.time() - start_time if start_time is not None else 0.0
                        self._log_line(
                            "fetch_ok idx=%s attempts=%s elapsed=%.3f frames=%s"
                            % (idx, attempts, elapsed, len(frames_filepaths or []))
                        )
                    return _pack(*payload)
                if self.use_precomputed_feats:
                    sequence_frames_path = _select_train_frame_paths(
                        frames_filepaths,
                        self.subset,
                        self.augmentations,
                        sequence_length=self.sequence_length,
                    )
                    if self.return_rgb_path:
                        rgb_path = sequence_frames_path[-1]
                    frames = self._load_feature_sequence(sequence_frames_path)
                    if self.augmentations["random_crop"] and self.subset == "train":
                        target_h, target_w = self._feature_hw()
                        frames = self._random_crop_features(frames, target_h, target_w)
                    if self.augmentations["random_horizontal_flip"] and self.subset == "train":
                        if np.random.rand() > 0.5:
                            frames = torch.flip(frames, dims=[2])
                else:
                    if self.return_rgb_path:
                        frames, frame_paths = process_trainmode(
                            frames_filepaths,
                            self.img_size,
                            self.subset,
                            self.augmentations,
                            self.sequence_length,
                            feature_extractor=self.feature_extractor,
                            return_paths=True,
                        )
                        rgb_path = frame_paths[-1]
                    else:
                        frames = process_trainmode(
                            frames_filepaths,
                            self.img_size,
                            self.subset,
                            self.augmentations,
                            self.sequence_length,
                            feature_extractor=self.feature_extractor,
                        )
                frames = frames.clone()
                if self.use_lang_features and self.use_annotations:
                    if self.return_language:
                        if log_this:
                            elapsed = time.time() - start_time if start_time is not None else 0.0
                            self._log_line(
                                "fetch_ok idx=%s attempts=%s elapsed=%.3f frames=%s"
                                % (idx, attempts, elapsed, len(frames_filepaths or []))
                            )
                        return _pack(frames, clip.get("cmd"), clip.get("blip"), text_tokens, text_mask)
                    if log_this:
                        elapsed = time.time() - start_time if start_time is not None else 0.0
                        self._log_line(
                            "fetch_ok idx=%s attempts=%s elapsed=%.3f frames=%s"
                            % (idx, attempts, elapsed, len(frames_filepaths or []))
                        )
                    return _pack(frames, text_tokens, text_mask)
                if self.return_language and self.use_annotations:
                    if log_this:
                        elapsed = time.time() - start_time if start_time is not None else 0.0
                        self._log_line(
                            "fetch_ok idx=%s attempts=%s elapsed=%.3f frames=%s"
                            % (idx, attempts, elapsed, len(frames_filepaths or []))
                        )
                    return _pack(frames, clip.get("cmd"), clip.get("blip"))
                if log_this:
                    elapsed = time.time() - start_time if start_time is not None else 0.0
                    self._log_line(
                        "fetch_ok idx=%s attempts=%s elapsed=%.3f frames=%s"
                        % (idx, attempts, elapsed, len(frames_filepaths or []))
                    )
                return _pack(frames)
            except (FileNotFoundError, OSError, ValueError, RuntimeError) as exc:
                last_exc = exc
                self._log_line(
                    "fetch_error idx=%s error=%s info=%s"
                    % (idx, repr(exc), last_info)
                )
                attempts += 1
                if self.use_annotations:
                    idx = np.random.randint(0, len(self.clips))
                elif self.single_video_frames is not None:
                    break
                else:
                    idx = np.random.randint(0, len(self.video_dirs))
                continue
        if last_exc is not None:
            msg = "OpenDV-YouTube clip failed to load after retries."
            if last_info is not None:
                msg = f"{msg} Last clip info: {last_info}"
            self._log_line("fetch_failed error=%s" % msg)
            raise FileNotFoundError(msg) from last_exc
        if last_lang_exc is not None:
            msg = "OpenDV-YouTube clip has missing language features after retries."
            if last_info is not None:
                msg = f"{msg} Last clip info: {last_info}"
            self._log_line("fetch_failed error=%s" % msg)
            raise FileNotFoundError(msg) from last_lang_exc
        msg = "OpenDV-YouTube clip has insufficient frames after filtering."
        if last_info is not None:
            msg = f"{msg} Last clip info: {last_info}"
        self._log_line("fetch_failed error=%s" % msg)
        raise FileNotFoundError(msg)


class OpenDV_VideoData(pl.LightningDataModule):
    def __init__(self, arguments, subset="train", batch_size=8):
        super().__init__()
        self.data_path = arguments.data_path
        self.subset = subset
        self.sequence_length = arguments.sequence_length
        self.batch_size = batch_size
        self.img_size = arguments.img_size
        self.arguments = arguments
        self.eval_mode = arguments.eval_mode
        self.num_workers = arguments.num_workers
        self.num_workers_val = arguments.num_workers if arguments.num_workers_val is None else arguments.num_workers_val
        self.dataloader_timeout = getattr(arguments, "dataloader_timeout", 0)
        self.feature_extractor = arguments.feature_extractor
        self.use_val_to_train = arguments.use_val_to_train
        self.use_train_to_val = arguments.use_train_to_val
        self.meta_path = arguments.opendv_meta_path
        self.train_root = arguments.opendv_train_root
        self.val_root = arguments.opendv_val_root
        if self.train_root is None or self.val_root is None:
            if arguments.opendv_root is not None:
                base_root = arguments.opendv_root
            else:
                base_root = self.data_path
                if osp.basename(base_root) in ("full_images", "val_images"):
                    base_root = osp.dirname(base_root)
            if self.train_root is None:
                self.train_root = osp.join(base_root, "full_images")
            if self.val_root is None:
                self.val_root = osp.join(base_root, "val_images")

    def _dataset(self, subset, eval_mode):
        return OpenDVYouTubeDataset(
            self.train_root,
            self.val_root,
            self.arguments,
            self.sequence_length,
            self.img_size,
            subset=subset,
            eval_mode=eval_mode,
            feature_extractor=self.feature_extractor,
            meta_path=self.meta_path,
        )

    def _log_stage(self, message):
        log_path = getattr(self.arguments, "ddp_stage_log_path", None)
        if not log_path:
            return
        try:
            log_dir = osp.dirname(log_path)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))
            pid = os.getpid()
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            with open(log_path, "a") as f:
                f.write(f"{ts} rank={rank} pid={pid} {message}\n")
        except Exception:
            pass

    def _dataloader(self, subset, shuffle=True, drop_last=False, eval_mode=False):
        dataset = self._dataset(subset, eval_mode)
        dataloader = data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=opendv_collate,
            timeout=self.dataloader_timeout,
            shuffle=shuffle,
            drop_last=drop_last,
        )
        return dataloader

    def train_dataloader(self):
        self._log_stage("train_dataloader_start")
        train_subset = "val" if self.use_val_to_train else "train"
        dataset = self._dataset(train_subset, eval_mode=False)
        world_size = int(getattr(self.arguments, "num_gpus", 1) or 1)
        drop_last = len(dataset) >= self.batch_size * world_size
        dataloader = data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=opendv_collate,
            timeout=self.dataloader_timeout,
            shuffle=True,
            drop_last=drop_last,
        )
        self._log_stage("train_dataloader_ready")
        return dataloader

    def val_dataloader(self):
        self._log_stage("val_dataloader_start")
        val_subset = "train" if self.use_train_to_val else "val"
        dataset = self._dataset(val_subset, self.eval_mode)
        dataloader = data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers_val,
            pin_memory=True,
            collate_fn=opendv_collate,
            timeout=self.dataloader_timeout,
            shuffle=False,
            drop_last=False,
        )
        self._log_stage("val_dataloader_ready")
        return dataloader
