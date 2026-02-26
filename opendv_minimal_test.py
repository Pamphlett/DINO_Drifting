import argparse

from src.data import OpenDV_VideoData


def parse_tuple(x):
    return tuple(map(int, x.split(',')))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='')
    parser.add_argument('--sequence_length', type=int, default=5)
    parser.add_argument('--img_size', type=parse_tuple, default=(224, 448))
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--num_workers_val', type=int, default=None)
    parser.add_argument('--feature_extractor', type=str, default='dino')
    parser.add_argument('--eval_mode', action='store_true', default=False)
    parser.add_argument('--eval_modality', type=str, default=None)
    parser.add_argument('--eval_midterm', action='store_true', default=False)
    parser.add_argument('--use_val_to_train', action='store_true', default=False)
    parser.add_argument('--use_train_to_val', action='store_true', default=False)
    parser.add_argument('--random_crop', action='store_true', default=False)
    parser.add_argument('--random_horizontal_flip', action='store_true', default=False)
    parser.add_argument('--random_time_flip', action='store_true', default=False)
    parser.add_argument('--timestep_augm', type=list, default=None)
    parser.add_argument('--no_timestep_augm', action='store_true', default=False)
    parser.add_argument('--opendv_root', type=str, default=None)
    parser.add_argument('--opendv_train_root', type=str, default=None)
    parser.add_argument('--opendv_val_root', type=str, default=None)
    parser.add_argument('--opendv_meta_path', type=str, default=None)
    parser.add_argument('--opendv_lang_root', type=str, default=None)
    parser.add_argument('--opendv_use_lang_annos', action='store_true', default=False)
    parser.add_argument('--opendv_filter_folder', type=str, default=None)
    parser.add_argument('--opendv_max_clips', type=int, default=1)
    parser.add_argument('--opendv_video_dir', type=str, default=None)
    parser.add_argument('--opendv_return_language', action='store_true', default=False)
    parser.add_argument('--batch_size', type=int, default=1)

    args = parser.parse_args()
    data = OpenDV_VideoData(arguments=args, subset='train', batch_size=args.batch_size)
    loader = data.train_dataloader()
    batch = next(iter(loader))
    if isinstance(batch, (list, tuple)) and len(batch) == 3:
        frames, cmd, blip = batch
        print('frames:', tuple(frames.shape))
        print('cmd:', cmd)
        print('blip:', blip)
    elif isinstance(batch, (list, tuple)) and len(batch) == 2:
        frames, gt = batch
        print('frames:', tuple(frames.shape))
        print('gt:', tuple(gt.shape))
    else:
        print('batch type:', type(batch))
        try:
            print('frames:', tuple(batch.shape))
        except AttributeError:
            print('batch:', batch)


if __name__ == '__main__':
    main()
