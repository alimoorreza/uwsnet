import _init_paths
import argparse
import os
import pprint

import logging
import timeit

import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
from tensorboardX import SummaryWriter

from networks import hrnet_v2 as models

from config import config_hrnet_v2 as config
from config import update_config_hrnet_v2 as update_config
from core.criterion import CrossEntropy, OhemCrossEntropy
from core.function import validate
from utils.hrnet_v2_utils.utils import create_logger, FullModel
from utils.hrnet_utils.normalization_utils import get_imagenet_mean_std
from semantic_dataloader import UWFSDataLoader
from semantic_dataloader_final import UWFSDataLoaderVal as UWFSDataLoader2
from utils.hrnet_utils import transform
from tqdm import tqdm
from scipy.io import loadmat
from sklearn.model_selection import StratifiedShuffleSplit

import glob
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description='Test segmentation network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('--seed', type=int, default=304)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args


def main():
    args = parse_args()

    if args.seed > 0:
        import random
        print('Seeding with', args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(config)

    writer_dict = {
        'writer': SummaryWriter(tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED
    gpus = list(config.GPUS)

    batch_size = config.TRAIN.BATCH_SIZE_PER_GPU

    # prepare data
    mean, std = get_imagenet_mean_std()

    if config.DATASET.DATASET == 'UWS3':
        train_transform_list = [
            transform.ResizeShort(config.TRAIN.IMAGE_SIZE[0]),
            transform.Crop(
                [config.TRAIN.IMAGE_SIZE[0], config.TRAIN.IMAGE_SIZE[1]],
                crop_type="rand",
                padding=mean,
                ignore_label=config.TRAIN.IGNORE_LABEL,
            ),
            transform.ToTensor(),
            transform.Normalize(mean=mean, std=std),
        ]
        # transform.ResizeTest((config.TRAIN.TRAIN_H, config.TRAIN.TRAIN_W)),
        # transform.ResizeShort(config.TRAIN.SHORT_SIZE),
        val_transform_list = [
            transform.ResizeShort(config.TRAIN.IMAGE_SIZE[0]),
            transform.Crop(
                [config.TRAIN.IMAGE_SIZE[0], config.TRAIN.IMAGE_SIZE[1]],
                crop_type="center",
                padding=mean,
                ignore_label=config.TRAIN.IGNORE_LABEL,
            ),
            transform.ToTensor(),
            transform.Normalize(mean=mean, std=std),
        ]

        train_dir = config.DATASET.TRAIN_SET
        val_dir = config.DATASET.TEST_SET

        images_files = glob.glob(
            os.path.join(
                config.DATASET.ROOT,
                train_dir,
                'images',
                '*.png'
            )
        )
        masks_files = \
            [os.path.join(config.DATASET.ROOT, train_dir, 'labels', os.path.basename(m_i)) for m_i in images_files]

        images = []
        masks = []

        for i_i_fl, img_fl in enumerate(tqdm(images_files)):
            images.append(np.array(
                Image.open(img_fl)
            ))
            masks.append(np.array(
                Image.open(masks_files[i_i_fl])
            ))

        dataset_len = len(images)
        logger.info(f'Total train image files: {dataset_len}')
        np.random.seed(0)
        rand_n = list(np.random.randint(low=0, high=dataset_len, size=dataset_len // 2))
        for i in rand_n:
            im = images[i]
            target = masks[i]
            # perform horizontal flip
            images.append(np.fliplr(im))
            masks.append(np.fliplr(target))

        # shift right
        rand_n = list(np.random.randint(low=0, high=dataset_len, size=dataset_len // 4))
        for i in rand_n:
            shift = 20
            im = images[i]
            target = masks[i]

            im[:, shift:] = im[:, :-shift]
            target[:, shift:] = target[:, :-shift]
            images.append(im)
            masks.append(target)

        # shift left
        rand_n = list(np.random.randint(low=0, high=dataset_len, size=dataset_len // 4))
        for i in rand_n:
            shift = 20
            im = images[i]
            target = masks[i]

            im[:, :-shift] = im[:, shift:]
            target[:, :-shift] = target[:, shift:]
            images.append(im)
            masks.append(target)

        # shift up
        rand_n = list(np.random.randint(low=0, high=dataset_len, size=dataset_len // 4))
        for i in rand_n:
            shift = 20
            im = images[i]
            target = masks[i]

            im[:-shift, :] = im[shift:, :]
            target[:-shift, :] = target[shift:, :]
            images.append(im)
            masks.append(target)

        # shift down
        rand_n = list(np.random.randint(low=0, high=dataset_len, size=dataset_len // 4))
        for i in rand_n:
            shift = 20
            im = images[i]
            target = masks[i]

            im[shift:, :] = im[:-shift, :]
            target[shift:, :] = target[:-shift, :]
            images.append(im)
            masks.append(target)

        images_train = images
        masks_train = masks

        images_files = glob.glob(
            os.path.join(
                config.DATASET.ROOT,
                val_dir,
                'images',
                '*.png'
            )
        )
        masks_files = \
            [os.path.join(config.DATASET.ROOT, val_dir, 'labels', os.path.basename(m_i)) for m_i in images_files]

        images_test = []
        masks_test = []

        for i_i_fl, img_fl in enumerate(tqdm(images_files)):
            images_test.append(np.array(
                Image.open(img_fl)
            ))
            masks_test.append(np.array(
                Image.open(masks_files[i_i_fl])
            ))

        dataset_len = len(images_test)
        logger.info(f'Total val files: {dataset_len}')

        train_dataset = UWFSDataLoader2(
            output_image_height=config.TRAIN.IMAGE_SIZE[0],
            images=images_train,
            masks=masks_train,
            normalizer=transform.Compose(train_transform_list),
            channel_values=None
        )
        val_dataset = UWFSDataLoader2(
            output_image_height=config.TRAIN.IMAGE_SIZE[0],
            images=images_test,
            masks=masks_test,
            normalizer=transform.Compose(val_transform_list),
            channel_values=None
        )
    else:
        train_dataset = None
        val_dataset = None
        logger.info("=> no dataset found. " 'Exiting...')
        exit()

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        num_workers=config.WORKERS,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True
    )
    logger.info(f'Train loader has len: {len(train_loader)}')

    val_sampler = None
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True,
        sampler=val_sampler,
        drop_last=True
    )
    logger.info(f'Validation loader has len: {len(val_loader)}')

    if torch.__version__.startswith('1'):
        module = eval('models.' + config.MODEL.NAME)
        module.BatchNorm2d_class = module.BatchNorm2d = torch.nn.BatchNorm2d
    model = eval('models.' + config.MODEL.NAME +
                 '.get_seg_model')(config)

    if config.TEST.MODEL_FILE:
        model_state_file = config.TEST.MODEL_FILE
    else:
        model_state_file = os.path.join(final_output_dir, 'final_state.pth')
    logger.info('=> loading model from {}'.format(model_state_file))

    pretrained_dict = torch.load(model_state_file)
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items()
                       if k[6:] in model_dict.keys()}
    for k, _ in pretrained_dict.items():
        logger.info(
            '=> loading {} from pretrained model'.format(k))
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    gpus = list(config.GPUS)
    model = nn.DataParallel(model, device_ids=gpus).cuda()

    extra_epoch_iters = 0
    start = timeit.default_timer()
    # end_epoch = config.TRAIN.END_EPOCH + config.TRAIN.EXTRA_EPOCH
    # num_iters = config.TRAIN.END_EPOCH * epoch_iters
    # extra_iters = config.TRAIN.EXTRA_EPOCH * extra_epoch_iters

    valid_loss, mean_IoU, IoU_array = validate(config,
                                               val_loader, model, writer_dict)

    msg = 'Loss: {:.3f}, MeanIU: {: 4.4f}'.format(
        valid_loss, mean_IoU)
    logging.info(msg)
    logging.info(IoU_array)

    writer_dict['writer'].close()
    end = timeit.default_timer()
    logger.info('Hours: %d' % np.int((end - start) / 3600))
    logger.info('Done')


if __name__ == '__main__':
    main()
