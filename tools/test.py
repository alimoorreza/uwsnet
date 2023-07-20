import _init_paths
from torch.utils.tensorboard import SummaryWriter
# default `log_dir` is "runs" - we'll be more specific here
from dataloader import *
from tqdm import tqdm
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import torch
from torch.optim.lr_scheduler import MultiStepLR
from config import config, update_config
import argparse
from utils.utils import create_logger, save_checkpoint
from utils.transform import *
from utils.metric import Metric
from utils.normalization_utils import get_imagenet_mean_std
from networks import FRPMMs, FPMMs, FPMMs_vgg
from utils import my_optim
import random


def get_writer():
    writer = SummaryWriter(
        f'{config.LOG_DIR}/v2-{config.TRAIN.ARCH}-5shots-split{config.TRAIN.TEST_LABEL_SPLIT_VALUE}'
    )
    return writer


def get_model():
    if config.TRAIN.ARCH == "FPMMs" or config.TRAIN.ARCH == "FPMMs_vgg" or config.TRAIN.ARCH == "FRPMMs":
        model = eval(config.TRAIN.ARCH).OneModel(config)
        optimizer = my_optim.get_finetune_optimizer(config, model)
        criterion = None
        scheduler = None
        print("FRMMs model will be used ......")
        return model, optimizer, criterion, scheduler
    elif config.TRAIN.ARCH == 'PAnet' or config.TRAIN.ARCH == 'PAnet_new':
        """
            UWSNet v1 -> eca_net_sup_que
            UWSNet v2 -> eca_net_sup_que_vgg16
            UWSNet v3 -> triplet_sup_que
            UWSNet v4 -> triplet_sup_que_dice
            UWSNet v5 -> triplet_sup_que_vgg16
            UWSNet v6 -> triplet_sup_que_vgg16_dice
        """
        if os.path.exists(config.TRAIN.VGG_MODEL_PATH):
            vgg16_model_weight = config.TRAIN.VGG_MODEL_PATH
        else:
            vgg16_model_weight = None
        vgg16_model_weight = config.TRAIN.VGG_MODEL_PATH
        if config.TRAIN.PA_NET_TYPE == 'basic':
            from networks.few_shot import FewShotSeg
            model = FewShotSeg(pretrained_path=vgg16_model_weight, cfg={'align': True})
        elif config.TRAIN.PA_NET_TYPE == 'eca_net_sup_que':
            from networks.few_shot_with_ecanet import FewShotSeg
            model = FewShotSeg(pretrained_path=vgg16_model_weight, cfg={'align': True}, vgg_type='vgg')
        elif config.TRAIN.PA_NET_TYPE == 'eca_net_sup_que_vgg16':
            from networks.few_shot_with_ecanet import FewShotSeg
            model = FewShotSeg(pretrained_path=vgg16_model_weight, cfg={'align': True}, vgg_type='ecanet')
        elif config.TRAIN.PA_NET_TYPE == 'triplet_sup_que_vgg16':
            from networks.few_shot_with_triplet import FewShotSeg
            model = FewShotSeg(pretrained_path=vgg16_model_weight, cfg={'align': True}, vgg_type='triplet')
        elif config.TRAIN.PA_NET_TYPE == 'triplet_sup_que':
            from networks.few_shot_with_triplet import FewShotSeg
            model = FewShotSeg(pretrained_path=vgg16_model_weight, cfg={'align': True}, vgg_type='vgg')
        elif config.TRAIN.PA_NET_TYPE == 'triplet_sup_que_vgg16_dice':
            from networks.few_shot_with_triplet_diceloss import FewShotSeg
            model = FewShotSeg(pretrained_path=vgg16_model_weight, cfg={'align': True}, vgg_type='triplet',
                               gpu=config.GPU)
        elif config.TRAIN.PA_NET_TYPE == 'triplet_sup_que_dice':
            from networks.few_shot_with_triplet_diceloss import FewShotSeg
            model = FewShotSeg(pretrained_path=vgg16_model_weight, cfg={'align': True}, vgg_type='vgg',
                               gpu=config.GPU)
        else:
            raise Exception("Invalid Network Type...")
            # from networks.few_shot import FewShotSeg
            # model = FewShotSeg(pretrained_path=vgg16_model_weight, cfg={'align': True})

        optimizer_val = {
            'lr': config.TRAIN.BASE_LR,
            'momentum': config.TRAIN.MOMENTUM,
            'weight_decay': config.TRAIN.WEIGHT_DECAY
        }

        # for param in model.encoder.backbone.conv1.parameters():
        #     param.requires_grad = False
        # for param in model.encoder.backbone.conv2.parameters():
        #     param.requires_grad = False
        # for param in model.encoder.backbone.conv3.parameters():
        #     param.requires_grad = False
        # for param in model.encoder.backbone.conv4.parameters():
        #     param.requires_grad = False
        # for param in model.encoder.backbone.conv5.parameters():
        #     param.requires_grad = False

        # print(model.keys())
        optimizer = optim.SGD(model.parameters(), **optimizer_val)
        # optimizer = optim.SGD(
        #     [
        #         {'params': model.encoder.backbone.triplet_attention1.parameters()},
        #         {'params': model.encoder.backbone.triplet_attention2.parameters()},
        #         {'params': model.encoder.backbone.triplet_attention3.parameters()},
        #         {'params': model.triplet_attention.parameters()},
        #         {'params': model.d3_conv.parameters()},
        #         {'params': model.d3_conv_2.parameters()},
        #         {'params': model.d2_conv.parameters()},
        #     ],  **optimizer_val
        # )
        # lr_milestones = config.TRAIN.LR_MILESTONE
        # scheduler = MultiStepLR(optimizer, milestones=lr_milestones, gamma=config.TRAIN.GAMMA)
        scheduler = None
        criterion = nn.CrossEntropyLoss(ignore_index=config.TRAIN.IGNORE_LABEL)
        print("PAnet model will be used ......")
        return model, optimizer, criterion, scheduler
    elif config.TRAIN.ARCH == 'hsnet':
        from networks.hsnet import HypercorrSqueezeNetwork
        model = HypercorrSqueezeNetwork(config.TRAIN.HSNET_BB, False)
        optimizer = optim.Adam([{"params": model.parameters(), "lr": config.TRAIN.BASE_LR}])
        scheduler = None
        criterion = None
        print("HSnet model will be used ......")
        return model, optimizer, criterion, scheduler
    elif config.TRAIN.ARCH == 'asnet':
        from networks.asnet import AttentiveSqueezeNetwork
        model = AttentiveSqueezeNetwork(config.TRAIN.ASNET_BB, False)
        optimizer = optim.Adam([{"params": model.parameters(), "lr": config.TRAIN.BASE_LR}])
        scheduler = None
        criterion = None
        print("HSnet model will be used ......")
        return model, optimizer, criterion, scheduler
    else:
        print("No {a} Model Found".format(a=config.TRAIN.ARCH))
        exit()


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Semantic Segmentation')
    parser.add_argument('--config', type=str,
                        default='../config/experiments/uw_few_shot_pmms_training_config_v1.yaml',
                        help='config file')

    args = parser.parse_args()
    assert args.config is not None
    update_config(config, args)
    return args


def main():
    print('Using PyTorch version: ', torch.__version__)
    args = get_parser()
    main_worker(args)


def main_worker(args):
    model, optimizer, criterion, scheduler = get_model()

    best_iou = 0
    best_model = False
    best_loss = float('inf')
    last_epoch = config.TRAIN.BEGIN_EPOCH

    logger, final_output_dir, tb_log_dir = create_logger(config, args.config, 'train')
    writer = get_writer()

    logger.info(config)
    logger.info("=> creating {} model ...".format(config.TRAIN.ARCH))
    # logger.info(model)

    print(config)
    print("=> creating {} model ...".format(config.TRAIN.ARCH))
    # print(model)

    num_of_parameters = sum(map(torch.numel, model.parameters()))
    logger.info("=> model parameters: {}".format(num_of_parameters))

    print("=> model parameters: {}".format(num_of_parameters))

    # model = torch.nn.DataParallel(model.cuda(), device_ids=config.GPUS)
    model = model.cuda(config.GPU)

    if config.TRAIN.RESUME:
        model_state_file = os.path.join(final_output_dir,
                                        'checkpoint.pth.tar')
        if os.path.isfile(model_state_file):
            checkpoint = torch.load(model_state_file)
            last_epoch = checkpoint['epoch']
            best_iou = checkpoint['perf']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint (epoch {})"
                        .format(checkpoint['epoch']))
            print("=> loaded checkpoint (epoch {})"
                  .format(checkpoint['epoch']))
            best_model = True
        else:
            logger.info("=> no checkpoint found, Running from Scratch")
            print("=> no checkpoint found, Running from Scratch")

    if config.TRAIN.PRETRAINED_MODEL:
        model_state_file = config.TRAIN.PRETRAINED_MODEL
        if os.path.isfile(model_state_file):
            checkpoint = torch.load(model_state_file, map_location=torch.device('cuda:0'))
            # print(checkpoint.module.keys())
            # last_epoch = checkpoint['epoch']
            # best_perf = checkpoint['perf']
            # model.module.load_state_dict(checkpoint['state_dict'])
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            # logger.info("=> loaded checkpoint (epoch {})"
            #             .format(checkpoint['epoch']))
            # best_model = True
            logger.info(f"=> loaded pretrained model {model_state_file}")
            print(f"=> loaded pretrained model {model_state_file}")
        else:
            logger.info("=> no pretrained model found")
            print("=> no pretrained model found")

    BASE_DIR = config.BASE_DIR
    labels_split = joblib.load(BASE_DIR + "dataset_split/labels_split.joblib")
    class2labels = joblib.load(BASE_DIR + "dataset_split/classes2labels.joblib")

    mean, std = get_imagenet_mean_std(scale_val=config.TRAIN.SCALE_VAL)

    train_transform = Compose(
        [
            Resize(size=(config.TRAIN.TRAIN_H, config.TRAIN.TRAIN_W)),
            RandomMirror(),
            ToTensorNormalize(mean=mean, std=std)
        ]
    )

    val_transform = Compose(
        [
            Resize(size=(config.TRAIN.TRAIN_H, config.TRAIN.TRAIN_W)),
            ToTensorNormalize(mean=mean, std=std)
        ]
    )

    directory = config.DATASET.ROOT
    labels_split = labels_split
    test_label_split_value = config.TRAIN.TEST_LABEL_SPLIT_VALUE
    episode_train = config.TRAIN.EPISODE_TRAIN
    episode_eval = config.TRAIN.EPISODE_EVAL
    n_ways = config.TRAIN.N_WAYS
    n_shots = config.TRAIN.N_SHOTS
    random_split_train = config.TRAIN.RANDOM_SPLIT_TRAIN
    random_split_eval = config.TRAIN.RANDOM_SPLIT_EVAL

    if config.TRAIN.ARCH == "FPMMs" or \
            config.TRAIN.ARCH == "FPMMs_vgg" or \
            config.TRAIN.ARCH == "FRPMMs" or \
            config.TRAIN.ARCH == "PAnet" or \
            config.TRAIN.ARCH == "PAnet_new":
        train_dataset = IUDataset(
            directory,
            class2labels,
            labels_split,
            test_label_split_value,
            episode_train,
            n_ways,
            n_shots,
            validation=False,
            transform=train_transform,
            random_split=random_split_train
        )
        validation_dataset = IUDataset(
            directory,
            class2labels,
            labels_split,
            test_label_split_value,
            episode_eval,
            n_ways,
            n_shots,
            validation=True,
            transform=val_transform,
            random_split=random_split_eval
        )
    elif config.TRAIN.ARCH == 'hsnet' or config.TRAIN.ARCH == 'asnet':
        train_dataset = IUDataset(
            directory,
            class2labels,
            labels_split,
            test_label_split_value,
            episode_train,
            n_ways,
            n_shots,
            validation=False,
            transform=train_transform,
            random_split=random_split_train
        )
        validation_dataset = IUDataset(
            directory,
            class2labels,
            labels_split,
            test_label_split_value,
            episode_eval,
            n_ways,
            n_shots,
            validation=True,
            transform=val_transform,
            random_split=random_split_eval
        )
    else:
        train_dataset = None
        validation_dataset = None
        logger.info(f"no {config.TRAIN.ARCH} model found")
        print(f"no {config.TRAIN.ARCH} model found")
        exit()

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config.TRAIN.BATCH_SIZE,
                                  num_workers=config.WORKERS,
                                  pin_memory=True,
                                  shuffle=True)
    validation_dataloader = DataLoader(validation_dataset,
                                       batch_size=config.TRAIN.BATCH_SIZE_VAL,
                                       num_workers=config.WORKERS,
                                       pin_memory=True,
                                       shuffle=True)

    print("TEST_LABEL_SPLIT_VALUE : ", config.TRAIN.TEST_LABEL_SPLIT_VALUE)
    # print("TRAINING START.....")
    print("N SHOTS", config.TRAIN.N_SHOTS)

    # logger.info(
    #     'Epoch [{}/{}] |'
    #     ' current lr: {:.8f} |'.format(
    #         last_epoch,
    #         config.TRAIN.END_EPOCH,
    #         get_lr(optimizer)
    #     )
    # )
    #
    # print(
    #     'Epoch [{}/{}] |'
    #     ' current lr: {:.8f} |'.format(
    #         last_epoch,
    #         config.TRAIN.END_EPOCH,
    #         get_lr(optimizer)
    #     )
    # )

    iou, bin_iou, val_loss = test(model, validation_dataloader,
                                  class2labels, labels_split,
                                  test_label_split_value, criterion)

    best_loss = min(val_loss, best_loss)

    logger.info(
        'Best Loss: {:.8f} \n'
        'current lr: {:.8f} \n'
        'Mean IoU: {} \n'
        'BIN Mean IoU: {} \n'
        'Std IoU: {} \n'
        'BIN Std IoU: {} \n'
        'Validation Loss: {} '.format(
            best_loss,
            get_lr(optimizer),
            iou['meanIoU'],
            bin_iou['meanIoU_binary'],
            iou['meanIoU_std'],
            bin_iou['meanIoU_std_binary'],
            val_loss
        )
    )

    print(
        'Best Loss: {:.8f} \n'
        'current lr: {:.8f} \n'
        'Mean IoU: {} \n'
        'BIN Mean IoU: {} \n'
        'Std IoU: {} \n'
        'BIN Std IoU: {} \n'
        'Validation Loss: {} '.format(
            best_loss,
            get_lr(optimizer),
            iou['meanIoU'],
            bin_iou['meanIoU_binary'],
            iou['meanIoU_std'],
            bin_iou['meanIoU_std_binary'],
            val_loss
        )
    )

    logger.info('>>>>>>>>>>>>>>>>>>Class IoU<<<<<<<<<<<<<<<<<')
    for ind, val_ in enumerate(iou['classIoU']):
        logger.info(f'Class {ind + 1}: {val_}')
    logger.info('>>>>>>>>>>>>>>>>Class IoU_std<<<<<<<<<<<<<<<')
    for ind, val_ in enumerate(iou['classIoU_std']):
        logger.info(f'Class {ind + 1}: {val_}')
    logger.info('>>>>>>>>>>>>>>Class IoU Binary<<<<<<<<<<<<<<')
    for ind, val_ in enumerate(bin_iou['classIoU_binary']):
        logger.info(f'Class {ind + 1}: {val_}')
    logger.info('>>>>>>>>>>>>Class IoU_std Binary<<<<<<<<<<<<')
    for ind, val_ in enumerate(bin_iou['classIoU_std_binary']):
        logger.info(f'Class {ind + 1}: {val_}')

    print('>>>>>>>>>>>>>>>>>>Class IoU<<<<<<<<<<<<<<<<<')
    for ind, val_ in enumerate(iou['classIoU']):
        print(f'Class {ind + 1}: {val_}')
    print('>>>>>>>>>>>>>>>>Class IoU_std<<<<<<<<<<<<<<<')
    for ind, val_ in enumerate(iou['classIoU_std']):
        print(f'Class {ind + 1}: {val_}')
    print('>>>>>>>>>>>>>>Class IoU Binary<<<<<<<<<<<<<<')
    for ind, val_ in enumerate(bin_iou['classIoU_binary']):
        print(f'Class {ind + 1}: {val_}')
    print('>>>>>>>>>>>>Class IoU_std Binary<<<<<<<<<<<<')
    for ind, val_ in enumerate(bin_iou['classIoU_std_binary']):
        print(f'Class {ind + 1}: {val_}')


def test(model, iterator, class2labels, labels_split, test_label_split_value, criterion):
    model.eval()
    metric = Metric(max_label=21, n_runs=1)
    labels = [class2labels[i] for i in labels_split[test_label_split_value]] + [0]

    with torch.no_grad():
        for run in range(1):
            for batch, idx in tqdm(iterator, desc=f'Validation {run + 1}'):
                label_ids = [class2labels[batch['class'][0]]]  # [class2labels[batch['class'][0]]+1]
                support_images = [batch["support_image"]]
                support_fg_mask = [batch["support_fg_mask"]]
                support_bg_mask = [batch["support_bg_mask"]]
                query_images = batch["query_image"]
                query_label = batch["query_label"][0]

                if config.TRAIN.ARCH == "FPMMs" or config.TRAIN.ARCH == "FPMMs_vgg" or config.TRAIN.ARCH == "FRPMMs":
                    support_images = torch.stack(
                        [torch.stack([shot.cuda(config.GPU) for shot in way]) for way in support_images])
                    support_fg_mask = torch.stack(
                        [torch.stack([shot.cuda(config.GPU) for shot in way]) for way in support_fg_mask])
                    support_bg_mask = torch.stack(
                        [torch.stack([shot.cuda(config.GPU) for shot in way]) for way in support_bg_mask])
                    query_images = torch.stack([query_image.cuda(config.GPU) for query_image in query_images])
                    query_label = query_label.cuda(config.GPU)

                    sup_imgs = support_images.swapaxes(1, 2)[0]
                    sup_mask = support_fg_mask.swapaxes(1, 2)[0][:, :, None, :, :]
                    qur_lbl = torch.stack([query_label]).swapaxes(0, 1)
                    qur_img = query_images[0]

                    logits = model(qur_img,
                                   sup_imgs,
                                   sup_mask)
                    loss_val, loss_part1, loss_part2 = model.get_loss(
                        logits,
                        qur_lbl,
                        idx
                    )
                    out_softmax, pred = model.get_pred(
                        logits,
                        qur_img
                    )

                    loss = loss_val
                    query_pred = out_softmax
                    metric.record(np.array(query_pred.argmax(dim=1)[0].cpu()),
                                  np.array(query_label[0].cpu()),
                                  labels=label_ids, n_run=run)
                elif config.TRAIN.ARCH == 'PAnet' or config.TRAIN.ARCH == 'PAnet_new':
                    support_images = [[shot.cuda(config.GPU) for shot in way] for way in support_images]
                    support_fg_mask = [[shot.cuda(config.GPU) for shot in way] for way in support_fg_mask]
                    support_bg_mask = [[shot.cuda(config.GPU) for shot in way] for way in support_bg_mask]
                    query_images = [query_image.cuda(config.GPU) for query_image in query_images]
                    query_label = query_label.cuda(config.GPU)

                    query_pred, _ = model(support_images, support_fg_mask, support_bg_mask, query_images)
                    # query_loss = criterion(query_pred, query_label)
                    # loss = query_loss + align_loss * 1

                    # logit_mask_agg = 0
                    # n_shot = len(support_images[0])
                    # for s_idx in range(n_shot):
                    #     query_pred, align_loss = model(
                    #         [[support_images[0][s_idx]]],
                    #         [[support_fg_mask[0][s_idx]]],
                    #         [[support_bg_mask[0][s_idx]]],
                    #         query_images
                    #     )
                    #
                    #     logit_mask_agg += query_pred.argmax(dim=1).clone()
                    #
                    # if n_shot > 1:
                    #     bsz = logit_mask_agg.size(0)
                    #     max_vote = logit_mask_agg.view(bsz, -1).max(dim=1)[0]
                    #     max_vote = torch.stack([max_vote, torch.ones_like(max_vote).long()])
                    #     max_vote = max_vote.max(dim=0)[0].view(bsz, 1, 1)
                    #     pred_mask = logit_mask_agg.float() / max_vote
                    #     pred_mask[pred_mask < 0.5] = 0
                    #     pred_mask[pred_mask >= 0.5] = 1
                    # else:
                    #     pred_mask = logit_mask_agg
                    loss = 0.0
                    metric.record(np.array(query_pred.argmax(dim=1)[0].cpu()),
                                  np.array(query_label[0].cpu()),
                                  labels=label_ids, n_run=run)
                elif config.TRAIN.ARCH == 'hsnet':
                    support_images = torch.stack(
                        [torch.stack([shot.cuda(config.GPU) for shot in way]) for way in support_images])
                    support_fg_mask = torch.stack(
                        [torch.stack([shot.cuda(config.GPU) for shot in way]) for way in support_fg_mask])
                    support_bg_mask = torch.stack(
                        [torch.stack([shot.cuda(config.GPU) for shot in way]) for way in support_bg_mask])
                    query_images = torch.stack([query_image.cuda(config.GPU) for query_image in query_images])
                    query_label = query_label.cuda(config.GPU)

                    sup_imgs = support_images.swapaxes(1, 2)[0]
                    sup_mask = support_fg_mask.swapaxes(1, 2)[0]
                    qur_lbl = torch.stack([query_label])[0]
                    qur_img = query_images[0]

                    pred_mask = model.predict_mask_nshot(
                        qur_img,
                        sup_imgs,
                        sup_mask
                    )

                    loss = 0.0
                    metric.record(np.array(pred_mask[0].cpu()),
                                  np.array(query_label[0].cpu()),
                                  labels=label_ids, n_run=run)
                elif config.TRAIN.ARCH == 'asnet':
                    support_images = torch.stack(
                        [torch.stack([shot.cuda(config.GPU) for shot in way]) for way in support_images])
                    support_fg_mask = torch.stack(
                        [torch.stack([shot.cuda(config.GPU) for shot in way]) for way in support_fg_mask])
                    support_bg_mask = torch.stack(
                        [torch.stack([shot.cuda(config.GPU) for shot in way]) for way in support_bg_mask])
                    query_images = torch.stack([query_image.cuda(config.GPU) for query_image in query_images])
                    query_label = query_label.cuda(config.GPU)

                    sup_imgs = support_images.swapaxes(1, 2)[0]
                    sup_mask = support_fg_mask.swapaxes(1, 2)[0]
                    qur_lbl = torch.stack([query_label])[0]
                    qur_img = query_images[0]
                    # print(sup_imgs.shape, sup_mask.shape, qur_lbl.shape, qur_img.shape)
                    pred_mask = model.predict_mask_nshot(
                        qur_img,
                        sup_imgs,
                        sup_mask
                    )
                    loss = 0.0
                    metric.record(np.array(pred_mask[0].cpu()),
                                  np.array(query_label[0].cpu()),
                                  labels=label_ids, n_run=run)
                else:
                    print("no model found")
                    loss = None
                    exit()

            classIoU, meanIoU = metric.get_mIoU(labels=sorted(labels), n_run=run)
            classIoU_binary, meanIoU_binary = metric.get_mIoU_binary(n_run=run)

    classIoU, classIoU_std, meanIoU, meanIoU_std = metric.get_mIoU(labels=sorted(labels))
    classIoU_binary, classIoU_std_binary, meanIoU_binary, meanIoU_std_binary = metric.get_mIoU_binary()

    iou = {
        'classIoU': classIoU,
        'classIoU_std': classIoU_std,
        'meanIoU': meanIoU,
        'meanIoU_std': meanIoU_std}

    bin_iou = {
        'classIoU_binary': classIoU_binary,
        'classIoU_std_binary': classIoU_std_binary,
        'meanIoU_binary': meanIoU_binary,
        'meanIoU_std_binary': meanIoU_std_binary
    }

    # print(iou)
    # print(bin_iou)

    return iou, bin_iou, loss


def save(filename, optimizer, epoch, model, train_loss):
    state = {'epoch': epoch + 1, 'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict(), 'train_loss': train_loss}
    torch.save(state, filename)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


if __name__ == '__main__':
    main()
