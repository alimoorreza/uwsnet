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
from utils.new_net_utils import poly_learning_rate as poly_lr_new_net


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
        elif config.TRAIN.PA_NET_TYPE == 'triplet_sup_que':
            from networks.few_shot_with_triplet import FewShotSeg
            model = FewShotSeg(pretrained_path=vgg16_model_weight, cfg={'align': True}, vgg_type='vgg')
        elif config.TRAIN.PA_NET_TYPE == 'triplet_sup_que_dice':
            from networks.few_shot_with_triplet_diceloss import FewShotSeg
            model = FewShotSeg(pretrained_path=vgg16_model_weight, cfg={'align': True}, vgg_type='vgg',
                               gpu=config.GPU)
        elif config.TRAIN.PA_NET_TYPE == 'triplet_sup_que_vgg16':
            from networks.few_shot_with_triplet import FewShotSeg
            model = FewShotSeg(pretrained_path=vgg16_model_weight, cfg={'align': True}, vgg_type='triplet')
        elif config.TRAIN.PA_NET_TYPE == 'triplet_sup_que_vgg16_dice':
            from networks.few_shot_with_triplet_diceloss import FewShotSeg
            model = FewShotSeg(pretrained_path=vgg16_model_weight, cfg={'align': True}, vgg_type='triplet',
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
        if config.TRAIN.ARCH == 'PAnet':
            lr_milestones = config.TRAIN.LR_MILESTONE
            scheduler = MultiStepLR(optimizer, milestones=lr_milestones, gamma=config.TRAIN.GAMMA)
        else:
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

    total_params = sum(
        param.numel() for param in model.parameters()
    )
    print("total number of parameters: ", total_params)
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print("total number of trainable parameters: ", trainable_params)

    best_iou = 0
    best_model = False
    best_loss = float('inf')
    last_epoch = config.TRAIN.BEGIN_EPOCH

    logger, final_output_dir, tb_log_dir = create_logger(config, args.config, 'train')
    writer = get_writer()

    logger.info(config)
    logger.info("=> creating {} model ...".format(config.TRAIN.ARCH))
    logger.info(model)

    print(config)
    print("=> creating {} model ...".format(config.TRAIN.ARCH))
    print(model)

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
            checkpoint = torch.load(model_state_file)
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
            RandomMirror(),
            ToTensorNormalize(mean=mean, std=std)
        ]
    )

    directory = config.DATASET.ROOT
    labels_split = labels_split
    test_label_split_value = config.TRAIN.TEST_LABEL_SPLIT_VALUE
    episode = config.TRAIN.EPISODE
    n_ways = config.TRAIN.N_WAYS
    n_shots = config.TRAIN.N_SHOTS
    random_split = config.TRAIN.RANDOM_SPLIT

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
            episode,
            n_ways,
            n_shots,
            validation=False,
            transform=train_transform,
            random_split=random_split
        )
        validation_dataset = IUDataset(
            directory,
            class2labels,
            labels_split,
            test_label_split_value,
            episode//1,
            n_ways,
            n_shots,
            validation=True,
            transform=val_transform,
            random_split=random_split
        )
    elif config.TRAIN.ARCH == 'hsnet' or config.TRAIN.ARCH == 'asnet':
        train_dataset = IUDataset(
            directory,
            class2labels,
            labels_split,
            test_label_split_value,
            episode,
            n_ways,
            n_shots,
            validation=False,
            transform=train_transform,
            random_split=random_split
        )
        validation_dataset = IUDataset(
            directory,
            class2labels,
            labels_split,
            test_label_split_value,
            episode // 1,
            n_ways,
            n_shots,
            validation=True,
            transform=val_transform,
            random_split=random_split
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
    print("TRAINING START.....")
    print("N SHOTS", config.TRAIN.N_SHOTS)

    logger.info(
        'Epoch [{}/{}] |'
        ' current lr: {:.8f} |'.format(
            last_epoch,
            config.TRAIN.END_EPOCH,
            get_lr(optimizer)
        )
    )

    print(
        'Epoch [{}/{}] |'
        ' current lr: {:.8f} |'.format(
            last_epoch,
            config.TRAIN.END_EPOCH,
            get_lr(optimizer)
        )
    )

    for ep_it in range(last_epoch, config.TRAIN.END_EPOCH):
        epoch = ep_it + 1

        logger.info(f'New epoch {epoch} starts on cuda')

        print(f'New epoch {epoch} starts on cuda')

        train_loss, train_align_loss, optimizer = train(model,
                                                        train_dataloader, optimizer,
                                                        scheduler, criterion, epoch)
        writer.add_scalar('Loss/batch_train_loss', train_loss, epoch)
        writer.add_scalar('Loss/batch_align_loss', train_align_loss, epoch)

        iou, bin_iou, val_loss = validation(model, validation_dataloader,
                                            class2labels, labels_split,
                                            test_label_split_value, criterion)

        # best_loss = min(train_loss, best_loss)
        best_loss = min(val_loss, best_loss)

        mean_iou = iou['meanIoU']
        mean_iou_bin = bin_iou['meanIoU_binary']

        if mean_iou > best_iou:
            best_iou = mean_iou
            best_model = True
        else:
            best_model = False

        if epoch % config.TRAIN.SAVE_FREQ == 0:
            logger.info('=> saving checkpoint to {}'.format(final_output_dir))
            print('=> saving checkpoint to {}'.format(final_output_dir))
            save_checkpoint({
                'epoch': epoch,
                'model': config.MODEL.NAME,
                'state_dict': model.state_dict(),
                'perf': mean_iou,
                'optimizer': optimizer.state_dict(),
            }, best_model, final_output_dir, filename='checkpoint.pth.tar')

        if best_model:
            logger.info('=> saving best model to {}'.format(final_output_dir))
            print('=> saving best model to {}'.format(final_output_dir))
            torch.save(
                {
                    'epoch': epoch,
                    'loss': train_loss,
                    'iou': mean_iou,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                },
                os.path.join(
                    final_output_dir,
                    f'{test_label_split_value}_split_{n_ways}_nways_{n_shots}_nshots_{epoch}_epoch_iou_{mean_iou:.3f}_model.pth.tar'
                )
            )

        if epoch == config.TRAIN.END_EPOCH:
            filename = final_output_dir + '/train_epoch_final.pth'
            logger.info('Saving checkpoint to: ' + filename)
            print('Saving checkpoint to: ' + filename)
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict()}, filename)
            exit()

        logger.info(
            'Epoch [{}/{}] Train Loss: {:.8f} \n'
            'Best Loss: {:.8f} \n'
            'current lr: {:.8f} \n'
            'train. Align loss: {:.5f} \n'
            'Mean IoU: {} \n'
            'BIN Mean IoU: {} \n'
            'Std IoU: {} \n'
            'BIN Std IoU: {} \n'
            'Validation Loss: {} '.format(
                epoch, config.TRAIN.END_EPOCH, train_loss,
                best_loss,
                get_lr(optimizer),
                train_align_loss,
                iou['meanIoU'],
                bin_iou['meanIoU_binary'],
                iou['meanIoU_std'],
                bin_iou['meanIoU_std_binary'],
                val_loss
            )
        )

        print(
            'Epoch [{}/{}] Train Loss: {:.8f} \n'
            'Best Loss: {:.8f} \n'
            'current lr: {:.8f} \n'
            'train. Align loss: {:.5f} \n'
            'Mean IoU: {} \n'
            'BIN Mean IoU: {} \n'
            'Std IoU: {} \n'
            'BIN Std IoU: {} \n'
            'Validation Loss: {} '.format(
                epoch, config.TRAIN.END_EPOCH, train_loss,
                best_loss,
                get_lr(optimizer),
                train_align_loss,
                iou['meanIoU'],
                bin_iou['meanIoU_binary'],
                iou['meanIoU_std'],
                bin_iou['meanIoU_std_binary'],
                val_loss
            )
        )

        logger.info('>>>>>>>>>>>>>>>>>>Class IoU<<<<<<<<<<<<<<<<<')
        for ind, val_ in enumerate(iou['classIoU']):
            logger.info(f'Class {ind+1}: {val_}')
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


def train(model, iterator, optimizer, scheduler, criterion, epoch):
    model.train()
    epoch_loss = 0
    epoch_align_loss = 0
    counter = 0
    for i, (batch, idx) in enumerate(tqdm(iterator, desc='Training Processing')):
        counter += 1
        support_images = [batch["support_image"]]
        support_fg_mask = [batch["support_fg_mask"]]
        support_bg_mask = [batch["support_bg_mask"]]
        query_images = batch["query_image"]
        query_label = batch["query_label"][0]

        # print('\n')
        # print(torch.max(support_images[0][0]), torch.max(query_images[0][0]))
        # print(torch.min(support_images[0][0]), torch.min(query_images[0][0]))
        # print('\n')

        if config.TRAIN.ARCH == "FPMMs" or config.TRAIN.ARCH == "FPMMs_vgg" or config.TRAIN.ARCH == "FRPMMs":
            support_images = torch.stack(
                [torch.stack([shot.cuda(config.GPU) for shot in way]) for way in support_images])
            support_fg_mask = torch.stack(
                [torch.stack([shot.cuda(config.GPU) for shot in way]) for way in support_fg_mask])
            support_bg_mask = torch.stack(
                [torch.stack([shot.cuda(config.GPU) for shot in way]) for way in support_bg_mask])
            query_images = torch.stack([query_image.cuda(config.GPU) for query_image in query_images])
            query_label = query_label.cuda(config.GPU)

            if scheduler is None:
                max_iter = config.TRAIN.END_EPOCH * config.TRAIN.EPISODE
                current_iter = (epoch - 1) * config.TRAIN.EPISODE + counter
                my_optim.adjust_learning_rate_poly(
                    config,
                    max_iter,
                    optimizer,
                    current_iter,
                    config.TRAIN.POWER
                )
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
            query_pred = out_softmax
            # print('\n#################')
            # print(torch.unique(query_pred), query_pred.shape)
            # print(torch.unique(query_pred[0][0]), query_pred[0][0].shape)
            # print(torch.unique(query_pred[0][1]), query_pred[0][1].shape)
            # print(np.unique(query_pred.max(1)[1].cpu().numpy()), query_pred.max(1)[1].shape)
            # print('\n#################')
            loss = loss_val
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            epoch_loss += loss.detach().data.cpu().numpy()
            epoch_align_loss += 0
        elif config.TRAIN.ARCH == 'PAnet_new':
            support_images = [[shot.cuda(config.GPU) for shot in way] for way in support_images]
            support_fg_mask = [[shot.cuda(config.GPU) for shot in way] for way in support_fg_mask]
            support_bg_mask = [[shot.cuda(config.GPU) for shot in way] for way in support_bg_mask]
            query_images = [query_image.cuda(config.GPU) for query_image in query_images]
            query_label = query_label.cuda(config.GPU)

            if scheduler is None:
                current_iter = epoch * config.TRAIN.EPISODE + i + 1
                index_split = -1
                max_iter = config.TRAIN.END_EPOCH * config.TRAIN.EPISODE
                if config.TRAIN.BASE_LR > 1e-6:
                    optimizer = poly_lr_new_net(optimizer, config.TRAIN.BASE_LR, current_iter, max_iter,
                                                power=config.TRAIN.POWER, index_split=index_split,
                                                warmup=config.TRAIN.WARMUP,
                                                warmup_step=config.TRAIN.EPISODE // 2)
            query_pred, align_loss = model(support_images, support_fg_mask, support_bg_mask, query_images)
            # print('\n**********************************')
            # print(query_pred.shape, query_label.shape)
            # print('\n**********************************')
            query_loss = criterion(query_pred, query_label)
            loss = query_loss + align_loss * 1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            # print(torch.unique(query_pred), query_pred.shape)
            # print(torch.unique(query_pred[0][0]), query_pred[0][0].shape)
            # print(torch.unique(query_pred[0][1]), query_pred[0][1].shape)
            # print(np.unique(query_pred.max(1)[1].cpu().numpy()), query_pred.max(1)[1].shape)
            epoch_loss += loss.detach().data.cpu().numpy()
            epoch_align_loss += align_loss.detach().data.cpu().numpy() if align_loss != 0 else 0

        elif config.TRAIN.ARCH == 'PAnet':
            support_images = [[shot.cuda(config.GPU) for shot in way] for way in support_images]
            support_fg_mask = [[shot.cuda(config.GPU) for shot in way] for way in support_fg_mask]
            support_bg_mask = [[shot.cuda(config.GPU) for shot in way] for way in support_bg_mask]
            query_images = [query_image.cuda(config.GPU) for query_image in query_images]
            query_label = query_label.cuda(config.GPU)

            query_pred, align_loss = model(support_images, support_fg_mask, support_bg_mask, query_images)
            query_loss = criterion(query_pred, query_label)
            loss = query_loss + align_loss * 1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            # print(torch.unique(query_pred), query_pred.shape)
            # print(torch.unique(query_pred[0][0]), query_pred[0][0].shape)
            # print(torch.unique(query_pred[0][1]), query_pred[0][1].shape)
            # print(np.unique(query_pred.max(1)[1].cpu().numpy()), query_pred.max(1)[1].shape)
            epoch_loss += loss.detach().data.cpu().numpy()
            epoch_align_loss += align_loss.detach().data.cpu().numpy() if align_loss != 0 else 0
        elif config.TRAIN.ARCH == 'hsnet':
            support_images = torch.stack(
                [torch.stack([shot.cuda(config.GPU) for shot in way]) for way in support_images])
            support_fg_mask = torch.stack(
                [torch.stack([shot.cuda(config.GPU) for shot in way]) for way in support_fg_mask])
            support_bg_mask = torch.stack(
                [torch.stack([shot.cuda(config.GPU) for shot in way]) for way in support_bg_mask])
            query_images = torch.stack([query_image.cuda(config.GPU) for query_image in query_images])
            query_label = query_label.cuda(config.GPU)

            sup_imgs = support_images[0][0]
            sup_mask = support_fg_mask[0][0]
            qur_lbl = torch.stack([query_label])[0]
            qur_img = query_images[0]

            logit_mask = model(
                qur_img,
                sup_imgs,
                sup_mask
            )
            pred_mask = logit_mask.argmax(dim=1)

            # 2. Compute loss & update model parameters
            loss = model.compute_objective(logit_mask, qur_lbl)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().data.cpu().numpy()
            epoch_align_loss += 0
        elif config.TRAIN.ARCH == 'asnet':
            support_images = torch.stack(
                [torch.stack([shot.cuda(config.GPU) for shot in way]) for way in support_images])
            support_fg_mask = torch.stack(
                [torch.stack([shot.cuda(config.GPU) for shot in way]) for way in support_fg_mask])
            support_bg_mask = torch.stack(
                [torch.stack([shot.cuda(config.GPU) for shot in way]) for way in support_bg_mask])
            query_images = torch.stack([query_image.cuda(config.GPU) for query_image in query_images])
            query_label = query_label.cuda(config.GPU)

            sup_imgs = support_images[0][0]
            sup_mask = support_fg_mask[0][0]
            qur_lbl = torch.stack([query_label])[0]
            qur_img = query_images[0]
            # print(sup_imgs.shape, sup_mask.shape, qur_lbl.shape, qur_img.shape)
            logit_mask = model(
                qur_img,
                sup_imgs,
                sup_mask
            )
            pred_mask = logit_mask.argmax(dim=1)

            # 2. Compute loss & update model parameters
            loss = model.compute_objective(logit_mask, qur_lbl)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().data.cpu().numpy()
            epoch_align_loss += 0
        else:
            print("no model found")
            loss = None
            exit()

    return (epoch_loss/(counter+1)), (epoch_align_loss/(counter+1)), optimizer


def validation(model, iterator, class2labels, labels_split, test_label_split_value, criterion):
    model.eval()
    metric = Metric(max_label=21, n_runs=5)
    labels = [class2labels[i] for i in labels_split[test_label_split_value]] + [0]

    with torch.no_grad():
        for run in range(5):
            for batch, idx in tqdm(iterator, desc=f'Validation {run+1}'):
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

                    query_pred, align_loss = model(support_images, support_fg_mask, support_bg_mask, query_images)
                    query_loss = criterion(query_pred, query_label)
                    loss = query_loss + align_loss * 1
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

                    sup_imgs = support_images[0][0]
                    sup_mask = support_fg_mask[0][0]
                    qur_lbl = torch.stack([query_label])[0]
                    qur_img = query_images[0]

                    logit_mask = model(
                        qur_img,
                        sup_imgs,
                        sup_mask
                    )
                    pred_mask = logit_mask.argmax(dim=1)

                    # 2. Compute loss & update model parameters
                    loss = model.compute_objective(logit_mask, qur_lbl)
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

                    sup_imgs = support_images[0][0]
                    sup_mask = support_fg_mask[0][0]
                    qur_lbl = torch.stack([query_label])[0]
                    qur_img = query_images[0]

                    logit_mask = model(
                        qur_img,
                        sup_imgs,
                        sup_mask
                    )
                    pred_mask = logit_mask.argmax(dim=1)

                    # 2. Compute loss & update model parameters
                    loss = model.compute_objective(logit_mask, qur_lbl)
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


def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    epoch_align_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(tqdm(iterator,desc='Training Processing')):
            support_images = [batch["support_image"]]
            support_fg_mask = [batch["support_fg_mask"]]
            support_bg_mask = [batch["support_bg_mask"]]
            query_images = batch["query_image"]
            query_label = batch["query_label"][0]

            support_images = [[shot.cuda(config.GPU) for shot in way] for way in support_images]
            support_fg_mask = [[shot.cuda(config.GPU) for shot in way] for way in support_fg_mask]
            support_bg_mask = [[shot.cuda(config.GPU) for shot in way] for way in support_bg_mask]
            query_images = [query_image.cuda(config.GPU) for query_image in query_images]
            query_label = query_label.cuda(config.GPU)

            query_pred,align_loss = model(support_images,support_fg_mask,support_bg_mask,query_images)
            query_loss = criterion(query_pred, query_label)
            loss = query_loss + align_loss * 1
            if i % 500 == 0:
                print("train loss per 500 iter of each epoch:",loss)
            epoch_loss += loss.detach().data.cpu().numpy()
            epoch_align_loss += align_loss.detach().data.cpu().numpy() if align_loss != 0 else 0
    return (epoch_loss/(i+1)), (epoch_align_loss/(i+1))


def save(filename, optimizer, epoch, model, train_loss):
    state = {'epoch': epoch + 1, 'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict(), 'train_loss': train_loss}
    torch.save(state, filename)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


if __name__ == '__main__':
    main()
