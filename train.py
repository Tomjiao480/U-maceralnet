import os
import datetime

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.unet import Unet
from nets.unet_training import get_lr_scheduler, set_optimizer_lr, weights_init
from utils.callbacks import LossHistory, EvalCallback
from utils.dataloader import UnetDataset, unet_dataset_collate
from utils.utils import download_weights, show_config
from utils.utils_fit import fit_one_epoch

if __name__ == "__main__":
    # Enable CUDA if available
    Cuda = True

    # Distributed training settings
    distributed = False
    sync_bn = False
    fp16 = False

    # Model settings
    num_classes = 4
    backbone = "vgg"
    pretrained = False
    model_path = "model_data/unet_vgg_voc.pth"
    input_shape = [512, 512]

    # Training parameters
    Init_Epoch = 0
    Freeze_Epoch = 100
    Freeze_batch_size = 2
    UnFreeze_Epoch = 300
    Unfreeze_batch_size = 4
    Freeze_Train = False

    # Learning rate settings
    Init_lr = 1e-4
    Min_lr = Init_lr * 0.01
    optimizer_type = "adam"
    momentum = 0.9
    weight_decay = 0
    lr_decay_type = 'cos'

    # Logging and evaluation settings
    save_period = 5
    save_dir = 'logs'
    eval_flag = True
    eval_period = 5

    # Dataset path
    VOCdevkit_path = 'VOCdevkit'

    # Loss function settings
    dice_loss = True
    focal_loss = False
    cls_weights = np.ones([num_classes], np.float32)

    num_workers = 4

    # Device setup
    ngpus_per_node = torch.cuda.device_count()
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank = 0

    # Load model
    model = Unet(num_classes=num_classes, pretrained=pretrained, backbone=backbone).train()
    if not pretrained:
        weights_init(model)
    if model_path != '':
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)

    # Logging setup
    if local_rank == 0:
        time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
        log_dir = os.path.join(save_dir, "loss_" + str(time_str))
        loss_history = LossHistory(log_dir, model, input_shape=input_shape)
    else:
        loss_history = None

    # Enable mixed precision training if needed
    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler

        scaler = GradScaler()
    else:
        scaler = None

    model_train = model.train()
    if Cuda:
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.cuda()

    # Load dataset
    with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/train.txt"), "r") as f:
        train_lines = f.readlines()
    with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"), "r") as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)

    # Show training configuration
    if local_rank == 0:
        show_config(
            num_classes=num_classes, backbone=backbone, model_path=model_path, input_shape=input_shape,
            Init_Epoch=Init_Epoch, Freeze_Epoch=Freeze_Epoch, UnFreeze_Epoch=UnFreeze_Epoch,
            Freeze_batch_size=Freeze_batch_size,
            Unfreeze_batch_size=Unfreeze_batch_size, Freeze_Train=Freeze_Train, Init_lr=Init_lr, Min_lr=Min_lr,
            optimizer_type=optimizer_type,
            momentum=momentum, lr_decay_type=lr_decay_type, save_period=save_period, save_dir=save_dir,
            num_workers=num_workers,
            num_train=num_train, num_val=num_val
        )

    # Define optimizer
    optimizer = {
        'adam': optim.Adam(model.parameters(), Init_lr, betas=(momentum, 0.999), weight_decay=weight_decay),
        'sgd': optim.SGD(model.parameters(), Init_lr, momentum=momentum, nesterov=True, weight_decay=weight_decay)
    }[optimizer_type]

    lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr, Min_lr, UnFreeze_Epoch)

    # Training loop
    for epoch in range(Init_Epoch, UnFreeze_Epoch):
        set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
        fit_one_epoch(model_train, model, loss_history, None, optimizer, epoch, num_train // 4, num_val // 4, None,
                      None, UnFreeze_Epoch, Cuda, dice_loss, focal_loss, cls_weights, num_classes, fp16, scaler,
                      save_period, save_dir, local_rank)
        if distributed:
            dist.barrier()

    if local_rank == 0:
        loss_history.writer.close()
