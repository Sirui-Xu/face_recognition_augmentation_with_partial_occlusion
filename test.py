import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from model_resnet import ResNet_50, MaskNet
from metrics import ArcFace
from focal import FocalLoss
from utils import make_weights_for_balanced_classes, get_val_data, separate_irse_bn_paras, separate_resnet_bn_paras, warm_up_lr, schedule_lr, perform_val, get_time, buffer_val, AverageMeter, accuracy, random_mask
from config import configurations

from tensorboardX import SummaryWriter
from tqdm import tqdm
import os

import random

if __name__ == '__main__':
    #======= hyperparameters & data loaders =======#
    cfg = configurations[1]

    SEED = cfg['SEED'] # random seed for reproduce results
    torch.manual_seed(SEED)

    DATA_ROOT = cfg['DATA_ROOT'] # the parent root where your train/val/test data are stored
    MODEL_ROOT = cfg['MODEL_ROOT'] # the root to buffer your checkpoints
    LOG_ROOT = cfg['LOG_ROOT'] # the root to log your train/val status
    TEST_ROOT = cfg['TEST_ROOT']
    BACKBONE_RESUME_ROOT = cfg['BACKBONE_RESUME_ROOT'] # the root to resume training from a saved checkpoint
    HEAD_RESUME_ROOT = cfg['HEAD_RESUME_ROOT']  # the root to resume training from a saved checkpoint
    MASKNET_RESUME_ROOT = cfg['MASKNET_RESUME_ROOT']

    BACKBONE_NAME = cfg['BACKBONE_NAME'] # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101', 'IR_SE_152']
    HEAD_NAME = cfg['HEAD_NAME'] # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
    LOSS_NAME = cfg['LOSS_NAME'] # support: ['Focal', 'Softmax']

    INPUT_SIZE = cfg['INPUT_SIZE']
    RGB_MEAN = cfg['RGB_MEAN'] # for normalize inputs
    RGB_STD = cfg['RGB_STD']
    EMBEDDING_SIZE = cfg['EMBEDDING_SIZE'] # feature dimension
    BATCH_SIZE = cfg['BATCH_SIZE']
    DROP_LAST = cfg['DROP_LAST'] # whether drop the last batch to ensure consistent batch_norm statistics
    LR = cfg['LR'] # initial LR
    NUM_EPOCH = cfg['NUM_EPOCH']
    WEIGHT_DECAY = cfg['WEIGHT_DECAY']
    MOMENTUM = cfg['MOMENTUM']
    STAGES = cfg['STAGES'] # epoch stages to decay learning rate

    DEVICE = cfg['DEVICE']
    MULTI_GPU = cfg['MULTI_GPU'] # flag to use multiple GPUs
    GPU_ID = cfg['GPU_ID'] # specify your GPU ids
    PIN_MEMORY = cfg['PIN_MEMORY']
    NUM_WORKERS = cfg['NUM_WORKERS']
    print("=" * 60)
    print("Overall Configurations:")
    print(cfg)
    print("=" * 60)
    writer = SummaryWriter(TEST_ROOT)
    lfw, lfw_issame = get_val_data(os.path.join(DATA_ROOT, 'lfw_aligned'))
    BACKBONE = ResNet_50(INPUT_SIZE)
    MASKNET = MaskNet()
    print("=" * 60)
    print(BACKBONE)
    print("{} Backbone Generated".format(BACKBONE_NAME))
    print("=" * 60)
    print("=" * 60)
    print(MASKNET)
    print("MaskNet Generated")
    print("=" * 60)
    # optionally resume from a checkpoint
    if BACKBONE_RESUME_ROOT:
        print("=" * 60)
        if os.path.isfile(BACKBONE_RESUME_ROOT):
            print("Loading Backbone Checkpoint '{}'".format(BACKBONE_RESUME_ROOT))
            BACKBONE.load_state_dict(torch.load(BACKBONE_RESUME_ROOT))
        else:
            raise Exception("No Checkpoint Found at '{}'. Please Have a Check or Continue to Train from Scratch".format(BACKBONE_RESUME_ROOT))
        print("=" * 60)

    if MASKNET_RESUME_ROOT:
        print("=" * 60)
        if os.path.isfile(MASKNET_RESUME_ROOT):
            print("Loading Backbone Checkpoint '{}'".format(MASKNET_RESUME_ROOT))
            MASKNET.load_state_dict(torch.load(MASKNET_RESUME_ROOT))
        else:
            raise Exception("No Checkpoint Found at '{}'. Please Have a Check or Continue to Train from Scratch".format(MASKNET_RESUME_ROOT))
        print("=" * 60)

    if MULTI_GPU:
        # multi-GPU setting
        BACKBONE = nn.DataParallel(BACKBONE, device_ids = GPU_ID)
        BACKBONE = BACKBONE.to(DEVICE)
        MASKNET = nn.DataParallel(MASKNET, device_ids=GPU_ID)
        MASKNET = MASKNET.to(DEVICE)
    else:
        # single-GPU setting
        BACKBONE = BACKBONE.to(DEVICE)
        MASKNET = MASKNET.to(DEVICE)

    print("=" * 60)
    parts = ['left_eye', 'right_eye', 'nose', 'left_mouth', 'right_mouth']

    for part in parts:
        print("Perform Evaluation on LFW in {} mask".format(part))
        accuracy_lfw, best_threshold_lfw, roc_curve_lfw = perform_val(MULTI_GPU, DEVICE, EMBEDDING_SIZE, BATCH_SIZE, BACKBONE, lfw, lfw_issame, mask = 2, save_tag=True, part=part, size=25)
        buffer_val(writer, "LFW", accuracy_lfw, best_threshold_lfw, roc_curve_lfw, 0, part)
        print("LFW Acc: {}".format(accuracy_lfw))

    print("Perform Evaluation on LFW with no mask")
    accuracy_lfw, best_threshold_lfw, roc_curve_lfw = perform_val(MULTI_GPU, DEVICE, EMBEDDING_SIZE, BATCH_SIZE, BACKBONE, lfw, lfw_issame, mask = 0, save_tag=False)
    buffer_val(writer, "LFW", accuracy_lfw, best_threshold_lfw, roc_curve_lfw, 0)
    print("LFW Acc: {}".format(accuracy_lfw))

    print("Perform Evaluation on LFW with random mask")
    accuracy_lfw, best_threshold_lfw, roc_curve_lfw = perform_val(MULTI_GPU, DEVICE, EMBEDDING_SIZE, BATCH_SIZE, BACKBONE, lfw, lfw_issame, mask = 1, save_tag=False, batch_same=False)
    buffer_val(writer, "LFW", accuracy_lfw, best_threshold_lfw, roc_curve_lfw, 0)
    print("LFW Acc: {}".format(accuracy_lfw))

    print("Perform Evaluation on LFW with same mask")
    accuracy_lfw, best_threshold_lfw, roc_curve_lfw = perform_val(MULTI_GPU, DEVICE, EMBEDDING_SIZE, BATCH_SIZE, BACKBONE, lfw, lfw_issame, mask = 1, save_tag=False, batch_same=True)
    buffer_val(writer, "LFW", accuracy_lfw, best_threshold_lfw, roc_curve_lfw, 0)
    print("LFW Acc: {}".format(accuracy_lfw))


