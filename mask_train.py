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

if __name__ == '__main__':

#======= hyperparameters & data loaders =======#
    cfg = configurations[1]

    SEED = cfg['SEED'] # random seed for reproduce results
    torch.manual_seed(SEED)

    DATA_ROOT = cfg['DATA_ROOT'] # the parent root where your train/val/test data are stored
    MODEL_ROOT = cfg['MODEL_ROOT'] # the root to buffer your checkpoints
    LOG_ROOT = cfg['LOG_ROOT'] # the root to log your train/val status
    MASK_ROOT = cfg['MASK_ROOT'] # the root to log your train mask status
    BACKBONE_RESUME_ROOT = cfg['BACKBONE_RESUME_ROOT'] # the root to resume training from a saved checkpoint
    HEAD_RESUME_ROOT = cfg['HEAD_RESUME_ROOT']  # the root to resume training from a saved checkpoint

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

    writer = SummaryWriter(MASK_ROOT) # writer for buffering intermedium results

    train_transform = transforms.Compose([ # refer to https://pytorch.org/docs/stable/torchvision/transforms.html for more build-in online data augmentation
        transforms.Resize([int(128 * INPUT_SIZE[0] / 112), int(128 * INPUT_SIZE[0] / 112)]), # smaller side resized
        transforms.RandomCrop([INPUT_SIZE[0], INPUT_SIZE[1]]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean = RGB_MEAN,
                            std = RGB_STD),
    ])

    dataset_train = datasets.ImageFolder(os.path.join(DATA_ROOT, 'celeba_aligned'), train_transform)

    # create a weighted random sampler to process imbalanced data
    weights = make_weights_for_balanced_classes(dataset_train.imgs, len(dataset_train.classes))
    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

    train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size = BATCH_SIZE, sampler = sampler, pin_memory = PIN_MEMORY,
        num_workers = NUM_WORKERS, drop_last = DROP_LAST
    )

    NUM_CLASS = len(train_loader.dataset.classes)
    print("Number of Training Classes: {}".format(NUM_CLASS))
    
    BACKBONE = ResNet_50(INPUT_SIZE)
    print("=" * 60)
    print(BACKBONE)
    print("{} Backbone Generated".format(BACKBONE_NAME))
    MaskNet = MaskNet()
    print("=" * 60)
    print(MaskNet)
    print("MaskNet Generated")
    
    masknet_paras_only_bn, masknet_paras_wo_bn = separate_irse_bn_paras(MaskNet) # separate batch_norm parameters from others; do not do weight decay for batch_norm parameters to improve the generalizability
    OPTIMIZER = optim.SGD([{'params': masknet_paras_wo_bn, 'weight_decay': WEIGHT_DECAY}, {'params': masknet_paras_only_bn}], lr = LR, momentum = MOMENTUM)
    print("=" * 60)
    print(OPTIMIZER)
    print("Optimizer Generated")
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

    # fix parameters in backbone
    for p in BACKBONE.parameters():
        p.requires_grad = False

    if MULTI_GPU:
        # multi-GPU setting
        # BACKBONE = nn.DataParallel(BACKBONE, device_ids = GPU_ID)
        BACKBONE = BACKBONE.to(DEVICE)
        MaskNet = nn.DataParallel(MaskNet, device_ids = GPU_ID)
        MaskNet = MaskNet.to(DEVICE)
    else:
        # single-GPU setting
        BACKBONE = BACKBONE.to(DEVICE)
        MaskNet = MaskNet.to(DEVICE)

    #======= train & validation & save checkpoint =======#
    DISP_FREQ = len(train_loader) // 100 # frequency to display training loss & acc

    NUM_EPOCH_WARM_UP = NUM_EPOCH // 25  # use the first 1/25 epochs to warm up
    NUM_BATCH_WARM_UP = len(train_loader) * NUM_EPOCH_WARM_UP  # use the first 1/25 epochs to warm up
    batch = 0  # batch index   
    lambda_t = 0.3
    L1_LOSS = nn.L1Loss()
    MSE_LOSS = nn.MSELoss()
    for epoch in range(NUM_EPOCH): # start training process
        
        if epoch == STAGES[0]: # adjust LR for each training stage after warm up, you can also choose to adjust LR manually (with slight modification) once plaueau observed
            schedule_lr(OPTIMIZER)
        if epoch == STAGES[1]:
            schedule_lr(OPTIMIZER)
        if epoch == STAGES[2]:
            schedule_lr(OPTIMIZER)

        BACKBONE.eval()  # set to different mode
        MaskNet.train()
        losses = AverageMeter()
        l1_losses = AverageMeter()
        mse_losses = AverageMeter()

        for inputs, labels in tqdm(iter(train_loader)):

            if (epoch + 1 <= NUM_EPOCH_WARM_UP) and (batch + 1 <= NUM_BATCH_WARM_UP): # adjust LR for each training batch during warm up
                warm_up_lr(batch + 1, NUM_BATCH_WARM_UP, LR, OPTIMIZER)

            # add mask
            inputs1, mask1 = random_mask(inputs, mask_return=True)
            inputs2, mask2 = random_mask(inputs, mask_return=True)
            '''
            inputs1_ = inputs1.view(inputs1.shape[0], -1)
            inputs2_ = inputs2.view(inputs2.shape[0], -1)
            inputs_ = inputs.view(inputs.shape[0], -1)
            diff1 = torch.sum((inputs1_ - inputs_).mul(inputs1_-inputs_), dim=1, keepdim=True)
            diff2 = torch.sum((inputs2_ - inputs_).mul(inputs2_-inputs_), dim=1, keepdim=True)
            print(diff1, diff2)
            '''
            inputs_cat = torch.cat([inputs1, mask1, inputs2, mask2], 1)
            # compute output
            inputs = inputs.to(DEVICE)
            inputs1 = inputs1.to(DEVICE)
            inputs2 = inputs2.to(DEVICE)
            inputs_cat = inputs_cat.to(DEVICE)

            with torch.no_grad():
                conv_out = BACKBONE.stage1_forward(inputs)
                conv_out1 = BACKBONE.stage1_forward(inputs1)
                conv_out2 = BACKBONE.stage1_forward(inputs2)
            conv_mask = MaskNet(inputs_cat)
            
            fc_in = torch.mul(conv_out, conv_mask)
            fc_in1 = torch.mul(conv_out1, conv_mask)
            fc_in2 = torch.mul(conv_out2, conv_mask)

            l1_loss = L1_LOSS(fc_in, fc_in1) + L1_LOSS(fc_in, fc_in2)
            '''
            fc_in_ = fc_in.view(fc_in.shape[0], -1)
            fc_in1_ = fc_in1.view(fc_in1.shape[0], -1)
            fc_in2_ = fc_in2.view(fc_in2.shape[0], -1)
            diff_sum1 = torch.sum((fc_in_ - fc_in1_).mul((fc_in_ - fc_in1_)), dim=1, keepdim=True)
            diff_sum2 = torch.sum((fc_in_ - fc_in2_).mul((fc_in_ - fc_in2_)), dim=1, keepdim=True)
            # print(diff_sum1, diff_sum2)
            l1_loss = L1_LOSS(diff_sum1, diff_sum2)
            '''
            conv_mask = conv_mask.view(conv_mask.shape[0], -1)
            mean = torch.mean(conv_mask, dim=1 ,keepdim=True)
            mse_loss = MSE_LOSS(mean, torch.ones_like(mean))
            loss = l1_loss + lambda_t*mse_loss

            losses.update(loss.data.item(), inputs.size(0))
            l1_losses.update(l1_loss.data.item(), inputs.size(0))
            mse_losses.update((lambda_t*mse_loss).data.item())

            # compute gradient and do SGD step
            OPTIMIZER.zero_grad()
            loss.backward()
            OPTIMIZER.step()
            
            # dispaly training loss & acc every DISP_FREQ
            if ((batch + 1) % DISP_FREQ == 0) and batch != 0:
                print("=" * 60)
                print('Epoch {}/{} Batch {}/{}\t'
                      'Training Loss {loss.val:.6f} ({loss.avg:.6f})\t'
                      'Training Loss@L1 {l1_loss.val:.6f} ({l1_loss.avg:.6f})\t'
                      'Training Loss@MSE {mse_loss.val:.6f} ({mse_loss.avg:.6f})'.format(
                    epoch + 1, NUM_EPOCH, batch + 1, len(train_loader) * NUM_EPOCH, loss = losses, l1_loss = l1_losses, mse_loss = mse_losses))
                print("=" * 60)

            batch += 1 # batch index

        # training statistics per epoch (buffer for visualization)
        epoch_loss = losses.avg
        writer.add_scalar("Training_Loss", epoch_loss, epoch + 1)
        print("=" * 60)
        print('Epoch: {}/{}\t'
              'Training Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Training Loss@L1 {l1_loss.val:.3f} ({l1_loss.avg:.3f})\t'
              'Training Loss@MSE {mse_loss.val:.3f} ({mse_loss.avg:.3f})'.format(
            epoch + 1, NUM_EPOCH, loss = losses, l1_loss = l1_losses, mse_loss = mse_losses))
        print("=" * 60)
        # perform validation & save checkpoints per epoch
        # validation statistics per epoch (buffer for visualization)
        print("=" * 60)
        print("Perform Evaluation on LFW, and Save Checkpoints...")
        accuracy_lfw, best_threshold_lfw, roc_curve_lfw = perform_val(MULTI_GPU, DEVICE, EMBEDDING_SIZE, BATCH_SIZE, BACKBONE, lfw, lfw_issame, masknet=MaskNet, mask=1, batch_same=False)
        buffer_val(writer, "LFW", accuracy_lfw, best_threshold_lfw, roc_curve_lfw, epoch + 1)
        '''
        accuracy_cfp_ff, best_threshold_cfp_ff, roc_curve_cfp_ff = perform_val(MULTI_GPU, DEVICE, EMBEDDING_SIZE, BATCH_SIZE, BACKBONE, cfp_ff, cfp_ff_issame)
        buffer_val(writer, "CFP_FF", accuracy_cfp_ff, best_threshold_cfp_ff, roc_curve_cfp_ff, epoch + 1)
        accuracy_cfp_fp, best_threshold_cfp_fp, roc_curve_cfp_fp = perform_val(MULTI_GPU, DEVICE, EMBEDDING_SIZE, BATCH_SIZE, BACKBONE, cfp_fp, cfp_fp_issame)
        buffer_val(writer, "CFP_FP", accuracy_cfp_fp, best_threshold_cfp_fp, roc_curve_cfp_fp, epoch + 1)
        accuracy_agedb, best_threshold_agedb, roc_curve_agedb = perform_val(MULTI_GPU, DEVICE, EMBEDDING_SIZE, BATCH_SIZE, BACKBONE, agedb, agedb_issame)
        buffer_val(writer, "AgeDB", accuracy_agedb, best_threshold_agedb, roc_curve_agedb, epoch + 1)
        accuracy_calfw, best_threshold_calfw, roc_curve_calfw = perform_val(MULTI_GPU, DEVICE, EMBEDDING_SIZE, BATCH_SIZE, BACKBONE, calfw, calfw_issame)
        buffer_val(writer, "CALFW", accuracy_calfw, best_threshold_calfw, roc_curve_calfw, epoch + 1)
        accuracy_cplfw, best_threshold_cplfw, roc_curve_cplfw = perform_val(MULTI_GPU, DEVICE, EMBEDDING_SIZE, BATCH_SIZE, BACKBONE, cplfw, cplfw_issame)
        buffer_val(writer, "CPLFW", accuracy_cplfw, best_threshold_cplfw, roc_curve_cplfw, epoch + 1)
        accuracy_vgg2_fp, best_threshold_vgg2_fp, roc_curve_vgg2_fp = perform_val(MULTI_GPU, DEVICE, EMBEDDING_SIZE, BATCH_SIZE, BACKBONE, vgg2_fp, vgg2_fp_issame)
        buffer_val(writer, "VGGFace2_FP", accuracy_vgg2_fp, best_threshold_vgg2_fp, roc_curve_vgg2_fp, epoch + 1)
        
        print("Epoch {}/{}, Evaluation: LFW Acc: {}".format(epoch + 1, NUM_EPOCH, accuracy_lfw))
        print("=" * 60)
        '''
        # save checkpoints per epoch
        if MULTI_GPU:
            torch.save(MaskNet.module.state_dict(), os.path.join(MODEL_ROOT, "MaskNet_{}_Epoch_{}_Batch_{}_Time_{}_checkpoint.pth".format(BACKBONE_NAME, epoch + 1, batch, get_time())))
        else:
            torch.save(MaskNet.state_dict(), os.path.join(MODEL_ROOT, "MaskNet_{}_Epoch_{}_Batch_{}_Time_{}_checkpoint.pth".format(BACKBONE_NAME, epoch + 1, batch, get_time())))
            