import torch
import torchvision.transforms as transforms
import torch.nn.functional as F

from verification import evaluate

from datetime import datetime
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
from PIL import Image
import bcolz
import io
import os
import random
# reference facial points, a list of coordinates (x,y)
REFERENCE_FACIAL_POINTS = [        # default reference facial points for crop_size = (112, 112); should adjust REFERENCE_FACIAL_POINTS accordingly for other crop_size
    [30.29459953,  51.69630051],
    [65.53179932,  51.50139999],
    [48.02519989,  71.73660278],
    [33.54930115,  92.3655014],
    [62.72990036,  92.20410156]
]
# Support: ['get_time', 'l2_norm', 'make_weights_for_balanced_classes', 'get_val_pair', 'get_val_data', 'separate_irse_bn_paras', 'separate_resnet_bn_paras', 'warm_up_lr', 'schedule_lr', 'de_preprocess', 'hflip_batch', 'ccrop_batch', 'gen_plot', 'perform_val', 'buffer_val', 'AverageMeter', 'accuracy']

def random_point():
    rand = random.randint(0, 3)
    if rand == 0:
        cx, cy = REFERENCE_FACIAL_POINTS[rand]
        dlx = random.random() * 20
        drx = random.random() * 70
        duy = random.random() * 20
        ddy = random.random() * 20

    elif rand == 1:
        cx, cy = REFERENCE_FACIAL_POINTS[rand]
        dlx = random.random() * 55
        drx = random.random() * 35
        duy = random.random() * 20
        ddy = random.random() * 20

    elif rand == 2:
        cx, cy = REFERENCE_FACIAL_POINTS[rand]
        dlx = random.random() * 24
        drx = random.random() * 40
        duy = random.random() * 20
        ddy = random.random() * 20

    else:
        cx, cy = REFERENCE_FACIAL_POINTS[rand]
        dlx = random.random() * 20
        drx = random.random() * 70
        duy = random.random() * 30
        ddy = random.random() * 18
    
    return (int(cx-dlx), int(cx+drx), int(cy-duy), int(cy+ddy)), (2*random.random() - 1,2*random.random() - 1,2*random.random() - 1)

def fix_point(size, part):
    hsize = size / 2
    color = (-1, -1, -1)
    if part == 'left_eye':    
        x1 = 25
        x2 = 50
        y1 = 40
        y2 = 65
    elif part == 'right_eye':  
        x1 = 65
        x2 = 90
        y1 = 40
        y2 = 65
    elif part == 'nose':  
        x1 = 43
        x2 = 68
        y1 = 57
        y2 = 82
    elif part == 'left_mouth':
        x1 = 30
        x2 = 55
        y1 = 85
        y2 = 85+25
    elif part == 'right_mouth':
        x1 = 56
        x2 = 56+25
        y1 = 85
        y2 = 85+25
    else:
        raise Exception('No matched parts')

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    x1 = int(cx - hsize) if cx >= hsize else 0
    x2 = int(cx + hsize) if int(cx + hsize) <= 111 else 111
    y1 = int(cy - hsize) if cy >= hsize else 0
    y2 = int(cy + hsize) if int(cy + hsize) <= 111 else 111
    return (x1, x2, y1, y2), color


def random_mask(inputs, part=None, size=25, batch_same=True, mask_return=False):
    outputs = inputs.clone().detach()
    if part is not None:
        (x1, x2, y1, y2), color = fix_point(size, part)

    else:
        (x1, x2, y1, y2), color = random_point()
        mask = torch.ones_like(outputs)
        if batch_same is not True:
            for i in range(outputs.shape[0]):
                for j in range(3):
                    outputs[i, j, y1:y2, x1:x2] = color[j]
                mask[i, :, y1:y2, x1:x2] = 0
                (x1, x2, y1, y2), color = random_point()
            if mask_return is True:
                return outputs, mask[:, 0:1, :, :]
            else:
                return outputs

    for i in range(3):
        outputs[:, i, y1:y2, x1:x2] = color[i]
    
    if mask_return is True:
        mask = torch.ones_like(outputs)
        mask[:, :, y1:y2, x1:x2] = 0
        return outputs, mask[:, 0:1, :, :]
    else:
        return outputs

def get_time():
    return (str(datetime.now())[:-10]).replace(' ', '-').replace(':', '-')


def l2_norm(input, axis = 1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)

    return output


def make_weights_for_balanced_classes(images, nclasses):
    '''
        Make a vector of weights for each image in the dataset, based
        on class frequency. The returned vector of weights can be used
        to create a WeightedRandomSampler for a DataLoader to have
        class balancing when sampling for a training batch.
            images - torchvisionDataset.imgs
            nclasses - len(torchvisionDataset.classes)
        https://discuss.pytorch.org/t/balanced-sampling-between-classes-with-torchvision-dataloader/2703/3
    '''
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1  # item is (img-data, label-id)
    weight_per_class = [0.] * nclasses
    N = float(sum(count))  # total number of images
    for i in range(nclasses):
        weight_per_class[i] = N / float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]

    return weight


def get_val_pair(path, name):
    carray = bcolz.carray(rootdir = os.path.join(path, name), mode = 'r')
    issame = np.load('{}/{}_list.npy'.format(path, name))

    return carray, issame


def get_val_data(data_path):
    lfw, lfw_issame = get_val_pair(data_path, 'lfw')
    '''
    cfp_ff, cfp_ff_issame = get_val_pair(data_path, 'cfp_ff')
    cfp_fp, cfp_fp_issame = get_val_pair(data_path, 'cfp_fp')
    agedb_30, agedb_30_issame = get_val_pair(data_path, 'agedb_30')
    calfw, calfw_issame = get_val_pair(data_path, 'calfw')
    cplfw, cplfw_issame = get_val_pair(data_path, 'cplfw')
    vgg2_fp, vgg2_fp_issame = get_val_pair(data_path, 'vgg2_fp')
    '''
    return lfw, lfw_issame

def separate_irse_bn_paras(modules):
    if not isinstance(modules, list):
        modules = [*modules.modules()]
    paras_only_bn = []
    paras_wo_bn = []
    for layer in modules:
        if 'model' in str(layer.__class__):
            continue
        if 'container' in str(layer.__class__):
            continue
        else:
            if 'batchnorm' in str(layer.__class__):
                paras_only_bn.extend([*layer.parameters()])
            else:
                paras_wo_bn.extend([*layer.parameters()])

    return paras_only_bn, paras_wo_bn


def separate_resnet_bn_paras(modules):
    all_parameters = modules.parameters()
    paras_only_bn = []

    for pname, p in modules.named_parameters():
        if pname.find('bn') >= 0:
            paras_only_bn.append(p)
            
    paras_only_bn_id = list(map(id, paras_only_bn))
    paras_wo_bn = list(filter(lambda p: id(p) not in paras_only_bn_id, all_parameters))
    
    return paras_only_bn, paras_wo_bn


def warm_up_lr(batch, num_batch_warm_up, init_lr, optimizer):
    for params in optimizer.param_groups:
        params['lr'] = batch * init_lr / num_batch_warm_up

    # print(optimizer)


def schedule_lr(optimizer):
    for params in optimizer.param_groups:
        params['lr'] /= 10.

    print(optimizer)


def de_preprocess(tensor):

    return tensor * 0.5 + 0.5

depre = transforms.Compose([
            de_preprocess,
            transforms.ToPILImage()
])

def depre_batch(imgs_tensor):
    imgs = []
    for i, img_ten in enumerate(imgs_tensor):
        imgs.append(depre(img_ten))
    return imgs

hflip = transforms.Compose([
            de_preprocess,
            transforms.ToPILImage(),
            transforms.functional.hflip,
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])


def hflip_batch(imgs_tensor):
    hfliped_imgs = torch.empty_like(imgs_tensor)
    for i, img_ten in enumerate(imgs_tensor):
        hfliped_imgs[i] = hflip(img_ten)

    return hfliped_imgs


ccrop = transforms.Compose([
            de_preprocess,
            transforms.ToPILImage(),
            transforms.Resize([128, 128]),  # smaller side resized
            transforms.CenterCrop([112, 112]),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])


def ccrop_batch(imgs_tensor):
    ccropped_imgs = torch.empty_like(imgs_tensor)
    for i, img_ten in enumerate(imgs_tensor):
        ccropped_imgs[i] = ccrop(img_ten)

    return ccropped_imgs


def gen_plot(fpr, tpr):
    """Create a pyplot plot and save to buffer."""
    plt.figure()
    plt.xlabel("FPR", fontsize = 14)
    plt.ylabel("TPR", fontsize = 14)
    plt.title("ROC Curve", fontsize = 14)
    plot = plt.plot(fpr, tpr, linewidth = 2)
    buf = io.BytesIO()
    plt.savefig(buf, format = 'jpeg')
    buf.seek(0)
    plt.close()

    return buf

def add_mask(inputs1, inputs2, mask1, mask2, masknet, device, backbone, embedding_size, batch_size):
    embedding = torch.zeros([batch_size, embedding_size])
    inputs_cat = torch.cat([inputs1, mask1, inputs2, mask2], 1)
    # compute output
    inputs1 = inputs1.to(device)
    inputs2 = inputs2.to(device)
    inputs_cat = inputs_cat.to(device)
    # print('inputs1.shape', inputs1.shape)

    with torch.no_grad():
        conv_out1 = backbone.stage1_forward(inputs1)
        conv_out2 = backbone.stage1_forward(inputs2)
        conv_mask = masknet(inputs_cat)
    
    fc_in1 = torch.mul(conv_out1, conv_mask)
    fc_in2 = torch.mul(conv_out2, conv_mask)

    with torch.no_grad():
        # print('embed.shape', embedding[0::2].shape)
        # print('back.shape', backbone.stage2_forward(fc_in1).shape)
        embedding[0::2] = backbone.stage2_forward(fc_in1)
        embedding[1::2] = backbone.stage2_forward(fc_in2)

    return embedding


def perform_val(multi_gpu, device, embedding_size, batch_size, backbone, carray, issame, nrof_folds = 10, tta = True, mask = 0, save_tag = False, part = None, size = 25, batch_same = True, masknet = None):
    if multi_gpu:
        backbone = backbone.module # unpackage model from DataParallel
        backbone = backbone.to(device)
    else:
        backbone = backbone.to(device)
    backbone.eval() # switch to evaluation mode

    if masknet is not None:
        if multi_gpu:
            masknet = masknet.module # unpackage model from DataParallel
            masknet = masknet.to(device)
        else:
            masknet = masknet.to(device)
        masknet.eval() # switch to evaluation mode

    idx = 0
    embeddings = np.zeros([len(carray), embedding_size])
    with torch.no_grad():
        while idx + batch_size <= len(carray):
            batch = torch.tensor(carray[idx:idx + batch_size][:, [2, 1, 0], :, :])
            if save_tag is True:
                if mask == 1:
                    batch = random_mask(batch, batch_same = batch_same)
                elif mask == 2:
                    batch = random_mask(batch, part, size)
                imgs = depre_batch(batch)
                for i, img in enumerate(imgs):
                    if part is not None:
                        img.save('./data/lfw_occlusion/' + part + '/' + str(idx + i) + '.jpg')
                    else:
                        img.save('./data/lfw_noocclusion/'  + str(idx + i) + '.jpg')
                
            if tta:
                ccropped = ccrop_batch(batch)
                fliped = hflip_batch(ccropped)
                if masknet is not None:
                    if mask == 1:
                        ccropped, ccropped_mask = random_mask(ccropped, batch_same=batch_same, mask_return=True)
                        fliped, fliped_mask = random_mask(fliped, batch_same=batch_same, mask_return=True)
                    elif mask == 2:
                        ccropped, ccropped_mask = random_mask(ccropped, part, size, mask_return=True)
                        fliped, fliped_mask = random_mask(fliped, part, size, mask_return=True)

                    emb_batch = add_mask(ccropped[0::2], ccropped[1::2], ccropped_mask[0::2], ccropped_mask[1::2], masknet, device, backbone, embedding_size, batch_size) + add_mask(fliped[0::2], fliped[1::2], fliped_mask[0::2], fliped_mask[1::2], masknet, device, backbone, embedding_size, batch_size)
                    embeddings[idx:idx+batch_size] = l2_norm(emb_batch.cpu())

                else:
                    if mask == 1:
                        ccropped = random_mask(ccropped, batch_same=batch_same)
                        fliped = random_mask(fliped, batch_same=batch_same)
                    elif mask == 2:
                        ccropped = random_mask(ccropped, part, size)
                        fliped = random_mask(fliped, part, size)
                    emb_batch = backbone(ccropped.to(device)).cpu() + backbone(fliped.to(device)).cpu()
                    embeddings[idx:idx + batch_size] = l2_norm(emb_batch)
            else:
                ccropped = ccrop_batch(batch)
                if masknet is not None:
                    if mask == 1:
                        ccropped, ccropped_mask = random_mask(ccropped, batch_same=batch_same, mask_return=True)
                    elif mask == 2:
                        ccropped, ccropped_mask = random_mask(ccropped, part, size, mask_return = True)
                    emb_batch = add_mask(ccropped[0::2], ccropped[1::2], ccropped_mask[0::2], ccropped_mask[1::2], masknet, device, backbone, embedding_size, batch_size)
                    embeddings[idx:idx+batch_size] = l2_norm(emb_batch.cpu())
                else:
                    if mask == 1:
                        ccropped = random_mask(ccropped, batch_same=batch_same)
                    elif mask == 2:
                        ccropped = random_mask(ccropped, part, size)
                    embeddings[idx:idx + batch_size] = l2_norm(backbone(ccropped.to(device))).cpu()

            idx += batch_size
        if idx < len(carray):
            batch = torch.tensor(carray[idx:])
            if save_tag is True:
                if mask == 1:
                    batch = random_mask(batch, batch_same = batch_same)
                elif mask == 2:
                    batch = random_mask(batch, part, size)
                imgs = depre_batch(batch)
                for i, img in enumerate(imgs):
                    if part is not None:
                        img.save('./data/lfw_occlusion/' + part + '/' + str(idx + i) + '.jpg')
                    else:
                        img.save('./data/lfw_noocclusion/'  + str(idx + i) + '.jpg')

            if tta:
                ccropped = ccrop_batch(batch)
                fliped = hflip_batch(ccropped)
                if masknet is not None:
                    if mask == 1:
                        ccropped, ccropped_mask = random_mask(ccropped, batch_same=batch_same, mask_return=True)
                        fliped, fliped_mask = random_mask(fliped, batch_same=batch_same, mask_return=True)
                    elif mask == 2:
                        ccropped, ccropped_mask = random_mask(ccropped, part, size, mask_return=True)
                        fliped, fliped_mask = random_mask(fliped, part, size, mask_return=True)
                    emb_batch = add_mask(ccropped[0::2], ccropped[1::2], ccropped_mask[0::2], ccropped_mask[1::2], masknet, device, backbone, embedding_size, len(carray)-idx) + add_mask(fliped[0::2], fliped[1::2], fliped_mask[0::2], fliped_mask[1::2], masknet, device, backbone, embedding_size, len(carray)-idx)
                    embeddings[idx:] = l2_norm(emb_batch.cpu())

                else:
                    if mask == 1:
                        ccropped = random_mask(ccropped, batch_same=batch_same)
                        fliped = random_mask(fliped, batch_same=batch_same)
                    elif mask == 2:
                        ccropped = random_mask(ccropped, part, size)
                        fliped = random_mask(fliped, part, size)
                    emb_batch = backbone(ccropped.to(device)).cpu() + backbone(fliped.to(device)).cpu()
                    embeddings[idx:] = l2_norm(emb_batch)
            else:
                ccropped = ccrop_batch(batch)
                if masknet is not None:
                    if mask == 1:
                        ccropped, ccropped_mask = random_mask(ccropped, batch_same=batch_same, mask_return=True)
                    elif mask == 2:
                        ccropped, ccropped_mask = random_mask(ccropped, part, size, mask_return = True)
                    emb_batch = add_mask(ccropped[0::2], ccropped[1::2], ccropped_mask[0::2], ccropped_mask[1::2], masknet, device, backbone, embedding_size, len(carray)-idx)
                    embeddings[idx:] = l2_norm(emb_batch.cpu())
                else:
                    if mask == 1:
                        ccropped = random_mask(ccropped, batch_same=batch_same)
                    elif mask == 2:
                        ccropped = random_mask(ccropped, part, size)
                    embeddings[idx:] = l2_norm(backbone(ccropped.to(device))).cpu()

                
    tpr, fpr, accuracy, best_thresholds = evaluate(embeddings, issame, nrof_folds)
    buf = gen_plot(fpr, tpr)
    roc_curve = Image.open(buf)
    roc_curve_tensor = transforms.ToTensor()(roc_curve)

    return accuracy.mean(), best_thresholds.mean(), roc_curve_tensor


def buffer_val(writer, db_name, acc, best_threshold, roc_curve_tensor, epoch, part=None):
    if part is None:
        writer.add_scalar('{}_Accuracy'.format(db_name), acc, epoch)
        writer.add_scalar('{}_Best_Threshold'.format(db_name), best_threshold, epoch)
        writer.add_image('{}_ROC_Curve'.format(db_name), roc_curve_tensor, epoch)
    else:
        writer.add_scalar('{}_{}_Accuracy'.format(db_name, part), acc, epoch)
        writer.add_scalar('{}_{}_Best_Threshold'.format(db_name, part), best_threshold, epoch)
        writer.add_image('{}_{}_ROC_Curve'.format(db_name, part), roc_curve_tensor, epoch)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val   = 0
        self.avg   = 0
        self.sum   = 0
        self.count = 0

    def update(self, val, n = 1):
        self.val   = val
        self.sum   += val * n
        self.count += n
        self.avg   = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred    = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res
