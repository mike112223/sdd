import sys
sys.path.insert(0, '.')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse

import pdb
import random
import os
import pickle
import cv2
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset
from albumentations import (Normalize, Compose)
from albumentations.pytorch import ToTensor
import torch.utils.data as data

from lib.model import Unet # import Unet model from the script
from lib.deeplab import deeplabv3_resnet50, deeplabv3_se_resnet50
from lib.data import provider
from lib.utils import init

def parse_args():
    parser = argparse.ArgumentParser(description='Train semantic seg')
    parser.add_argument('--data_folder', default='/DATA5_DB8/data/yanjia/data/steel_defect/train')
    parser.add_argument('--ckpt_path', default=None)
    parser.add_argument('--phase', default='val', type=str)
    parser.add_argument('--backbone',default='resnet18',help='backbone')
    parser.add_argument('--arch', choices=['Unet', 'deeplabv3_resnet50', 'deeplabv3_se_resnet50'], default='Unet')
    parser.add_argument('--classes', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--train_batch', type=int, default=16)
    parser.add_argument('--val_batch', type=int, default=4)

    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--need_split', action='store_true', default=False)
    parser.add_argument('--downsample', type=int, default=2)
    parser.add_argument('--aspp_dilation', type=int, default=6)
    parser.add_argument('--replace',help='replace_stride_with_dilation', type=str, default='0,0,1')
    parser.add_argument('--freeze', action='store_true', default=False, help='whether freeze bn')
    parser.add_argument('--multigrid', action='store_true', default=False)

    args = parser.parse_args()

    return args

def class_metric(probability, truth, min_size, thres_range=np.arange(0.3, 0.7, 0.05), reduction='none'):
    '''Calculates dice of positive and negative images seperately'''
    '''probability and truth must be torch tensors'''
    batch_size = len(truth)
    dice_list = []
    for thres in thres_range:
        new_prob = filter_maps(probability, thres, min_size)
        new_prob = new_prob.reshape(batch_size, -1)
        truth = truth.reshape(batch_size, -1)
        assert(new_prob.shape == truth.shape)

        p = (new_prob > thres).astype(np.float)
        t = (truth > 0.5).astype(np.float)

        t_sum = t.sum(-1)
        p_sum = p.sum(-1)

        dice = 2 * (p*t).sum(-1)/((p+t).sum(-1))

        dice[np.isnan(dice)] = 1
        dice_list.append(dice)

    return dice_list


def metric(probability, truth, min_size_dict, thres_range=np.arange(0.1, 0.7, 0.05), reduction='none'):
    '''Calculates dice of positive and negative images seperately'''
    '''probability and truth must be torch tensors'''
    nclasses = truth.shape[1]
    all_dice = []
    for ii in range(nclasses):
        #print('truth', truth[:, ii, :, :].astype(np.float).sum())
        class_dice = class_metric(probability[:, ii, :, :], truth[:, ii, :, :], min_size_dict[ii], thres_range)
        all_dice.append(class_dice)
    return all_dice


def mask2rle(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels= img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def filter_per_map(probability, threshold, min_size):
    '''Post processing of each predicted mask, components with lesser number of pixels
    than `min_size` are ignored'''
    mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]
    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
    predictions = np.zeros_like(probability)
    num = 0
    after_component = 0
    for c in range(1, num_component):
        p = (component == c)
        if p.sum() > min_size:
            after_component += 1
            predictions[p] = 1
            num += 1
    return predictions


def filter_maps(probability, threshold, min_size):
    '''Post processing of each predicted mask, components with lesser number of pixels
    than `min_size` are ignored'''
    predictions = np.zeros_like(probability)
    for ii in range(predictions.shape[0]):
        predictions[ii, ...] = filter_per_map(probability[ii], threshold, min_size)
    return predictions


def main(args):

    init()
    print(np.random.random(1), random.random() , torch.rand(1))
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '%d'%(args.gpu)

    min_size_dict = {0: 1000/4.0, 1: 1000/4.0, 2: 1000/4.0, 3: 1000/4.0}
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    color_map = {
            0: (255, 150, 0),
            1: (255, 0, 0),
            2: (0, 255, 0),
            3: (0, 0, 255),
            }
    ckpt_path = args.ckpt_path
    cache = 'pred_%s.pkl'%args.phase

    data_folder = args.data_folder
    total_df_path = '%s/train.csv' % data_folder
    train_df = '%s/../split_train.csv' % data_folder
    if args.phase == 'val':
        val_df = '%s/../split_val.csv' % data_folder
    else:
        val_df = '%s/../split_train.csv' % data_folder
    need_split = args.need_split
    downsample = args.downsample
    aspp_dilation = args.aspp_dilation
    replace_stride_with_dilation = [int(_) for _ in args.replace.split(',')]
    freeze = args.freeze
    multigrid = args.multigrid

    test_data_folder = '%s/images' % data_folder

    testloader = provider(
                data_folder=data_folder,
                df_path=total_df_path,
                phases=['train', 'val'],
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                batch_sizes={'train': args.train_batch, 'val': args.val_batch},
                num_workers=2,
                need_split=args.need_split,
                train_df=train_df, 
                val_df=val_df,
                downsample=downsample
                )['val']

    inv_mean = tuple(-ii[0]/ii[1] for ii in zip(mean, std))
    inv_std = tuple(1.0/(ii*255) for ii in std)
    df = pd.read_csv(total_df_path)

    device = torch.device('cuda')

    if args.arch == 'Unet':
        model = Unet('resnet18', encoder_weights=None, classes=4, activation=None)
    elif args.arch == 'deeplabv3_resnet50':
        model = deeplabv3_resnet50(pretrained=False, progress=True, num_classes=4, 
            aux_loss=None, resume_fp=None, aspp_dilation=aspp_dilation,
            replace=replace_stride_with_dilation, freeze=freeze, multigrid=multigrid)
    elif args.arch == 'deeplabv3_se_resnet50':
        model = deeplabv3_se_resnet50(pretrained=False, progress=True, num_classes=4, 
            aux_loss=None, resume_fp=None, aspp_dilation=aspp_dilation,
            replace=replace_stride_with_dilation, freeze=freeze, multigrid=multigrid)  

    model.to(device)
    model.eval()

    state = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(state['state_dict'])

    predictions = []

    res_dice = []
    # if os.path.isfile(cache):
    #     with open(cache, 'rb') as fd:
    #         all_pred = pickle.load(fd)
    # else:
    all_pred = []

    with torch.no_grad():
        for i, batch in enumerate(tqdm(testloader)):
            images, gt_masks = batch
            images_flip = torch.flip(images, [3])
            images_shift = torch.roll(images, -64, 3)
            # if os.path.isfile(cache):
            #     batch_preds = all_pred[i]
            # else:
                #batch_preds_flip = torch.sigmoid(model(images_flip.to(device)))
                #batch_preds_1 = torch.flip(batch_preds_flip, [3])
                #batch_preds_shift = torch.sigmoid(model(images_shift.to(device)))
                #batch_preds_1 = torch.roll(batch_preds_shift, 32, 3)
            outputs = model(images.to(device))
            if isinstance(outputs, dict):
                outputs = outputs['out']

            batch_preds_2 = torch.sigmoid(outputs)

            batch_preds = (batch_preds_2 + batch_preds_2) / 2
            #batch_preds = torch.nn.functional.upsample(batch_preds, scale_factor=2)
            batch_preds = batch_preds.cpu().numpy()

            all_pred.append(batch_preds)

            tmp_dice = metric(batch_preds, gt_masks.numpy(), min_size_dict)
            tmp_dice = np.array(tmp_dice)
            if i == 0:
                res_dice = tmp_dice
            else:
                res_dice = np.concatenate([res_dice, tmp_dice], axis=2)

            cur_dice = np.array(res_dice).mean(2)
            print(cur_dice)
            print('-------')
            print(cur_dice.max(axis=1).mean(), cur_dice.max(axis=1))
            print('=============')
    # if not os.path.isfile(cache):
    #     with open(cache, 'wb') as fd:
    #         pickle.dump(all_pred, fd)


if __name__ == '__main__':
    main(parse_args())
