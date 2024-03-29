import os
import pandas as pd
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, sampler

from albumentations import (HorizontalFlip, VerticalFlip, RandomBrightnessContrast, RandomGamma, ShiftScaleRotate, Normalize, Resize, Compose, GaussNoise)
from albumentations.pytorch import ToTensor

from .utils import make_mask, random_scaling, pad_to_bounding_box, random_crop


def downsample(mask, kernel_size, thresh):
    mask = mask.float()
    new_mask = F.avg_pool2d(mask.unsqueeze(0), kernel_size, kernel_size)
    # new_mask = new_mask.squeeze() > thresh
    new_mask = new_mask.squeeze()
    return new_mask.float()

def mask2contours(mask):

    mask = mask*255

    mask_shape = mask.shape
    width = mask_shape[0]
    height = mask_shape[1]

    contour_mask = np.zeros([width, height, 3], np.uint8)

    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for i in range(0, len(contours)):
        cv2.polylines(contour_mask, contours[i], True, (255,255,255), 2)

    contour_mask = cv2.cvtColor(contour_mask, cv2.COLOR_BGR2GRAY)
    ret, contour_mask = cv2.threshold(contour_mask, 200, 255, cv2.THRESH_BINARY)

    return contour_mask

class SteelDataset(Dataset):
    def __init__(self, df, data_folder, mean, std, phase, inference=False, downsample=1, patch=False, crop_size=[256, 256]):
        self.df = df
        self.root = data_folder
        self.mean = mean
        self.std = std
        self.phase = phase
        self.transforms = get_transforms(phase, mean, std)
        self.fnames = self.df.index.tolist()
        self.inference = inference
        self.downsample = downsample
        self.patch = patch
        self.crop_h = crop_size[0]
        self.crop_w = crop_size[1]

    def __getitem__(self, idx):
        image_id, mask = make_mask(idx, self.df)
        image_path = os.path.join(self.root, 'images',  image_id)
        img = cv2.imread(image_path)
        if self.patch and self.phase == 'train':
            ## vis 
            # print('aaaaaaaaa')       
            # color_dict = {
            #     0: (255, 0, 0),
            #     1: (0, 255, 0),
            #     2: (0, 0, 255),
            #     3: (0, 150, 255),
            # }
            # mask = mask.astype(np.uint8)
            # print(mask.shape)
            # for j in range(4):
            #     _mask = mask2contours(mask[:,:,j])
            #     img[_mask > 0] = color_dict[j]
            # cv2.imshow('ori',img)

            # 1.random scaling
            img, mask = random_scaling(img, mask)
            # 2.padding
            img, mask = pad_to_bounding_box(img, mask, self.crop_h, self.crop_w, 0, 0)
            # 3.random crop
            img, mask = random_crop([img, mask], self.crop_h, self.crop_w)

        augmented = self.transforms(image=img, mask=mask)
        img = augmented['image']
        mask = augmented['mask'] # 256x1600x4
        mask = mask[0].permute(2, 0, 1) # 4x256x1600
        #print('before', mask.dtype, mask.shape)
        if self.inference:
            return image_id, img
        else:
            mask = downsample(mask, self.downsample, 0)
            #print('after', mask.dtype, mask.shape)
            return img, mask

    def __len__(self):
        return len(self.fnames)


def get_transforms(phase, mean, std):
    list_transforms = []
    if phase == "train":
        list_transforms.extend(
            [
                HorizontalFlip(), # only horizontal flip as of now
                #VerticalFlip(),
                #RandomBrightnessContrast(), 
                #RandomGamma(),
                #ShiftScaleRotate(rotate_limit=0),
            ]
        )
    list_transforms.extend(
        [
            Normalize(mean=mean, std=std, p=1),
            ToTensor(),
        ]
    )
    list_trfms = Compose(list_transforms)
    return list_trfms


def _init_fn(worker_id):
    np.random.seed(0)

def pd_transfer(df):
    df.columns.name='ClassId' 
    df.index = df.get('ImageId')
    df = df.drop('ImageId', axis=1)
    for i in range(1,5):
        df.columns.values[i-1] = i
    return df

def provider(
    data_folder,
    df_path=None,
    phases=['train', 'val'],
    mean=None,
    std=None,
    batch_sizes={'train': 8, 'val': 4},
    num_workers=4,
    inference=False,
    need_split=True,
    train_df=None, 
    val_df=None,
    downsample=1,
    patch=False
):
    '''Returns dataloader for the model training'''
    if need_split:
        print('split data')
        df = pd.read_csv(df_path)
        # some preprocessing
        # https://www.kaggle.com/amanooo/defect-detection-starter-u-net
        #df['ImageId'], df['ClassId'] = zip(*df['ImageId_ClassId'].str.split('_'))
        df['ImageId'], df['ClassId'] = df['ImageId_ClassId'].str.slice(0, -2), df['ImageId_ClassId'].str.slice(-1)
        df['ClassId'] = df['ClassId'].astype(int)
        df = df.pivot(index='ImageId',columns='ClassId',values='EncodedPixels')
        df['defects'] = df.count(axis=1)
        train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["defects"])
        val_df.to_csv('/DATA5_DB8/data/yanjia/data/steel_defect/split_val.csv')
        train_df.to_csv('/DATA5_DB8/data/yanjia/data/steel_defect/split_train.csv')
        # train_df = df
        # val_df = df
    else:
        print('read from csv')
        train_df = pd_transfer(pd.read_csv(train_df))
        val_df = pd_transfer(pd.read_csv(val_df))
    print(train_df.columns)

    # train_df = df
    print(train_df.head(), len(train_df))
    print(val_df.head(), len(val_df))
    dataloaders = {}
    for phase in phases:
        df = train_df if phase == "train" else val_df
        image_dataset = SteelDataset(df, data_folder, mean, std, phase, inference, downsample, patch)
        dataloader = DataLoader(
            image_dataset,
            batch_size=batch_sizes[phase],
            num_workers=num_workers,
            pin_memory=True,
            shuffle=True,
            #worker_init_fn=_init_fn
        )
        dataloaders[phase] = dataloader

    return dataloaders

