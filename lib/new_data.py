import os
import pandas as pd
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Sampler

from albumentations import (HorizontalFlip, VerticalFlip, RandomBrightnessContrast, RandomGamma, ShiftScaleRotate, Normalize, Resize, Compose, GaussNoise)
from albumentations.pytorch import ToTensor

def run_length_decode(rle, height=256, width=1600, fill_value=1):
    mask = np.zeros((height,width), np.float32)
    if rle != '':
        mask=mask.reshape(-1)
        r = [int(r) for r in rle.split(' ')]
        r = np.array(r).reshape(-1, 2)
        for start,length in r:
            start = start-1  #???? 0 or 1 index ???
            mask[start:(start + length)] = fill_value
        mask=mask.reshape(width, height).T
    return mask

def random_scaling(img, mask, min_scale_factor=0.5, max_scale_factor=2.0):
    scale = np.random.uniform(min_scale_factor, max_scale_factor)
    # print('scale:',scale)
    h, w, _ = img.shape
    scale_w, scale_h = int(scale*w), int(scale*h)
    # print(scale_h, scale_w)
    img = cv2.resize(img, (scale_w, scale_h), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (scale_w, scale_h), interpolation=cv2.INTER_NEAREST)

    return img, mask

def pad_to_bounding_box(img, mask, crop_h=256, crop_w=400, offset_height=0, offset_width=0):

    pad_value = [123.675, 116.28 ,103.53]
    ignore_value = -255
    h, w, _ = img.shape
    target_height = max(crop_h - h, 0)
    target_width = max(crop_w - w, 0)
    # print('padding:', target_height, target_width)

    img = np.pad(img, ((offset_height, target_height),(offset_width, target_width),(0,0)),
                'constant', constant_values=np.repeat(pad_value,2).reshape(3,2))

    mask = np.pad(mask, ((offset_height, target_height),(offset_width, target_width)),
            'constant', constant_values=np.repeat(ignore_value,4).reshape(2,2))        

    return img, mask

# def random_crop(image_list, crop_size):
#     image_height, image_width, _ = image_list[0].shape 

#     max_offset_height = image_height - crop_size + 1
#     max_offset_width = image_width - crop_size + 1
#     offset_height = np.random.randint(0, max_offset_height)
#     offset_width = np.random.randint(0, max_offset_width)

#     print('offset:', offset_height, offset_width)

#     return [image[offset_height:offset_height+crop_size,offset_width:offset_width+crop_size,:] for image in image_list]

def random_crop(img, mask, crop_h=256, crop_w=400, iof_thresh=0.3, area_thresh=100):

    image_height, image_width, _ = img.shape

    max_offset_height = image_height - crop_h + 1
    max_offset_width = image_width - crop_w + 1

    fg_pixels = np.where(mask == 1)
    while True:
        if len(fg_pixels[0]):
            idx = np.random.randint(0, len(fg_pixels[0]))
            y, x = fg_pixels[0][idx], fg_pixels[1][idx]

            y = max(0, y-crop_h//2)
            x = max(0, x-crop_w//2)

            offset_height = y if y < max_offset_height else np.random.randint(0, max_offset_height)
            offset_width = x if x < max_offset_width else np.random.randint(0, max_offset_width)

            crop_mask = mask[offset_height:offset_height+crop_h,offset_width:offset_width+crop_w]
            area = len(np.where(crop_mask==1)[0])
            iof = area/len(np.where(mask[:,:]==1)[0])
            # print(area, iof)
            if iof > iof_thresh or area > area_thresh:
                break

        else:
            offset_height = np.random.randint(0, max_offset_height)
            offset_width = np.random.randint(0, max_offset_width)
            break

    # print('offset:', offset_height, offset_width)

    return img[offset_height:offset_height+crop_h,offset_width:offset_width+crop_w,:], mask[offset_height:offset_height+crop_h,offset_width:offset_width+crop_w]


def make_mask(rle):
    '''Given a row index, return image_id and mask (256, 1600, 4)'''
    masks = np.zeros((256, 1600, 4), dtype=np.float32) # float32 is V.Imp
    # 4:class 1～4 (ch:0～3)

    for idx, label in enumerate(rle):
        if label != '':
            label = label.split(' ')
            positions = map(int, label[0::2])
            length = map(int, label[1::2])
            mask = np.zeros(256 * 1600, dtype=np.uint8)
            for pos, le in zip(positions, length):
                pos -= 1
                mask[pos:(pos + le)] = 1
            masks[:, :, idx] = mask.reshape(256, 1600, order='F')
    return masks

def df_loc_by_list(df, column_name, sort_list):
    df = df.loc[df[column_name].isin(sort_list)]
    df.index = df[column_name]
    df = df.reindex(sort_list)
    df.index.name = 'tmp'
    df = df.reset_index().drop('tmp', axis=1)
    return df

def downsample(mask, kernel_size, thresh):
    mask = mask.float()
    new_mask = F.avg_pool2d(mask.unsqueeze(0), kernel_size, kernel_size)
    # new_mask = new_mask.squeeze() > thresh
    new_mask = new_mask.squeeze()
    return new_mask.float()

class NewSteelDataset(Dataset):
    def __init__(self, split, csv, data_folder, mean, std, phase, 
                inference=False, downsample=1, patch=False, crop_size=[256, 400]):

        self.split   = split
        self.csv     = csv
        self.root = data_folder
        self.mean = mean
        self.std = std
        self.phase = phase
        self.transforms = get_transforms(phase, mean, std)
        self.inference = inference
        self.downsample = downsample
        self.patch = patch
        self.crop_h = crop_size[0]
        self.crop_w = crop_size[1]

        # self.uid = list(np.concatenate([np.load(DATA_DIR + '/split/%s'%f , allow_pickle=True) for f in split]))
        self.uid = list(np.concatenate([pd.read_csv(f)['ImageId'].values for f in split]))
        df = pd.concat([pd.read_csv(f).fillna('') for f in csv])

        df['Class'] = df['ImageId_ClassId'].str[-1].astype(np.int32)
        # df['Label']= (np.array([isinstance(_, str)  for _ in df['EncodedPixels']])).astype(np.int32)  
        df['Label'] = (df['EncodedPixels']!='').astype(np.int32)
        df = df_loc_by_list(df, 'ImageId_ClassId', [ u + '_%d'%c  for u in self.uid for c in [1,2,3,4] ])
        self.df = df
        self.num_image = len(df)//4


    def __str__(self):
        num1 = (self.df['Class']==1).sum()
        num2 = (self.df['Class']==2).sum()
        num3 = (self.df['Class']==3).sum()
        num4 = (self.df['Class']==4).sum()
        pos1 = ((self.df['Class']==1) & (self.df['Label']==1)).sum()
        pos2 = ((self.df['Class']==2) & (self.df['Label']==1)).sum()
        pos3 = ((self.df['Class']==3) & (self.df['Label']==1)).sum()
        pos4 = ((self.df['Class']==4) & (self.df['Label']==1)).sum()
        neg1 = num1-pos1
        neg2 = num2-pos2
        neg3 = num3-pos3
        neg4 = num4-pos4

        length = len(self)
        num = len(self)
        pos = (self.df['Label']==1).sum()
        neg = num-pos

        #---

        string  = ''
        string += '\tmode    = %s\n'%self.phase
        string += '\tsplit   = %s\n'%self.split
        string += '\tcsv     = %s\n'%str(self.csv)
        string += '\tnum_image = %8d\n'%self.num_image
        string += '\tlen       = %8d\n'%len(self)
        if self.phase == 'train':
            string += '\t\tpos1, neg1 = %5d  %0.3f,  %5d  %0.3f\n'%(pos1,pos1/num,neg1,neg1/num)
            string += '\t\tpos2, neg2 = %5d  %0.3f,  %5d  %0.3f\n'%(pos2,pos2/num,neg2,neg2/num)
            string += '\t\tpos3, neg3 = %5d  %0.3f,  %5d  %0.3f\n'%(pos3,pos3/num,neg3,neg3/num)
            string += '\t\tpos4, neg4 = %5d  %0.3f,  %5d  %0.3f\n'%(pos4,pos4/num,neg4,neg4/num)
        return string


    def __len__(self):
        return len(self.uid)


    def __getitem__(self, index):
        # print(index)
        image_id = self.uid[index]

        rle = [
            self.df.loc[self.df['ImageId_ClassId']==image_id + '_1','EncodedPixels'].values[0],
            self.df.loc[self.df['ImageId_ClassId']==image_id + '_2','EncodedPixels'].values[0],
            self.df.loc[self.df['ImageId_ClassId']==image_id + '_3','EncodedPixels'].values[0],
            self.df.loc[self.df['ImageId_ClassId']==image_id + '_4','EncodedPixels'].values[0],
        ]
        image_path = os.path.join(self.root, 'images',  image_id)
        image = cv2.imread(image_path)

        label = [ 0 if r=='' else 1 for r in rle]
        mask  = np.array([run_length_decode(r, height=256, width=1600, fill_value=c) for c,r in zip([1,2,3,4],rle)])
        mask  = mask.max(0, keepdims=0)

        infor = {
            'index':index,
            'folder':self.root,
            'image_id':image_id,
        }

        if self.patch and self.phase == 'train':
            image, mask = patch_transforms(image, mask, self.crop_h, self.crop_w)

        augmented = self.transforms(image=image, mask=mask)
        image = augmented['image']
        mask = augmented['mask'] # 256x1600x4
        # mask = mask[0].permute(2, 0, 1) # 4x256x1600
        #print('before', mask.dtype, mask.shape)
        if self.inference:
            return image, mask, label, infor
        else:
            mask = downsample(mask, self.downsample, 0)
            return image, mask

class FiveBalanceClassSampler(Sampler):

    def __init__(self, dataset):
        self.dataset = dataset

        label = (self.dataset.df['Label'].values)
        label = label.reshape(-1,4)
        label = np.hstack([label.sum(1,keepdims=True)==0,label]).T

        self.neg_index  = np.where(label[0])[0]
        self.pos1_index = np.where(label[1])[0]
        self.pos2_index = np.where(label[2])[0]
        self.pos3_index = np.where(label[3])[0]
        self.pos4_index = np.where(label[4])[0]

        #5x
        self.num_image = len(self.dataset.df)//4
        self.length = self.num_image*5


    def __iter__(self):
        # neg = self.neg_index.copy()
        # random.shuffle(neg)

        neg  = np.random.choice(self.neg_index,  self.num_image, replace=True)
        pos1 = np.random.choice(self.pos1_index, self.num_image, replace=True)
        pos2 = np.random.choice(self.pos2_index, self.num_image, replace=True)
        pos3 = np.random.choice(self.pos3_index, self.num_image, replace=True)
        pos4 = np.random.choice(self.pos4_index, self.num_image, replace=True)

        l = np.stack([neg,pos1,pos2,pos3,pos4]).T
        l = l.reshape(-1)
        return iter(l)

    def __len__(self):
        return self.length

def patch_transforms(image, mask, crop_h, crop_w):

    # 1.random scaling
    image, mask = random_scaling(image, mask)
    # print(image.shape)
    # 2.padding
    image, mask = pad_to_bounding_box(image, mask, crop_h, crop_w, 0, 0)
    # print(image.shape)
    # 3.random crop
    image, mask = random_crop(image, mask, crop_h, crop_w, 0, 0)

    return image, mask

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

    dataloaders = {}
    if not isinstance(df_path, list):
        df_path = [df_path]
    if not isinstance(train_df, list):
        train_df = [train_df]
    if not isinstance(val_df, list):
        val_df = [val_df]

    for phase in phases:
        df = train_df if phase == "train" else val_df
        image_dataset = NewSteelDataset(df, df_path, data_folder, mean, std, phase, inference, downsample, patch)
        dataloader = DataLoader(
            image_dataset,
            sampler=FiveBalanceClassSampler(image_dataset),
            batch_size=batch_sizes[phase],
            drop_last=True,
            num_workers=num_workers,
            pin_memory=True,
            #worker_init_fn=_init_fn
        )
        dataloaders[phase] = dataloader

    return dataloaders
