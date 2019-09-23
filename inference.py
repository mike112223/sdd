import sys
sys.path.insert(0, '.')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pdb
import os
import cv2
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset
from albumentations import (Normalize, Compose)
from albumentations.torch import ToTensor
import torch.utils.data as data

from lib.model import Unet # import Unet model from the script


def mask2rle(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels= img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    print('debug', pixels.shape)
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    import pdb
    pdb.set_trace()
    print('debug', runs.shape)
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


class TestDataset(Dataset):
    '''Dataset for test prediction'''
    def __init__(self, root, df, mean, std):
        self.root = root
        df['ImageId'] = df['ImageId_ClassId'].apply(lambda x: x.split('_')[0])
        self.fnames = df['ImageId'].unique().tolist()
        self.num_samples = len(self.fnames)
        self.transform = Compose(
            [
                Normalize(mean=mean, std=std, p=1),
                ToTensor(),
            ]
        )

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        path = os.path.join(self.root, fname)
        image = cv2.imread(path)
        images = self.transform(image=image)["image"]
        return fname, images

    def __len__(self):
        return self.num_samples


def post_process(probability, threshold, min_size):
    '''Post processing of each predicted mask, components with lesser number of pixels
    than `min_size` are ignored'''
    mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]
    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
    predictions = np.zeros((256, 1600), np.float32)
    num = 0
    for c in range(1, num_component):
        p = (component == c)
        if p.sum() > min_size:
            predictions[p] = 1
            num += 1
    return predictions, num


data_folder = '/data/user/fye/kaggle_steel_competition/input'
sample_submission_path = '%s/sample_submission.csv' % data_folder
test_data_folder = '%s/test_images' % data_folder


# initialize test dataloader
best_threshold = 0.5
num_workers = 2
batch_size = 4
print('best_threshold', best_threshold)
min_size = 3500
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
inv_mean = tuple(-ii[0]/ii[1] for ii in zip(mean, std))
inv_std = tuple(1.0/(ii*255) for ii in std)
df = pd.read_csv(sample_submission_path)
testset = DataLoader(
    TestDataset(test_data_folder, df, mean, std),
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True
)
# Initialize mode and load trained weights
ckpt_path = 'workdir/resnet18/model.pth'
device = torch.device('cuda')
model = Unet('resnet18', encoder_weights=None, classes=4, activation=None)
model.to(device)
model.eval()
state = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
model.load_state_dict(state['state_dict'])

# start prediction
predictions = []
color_map = {
        0: (255, 150, 0),
        1: (255, 0, 0),
        2: (0, 255, 0),
        3: (0, 0, 255),
        }
for i, batch in enumerate(tqdm(testset)):
    fnames, images = batch
    batch_preds = torch.sigmoid(model(images.to(device)))
    batch_preds = batch_preds.detach().cpu().numpy()
    img_idx = 0
    for fname, preds in zip(fnames, batch_preds):
        for cls, pred in enumerate(preds):
            pred, num = post_process(pred, best_threshold, min_size)
            inv_norm = Normalize(mean=inv_mean, std=inv_std, p=1, max_pixel_value=1.0)
            image = images[img_idx].permute(1, 2, 0).numpy()
            image_orig = inv_norm(image=image)['image'].astype(np.uint8)
            #import pdb
            #pdb.set_trace()
            #print(pred.max())
            mask = pred.astype(np.uint8)
            if pred.sum() > 3000:
                print(pred.sum())
                fig=plt.figure()
                fig.add_subplot(2, 1, 1)
                plt.imshow(image_orig)
                fig.add_subplot(2, 1, 2)
                image_orig[mask==1] = color_map[cls]
                plt.imshow(image_orig)
                #cv2.imwrite('test.jpg', image_orig)
                plt.savefig('test.jpg')
            rle = mask2rle(pred)
            name = fname + f'_{cls+1}'
            predictions.append([name, rle])
        img_idx += 1

# save predictions to submission.csv
df = pd.DataFrame(predictions, columns=['ImageId_ClassId', 'EncodedPixels'])
df.to_csv('submission.csv', index=False)
