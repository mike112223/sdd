import matplotlib.pyplot as plt
import random
import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import os
import torch

### utils starts
def rle2mask(rle, imgshape):
    width = imgshape[0]
    height= imgshape[1]

    mask= np.zeros( width*height ).astype(np.uint8)

    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        mask[int(start):int(start+lengths[index])] = 1
        current_position += lengths[index]

    return np.flipud(np.rot90( mask.reshape(height,width), k=1))

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

def downsample(mask, kernel_size, thresh):
    new_mask = torch.nn.functional.avg_pool2d(torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0),
                                              (kernel_size, kernel_size), kernel_size)
    new_mask = (new_mask.squeeze() > thresh).float()
    return new_mask

### utils ends

def main():
    train_annot_fp = '/data/steel_detection/train/train.csv'
    color_dict = {
        '1': (255, 0, 0),
        '2': (0, 255, 0),
        '3': (0, 0, 255),
        '4': (255, 150, 0),
    }

    train = pd.read_csv(train_annot_fp)
    train = train[train['EncodedPixels'].notnull()]
    print(train.shape)
    train.head()
    train.tail()

    fig=plt.figure(figsize=(20,100))

    for i in range(0, len(train['ImageId_ClassId'])):
        fn = train['ImageId_ClassId'].iloc[i].split('_')[0]

        img_name, label = train['ImageId_ClassId'].iloc[i].split('_')

        img = cv2.imread('/data/steel_detection/train/images/'+fn )

        fig.add_subplot(2, 1, 1)
        plt.imshow(img)

        mask = rle2mask( train['EncodedPixels'].iloc[i], img.shape  )
        mask = mask2contours(mask)

        #img[mask==1,0] = 255
        #img[mask > 0] = (255, 0, 0)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        '''
        kernel_size = 32
        thresh = 0.2
        for ii in range(0, 0):
            new_mask = downsample(mask, kernel_size, thresh)
            kernel_size = int(kernel_size / 2)
            fig.add_subplot(6, 1, ii+1)
            plt.imshow(new_mask.numpy().astype(np.uint8))
        #plt.figure(2)
        '''
        fig.add_subplot(2, 1, 2)
        img[mask > 0] = color_dict[label]
        plt.imshow(img)
        plt.waitforbuttonpress()

if __name__ == '__main__':
    main()
