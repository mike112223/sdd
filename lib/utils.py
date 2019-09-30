import numpy as np
import time
import torch
import warnings
import random
import os
import cv2


#https://www.kaggle.com/paulorzp/rle-functions-run-lenght-encode-decode
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


def make_mask(row_id, df):
    '''Given a row index, return image_id and mask (256, 1600, 4)'''
    fname = df.iloc[row_id].name
    labels = df.iloc[row_id][:4]
    masks = np.zeros((256, 1600, 4), dtype=np.float32) # float32 is V.Imp
    # 4:class 1～4 (ch:0～3)

    for idx, label in enumerate(labels.values):
        if label is not np.nan:
            label = label.split(' ')
            positions = map(int, label[0::2])
            length = map(int, label[1::2])
            mask = np.zeros(256 * 1600, dtype=np.uint8)
            for pos, le in zip(positions, length):
                pos -= 1
                mask[pos:(pos + le)] = 1
            masks[:, :, idx] = mask.reshape(256, 1600, order='F')
    return fname, masks


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


def class_metric(probability, truth, min_size, thres_range=np.arange(0.3, 0.7, 0.05)):
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

        # dice = np.nan_to_num(dice, nan=1)
        dice[np.isnan(dice)] = 1

        dice_list.append(dice)

    return dice_list


def metric(probability, truth, min_size_dict, thres_range=np.arange(0.1, 0.7, 0.05)):
    '''Calculates dice of positive and negative images seperately'''
    '''probability and truth must be torch tensors'''
    nclasses = truth.shape[1]
    all_dice = []
    for ii in range(nclasses):
        #print('truth', truth[:, ii, :, :].astype(np.float).sum())
        class_dice = class_metric(probability[:, ii, :, :], truth[:, ii, :, :], min_size_dict[ii], thres_range)
        all_dice.append(class_dice)

    cur_dice = np.array(all_dice).mean(2)
    mean_dice, max_dice = cur_dice.max(axis=1).mean(), cur_dice.max(axis=1)

    return mean_dice, max_dice 


class Meter:
    '''A meter to keep track of iou and dice scores throughout an epoch'''
    def __init__(self, phase, min_size_dict={0: 0, 1: 0, 2: 0, 3: 0}):
        self.phase = phase
        self.min_size_dict = min_size_dict 
        # self.thres_range = np.arange(0.1, 0.7, 0.05)
        self.thres_range = np.arange(0.5, 0.51, 0.05)

        self.mean_dices = []
        self.max_dices = []
        self.losses = []

    # def train_update(self, loss, progress):
    #     loss = loss.item()
    #     self.losses.append(loss)
    #     stamp = time.strftime('%m/%d-%H:%M:%S')
    #     print('%s: %s, loss %.4f' % (stamp, progress, loss))

    def update(self, targets, outputs, loss, progress):
        loss = loss.item()
        self.losses.append(loss)
        stamp = time.strftime('%m/%d-%H:%M:%S')

        if self.phase == 'train':
            print('%s: %s, loss %.4f' % (stamp, progress, loss))
        else:
            probs = torch.sigmoid(outputs)
            mean_dice, max_dice = metric(probs.detach().cpu().numpy(), targets.cpu().numpy(), self.min_size_dict, self.thres_range)

            self.mean_dices.append(mean_dice)
            self.max_dices.append(max_dice)
        
            max_dice_str = ''
            for ii in max_dice:
                max_dice_str += ' %.4f' % ii
            print('%s: %s, loss %.4f,  mean dice %.4f, max dice %s' % (stamp, progress, loss, mean_dice, max_dice_str))

    def summary(self):
        loss = np.array(self.losses).mean()

        if self.phase == 'train':
            print('summary: loss %.4f' % (loss)) 
            return loss
        else:
            mean_dice = np.array(self.mean_dices).mean()
            max_dice = np.array(self.max_dices).mean(0)

            max_dice_str = ''
            for ii in max_dice:
                max_dice_str += '%.4f ' % ii
            max_dice_str = max_dice_str.strip()
            print('summary: loss %.4f,  mean dice %.4f, max dice %s' % (loss, mean_dice, max_dice_str))
            return loss, mean_dice


def init(seed=69):
    warnings.filterwarnings("ignore")
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True #False
    #torch.backends.cudnn.deterministic = True


def plot(scores, name):
    plt.figure(figsize=(15,5))
    plt.plot(range(len(scores["train"])), scores["train"], label=f'train {name}')
    plt.plot(range(len(scores["train"])), scores["val"], label=f'val {name}')
    plt.title(f'{name} plot'); plt.xlabel('Epoch'); plt.ylabel(f'{name}');
    plt.legend(); 
    plt.show()


def random_scaling(img, mask, min_scale_factor=0.5, max_scale_factor=2.0):
    scale = np.random.uniform(min_scale_factor, max_scale_factor)
    # print('scale:',scale)
    h, w, _ = img.shape
    scale_w, scale_h = int(scale*w), int(scale*h)
    # print(scale_h, scale_w)
    img = cv2.resize(img, (scale_w, scale_h), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (scale_w, scale_h), interpolation=cv2.INTER_NEAREST)

    return img, mask

def pad_to_bounding_box(img, mask, crop_size, offset_height, offset_width):
    pad_value = [123.675, 116.28 ,103.53]
    ignore_value = -255
    h, w, _ = img.shape
    target_height = max(crop_size - h, 0)
    target_width = max(crop_size - w, 0)
    # print('padding:', target_height, target_width)

    img = np.pad(img, ((offset_height, target_height),(offset_width,target_width),(0,0)),
                'constant', constant_values=np.repeat(pad_value,2).reshape(3,2))

    mask = np.pad(mask, ((offset_height, target_height),(offset_width,target_width),(0,0)),
                'constant', constant_values=np.repeat(ignore_value,6).reshape(3,2))
    return img, mask

# def random_crop(image_list, crop_size):
#     image_height, image_width, _ = image_list[0].shape 

#     max_offset_height = image_height - crop_size + 1
#     max_offset_width = image_width - crop_size + 1
#     offset_height = np.random.randint(0, max_offset_height)
#     offset_width = np.random.randint(0, max_offset_width)

#     print('offset:', offset_height, offset_width)

#     return [image[offset_height:offset_height+crop_size,offset_width:offset_width+crop_size,:] for image in image_list]

def random_crop(image_list, crop_size, iof_thresh=0.3, area_thresh=100):

    img = image_list[0]
    mask = image_list[1]
    image_height, image_width, _ = img.shape

    max_offset_height = image_height - crop_size + 1
    max_offset_width = image_width - crop_size + 1

    fg_pixels = np.where(mask == 1)
    while True:
        if len(fg_pixels[0]):
            idx = np.random.randint(0, len(fg_pixels[0]))
            y, x, label = fg_pixels[0][idx], fg_pixels[1][idx], fg_pixels[2][idx]

            y = max(0, y-crop_size//2)
            x = max(0, x-crop_size//2)

            offset_height = y if y < max_offset_height else np.random.randint(0, max_offset_height)
            offset_width = x if x < max_offset_width else np.random.randint(0, max_offset_width)

            crop_mask = mask[offset_height:offset_height+crop_size,offset_width:offset_width+crop_size,label]
            area = len(np.where(crop_mask==1)[0])
            iof = area/len(np.where(mask[:,:,label]==1)[0])
            # print(area, iof)
            if iof > iof_thresh or area > area_thresh:
                break

        else:
            offset_height = np.random.randint(0, max_offset_height)
            offset_width = np.random.randint(0, max_offset_width)
            break

    # print('offset:', offset_height, offset_width)

    return [image[offset_height:offset_height+crop_size,offset_width:offset_width+crop_size,:] for image in image_list]
