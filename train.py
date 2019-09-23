import sys
sys.path.insert(0, '.')
import torch
import random
import numpy as np

#import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from lib.model import Unet # import Unet model from the script
from lib.engine import Trainer
from lib.utils import init, plot
from lib.data import provider
from lib.loss import BalanceBCE, DiceLoss


def main():
    
    ### 1. configure
    data_folder = '/data/user/fye/kaggle_steel_competition/input/yichaoxiong'
    #data_folder = '/data/user/cxhuang/datasets/kaggle_steel_competition/aug/'
    #train_df_path = '%s/train_epoch80_minsize250_thres0.5.csv' % data_folder
    #train_df_path = '%s/train.csv' % data_folder
    train_df_path = '%s/train_clean.csv' % data_folder
    phases = ['train', 'val']
    classes = 4
    num_epochs = 20
    model_name = 'resnet18'
    batch_sizes = {"train": 16, "val": 4}
    num_workers = 4
    savedir = 'workdir/resnet18_80epochs/clean'
    resume_fp = 'workdir/resnet18_60epochs/model.pth'
    #lr = 1e-3
    lr = 5e-4
    device = torch.device('cuda:0')

    ### 2. init
    init()
    print(np.random.random(1), random.random() , torch.rand(1)) #, torch.cuda.random(1, 1))

    ### 3. data
    dataloader = provider(
                    data_folder=data_folder,
                    df_path=train_df_path,
                    phases=phases,
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                    batch_sizes=batch_sizes,
                    num_workers=num_workers,
                    )

    ### 4. model
    model = Unet(model_name, encoder_weights='imagenet', classes=classes, activation=None, resume_fp=resume_fp)

    ### 5. criterion
    criterion = torch.nn.BCEWithLogitsLoss()
    #criterion = DiceLoss() #BalanceBCE(5)

    ### 6. trainer
    trainer = Trainer(model, criterion, dataloader, phases, batch_sizes, lr, num_epochs, device)
    trainer.start(savedir)


if __name__ == '__main__':
    main()

