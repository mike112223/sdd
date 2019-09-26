import sys
sys.path.insert(0, '.')
import torch
import random
import numpy as np
import argparse

from lib.model import Unet  # import Unet model from the script
from lib.deeplab import deeplabv3_resnet50
from lib.engine import Trainer
from lib.utils import init, plot
from lib.data import provider
from lib.loss import BalanceBCE, DiceLoss

def parse_args():
    parser = argparse.ArgumentParser(description='Train semantic seg')
    parser.add_argument('--data_folder', default='/DATA5_DB8/data/yanjia/data/steel_defect/train')
    parser.add_argument('--epoch', dest='num_epochs', default=61, type=int)
    parser.add_argument('--work_dir', default=None, help='the dir to save logs and models')
    parser.add_argument('--resume_from', default=None, help='the checkpoint file to resume from')
    parser.add_argument('--arch', choices=['Unet', 'deeplabv3_resnet50'], default='Unet')
    parser.add_argument('--backbone',default='resnet18',help='backbone')
    parser.add_argument('--classes', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--train_batch', type=int, default=16)
    parser.add_argument('--val_batch', type=int, default=4)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--downsample', type=int, default=2)
    parser.add_argument('--restart_epoch', type=int, default=20)
    parser.add_argument('--need_split', action='store_true', default=False)
    parser.add_argument('--save_frequency', type=int, default=20)

    args = parser.parse_args()

    return args


def main(args):
    print(args)
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '%d'%(args.gpu)
    
    ### 1. configure
    data_folder = args.data_folder
    #data_folder = '/data/user/cxhuang/datasets/kaggle_steel_competition/aug/'
    #train_df_path = '%s/train_epoch80_minsize250_thres0.5.csv' % data_folder
    total_df_path = '%s/train.csv' % data_folder
    train_df = '%s/../split_train.csv' % data_folder
    val_df = '%s/../split_val.csv' % data_folder
    need_split = args.need_split
    # train_df_path = '%s/train_clean.csv' % data_folder
    phases = ['train', 'val']
    classes = args.classes
    num_epochs = args.num_epochs
    model_name = args.backbone
    batch_sizes = {"train": args.train_batch, "val": args.val_batch}
    num_workers = args.num_workers
    savedir = args.work_dir
    resume_fp = args.resume_from
    downsample = args.downsample
    lr = args.lr
    restart_epoch = args.restart_epoch
    save_frequency = args.save_frequency

    device = torch.device('cuda:0')

    ### 2. init
    # fix seed
    init()
    print(np.random.random(1), random.random() , torch.rand(1)) #, torch.cuda.random(1, 1))

    ### 3. data
    dataloader = provider(
                    data_folder=data_folder,
                    df_path=total_df_path,
                    phases=phases,
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                    batch_sizes=batch_sizes,
                    num_workers=num_workers,
                    inference=False,
                    need_split=need_split,
                    train_df=train_df, 
                    val_df=val_df,
                    downsample=downsample
                    )

    ### 4. model
    if args.arch == 'Unet':
        model = Unet(model_name, encoder_weights='imagenet', classes=classes, activation=None, resume_fp=resume_fp)
    elif args.arch == 'deeplabv3_resnet50':
        model = deeplabv3_resnet50(pretrained=False, progress=True, num_classes=4, aux_loss=None, resume_fp=resume_fp)

    ### 5. criterion
    criterion = torch.nn.BCEWithLogitsLoss()
    #criterion = DiceLoss() #BalanceBCE(5)

    ### 6. trainer
    trainer = Trainer(model, criterion, dataloader, phases, batch_sizes, lr, num_epochs, device, restart_epoch, save_frequency)
    trainer.start(savedir)


if __name__ == '__main__':
    main(parse_args())

