import sys
sys.path.insert(0, '.')
import torch
import random
import numpy as np
import argparse

from lib.model import Unet  # import Unet model from the script
from lib.engine import Trainer
from lib.utils import init, plot
from lib.data import provider
from lib.loss import BalanceBCE, DiceLoss
from lib import deeplab

def parse_args():
    parser = argparse.ArgumentParser(description='Train semantic seg')
    parser.add_argument('--data_folder', default='/DATA5_DB8/data/yanjia/data/steel_defect/train')
    parser.add_argument('--epoch', dest='num_epochs', default=61, type=int)
    parser.add_argument('--work_dir', default='tmp', help='the dir to save logs and models')
    parser.add_argument('--resume_from', default=None, help='the checkpoint file to resume from')
    parser.add_argument('--arch', choices=['Unet', 'deeplabv3_resnet50', 'deeplabv3_se_resnet50', 'deeplabv3_resnet101'], default='Unet')
    parser.add_argument('--backbone',default='resnet18',help='backbone')
    parser.add_argument('--classes', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--train_batch', type=int, default=16)
    parser.add_argument('--val_batch', type=int, default=4)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--downsample', type=int, default=2)
    parser.add_argument('--restart_epoch', type=int, default=30)
    parser.add_argument('--need_split', action='store_true', default=False)
    parser.add_argument('--save_frequency', type=int, default=20)
    parser.add_argument('--aspp_dilation', type=int, default=6)
    parser.add_argument('--replace',help='replace_stride_with_dilation', type=str, default='0,0,1')
    parser.add_argument('--freeze', action='store_true', default=False, help='whether freeze bn')
    parser.add_argument('--multigrid', action='store_true', default=False)
    parser.add_argument('--aux_loss', action='store_true', default=False)
    parser.add_argument('--patch', action='store_true', default=False)
    parser.add_argument('--patience', type=int, default=3)

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
    arch = args.arch
    downsample = args.downsample
    lr = args.lr
    restart_epoch = args.restart_epoch
    save_frequency = args.save_frequency
    aspp_dilation = args.aspp_dilation
    replace_stride_with_dilation = [int(_) for _ in args.replace.split(',')]
    freeze = args.freeze
    multigrid = args.multigrid
    aux_loss = args.aux_loss
    patch = args.patch
    patience = args.patience

    if not os.path.exists(savedir):
        os.makedirs(savedir)

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
                    downsample=downsample,
                    patch=patch
                    )

    ### 4. model
    if arch == 'Unet':
        model = Unet(model_name, encoder_weights='imagenet', classes=classes, activation=None, resume_fp=resume_fp)
    else:
        model = deeplab.__dict__[arch](pretrained=False, progress=True, num_classes=4, 
            aux_loss=aux_loss, resume_fp=resume_fp, aspp_dilation=aspp_dilation,
            replace=replace_stride_with_dilation, freeze=freeze, multigrid=multigrid)

    ### 5. criterion
    criterion = torch.nn.BCEWithLogitsLoss()
    #criterion = DiceLoss() #BalanceBCE(5)

    ### 6. trainer
    trainer = Trainer(model, criterion, dataloader, phases, batch_sizes, 
                    lr, num_epochs, device, restart_epoch, save_frequency, patience)
    trainer.start(savedir)


if __name__ == '__main__':
    main(parse_args())

