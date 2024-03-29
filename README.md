# Base info
## 1、比赛链接
Steel Defect Detection
https://www.kaggle.com/c/severstal-steel-defect-detection/overview

## 2、训练baseline
https://www.kaggle.com/rishabhiitbhu/unet-starter-kernel-pytorch-lb-0-88

## 3、前向baseline
https://www.kaggle.com/rishabhiitbhu/unet-pytorch-inference-kernel

## 4、可能会用到的trick
MixMatch https://www.kaggle.com/c/freesound-audio-tagging-2019/discussion/91827

Duplicates & Pattern
https://www.kaggle.com/c/severstal-steel-defect-detection/discussion/107053#latest-618971

model ensemble

test time augmentation（horizontal flip）

## 5、可供参考的discussion
Single Model Segmentation Score
https://www.kaggle.com/c/severstal-steel-defect-detection/discussion/109174#latest-628801

# train instruction template

## deeplabv3
'''bash
nohup python -u train.py \
    --gpu 2 --epoch 21 --arch deeplabv3_resnet50 \
    --downsample 1 --val_batch 32 --train_batch 8 \
    --work_dir /DATA5_DB8/data/yanjia/results/deeplab_20-40 \
    --resume_from /DATA5_DB8/data/yanjia/results/deeplab/model_20.pth > workdir/20190926_1938.log 2>&1 &
'''

## Unet
nohup python -u train.py \
    --gpu 1 --epoch 21 --val_batch 32 \
    --work_dir workdir/Unet_resnet18_Adam_baseline_40-60epoch \
    --resume_from workdir/Unet_resnet18_Adam_baseline_20-40epoch_v1/model_20.pth > workdir/20190926_0952.log 2>&1 &

# test intruction template
## deeplabv3
python valid.py \
    --gpu 3 --val_batch 32 --downsample 1 \
    --arch deeplabv3_resnet50 --ckpt_path /DATA5_DB8/data/yanjia/results/deeplabv3_dilation2/model_best.pth 

## Unet
python valid.py \
    --val_batch 32 --gpu 0 \
    --ckpt_path workdir/Unet_resnet18_Adam_baseline_20-40epoch_v1/model_best.pth

