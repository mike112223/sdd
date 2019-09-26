nohup python -u train.py \
    --gpu 2 --epoch 21 --arch deeplabv3_resnet50 \
    --downsample 1 --val_batch 16 --train_batch 8 \
    --work_dir workdir/deeplab_20-40 \
    --resume_from workdir/deeplab/model_20.pth > workdir/20190926_0955.log 2>&1 &


