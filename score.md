## Unet

|folder| log | lr策略 | batch_size | lr | model | train_score | val_score | submission(test_score) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unet_resnet18_lr1e-2 | 20190924_1939 | Adam | 16 | 5e-4 | best | 0.9213 | 0.9168 | 0.88101(best+30) | 
|  |
| unet_resnet18_Adam_baseline | 20190925_1138 | Adam | 16 | 5e-4 | best(18) | 0.9539 | 0.9405 | 0.88945(best+20-40v1 best) | 
|  |
| unet_resnet18_Adam_baseline_20-40v1  | 20190925_2009 | Adam | 16 | 5e-4 | best(6) | 0.9640 | 0.9413 |   
|   |  |  |  |  | 20 | 0.97117 | 0.9410 |  |
|  |
| unet_resnet18_Adam_baseline_40-60  | 20190926_0952 | Adam | 16 | 5e-4 | best(1) |  | 0.9413 |  | 
|   |  |  |  |  | 20 | 0.97645 | 0.9411 |  |  

## Deeplabv3

|folder| log | lr策略 | batch_size | lr | model | grid(stage4) | aspp_dilation | replace | freeze | train_score | val_score | submission(test_score) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| deeplab | 20190925_2111 | Adam | 8 | 5e-4 | best(20) | 1,2,2 | 6 | 0,0,1 | F | 0.95229 | 0.9409 | 0.88937 | d
|  |
| deeplab_20-40 | 20190926_1942 | Adam | 8 | 5e-4 | best(10) |1,2,2 | 6 | 0,0,1 | F | | 0.9382 | | 
|  |  |  |  |  | 20 |1,2,2 | 6 | 0,0,1 | F | | 0.94335 | 0.89448 | d
|  |
| deeplabv3_dilation2 | 20190926_1943 | Adam | 8 | 5e-4 | best(20) | 1,2,2 | 2 | 0,0,1 | F | | 0.94307 | 0.89299 | d
|  |
| deeplabv3_multigrid124 | 20190926_1948 | Adam | 8 | 5e-4 | best(15) | 1,2,4 | 6 | 0,0,1 | F | | 0.93877 ||
|  |  |  |  |  | 20 |  |  |  |  | | 0.94124 ||
|  |
| deeplabv3_20_freeze | 20190926_2128 | Adam | 4 | 1e-4 | best(8) | 1,2,2 | 6 | 0,1,1 | T || 0.94177 ||
|  |  |  |  |  | 20 | |  |  |  || 0.94436 | 0.88470 |
|  |
| deeplabv3_aspp2_multigrid124 | 20190927_1237 | Adam | 8 | 5e-4 | loss_best(13) | 1,2,4 | 2 | 0,0,1 | F || 0.93956 ||
|  |  |  |  |  | latest | |  |  |  || 0.94174 ||
|  |
| deeplabv3_se | 20190927_1733 | Adam | 8 | 5e-4 | loss_best(17) | 1,2,2 | 6 | 0,0,1 | F || 0.93314 ||
|  |  |  |  |  | latest | |  |  |  || 0.93284 ||
|  |
| deeplabv_wo_dropout | 20190927_1231 | Adam | 8 | 5e-4 | loss_best(20) | 1,2,2 | 6 | 0,0,1 | F || 0.93895 ||
|  |
| deeplabv3_50_se_aspp2 | 20190929_1242 | Adam | 8 | 5e-4 | loss_best(20) | 1,2,2 | 2 | 0,0,1 | F || 0.9437 | 0.88426 |
|  |
| deeplabv3_se_aspp2_debug | 20191009_1102 | Adam | 8 | 5e-4 | latest(20) | 1,2,2 | 2 | 0,0,1 | F || 0.93883 ||
|  |
| deeplabv3_scse_aspp2 | 20191009_1103 | Adam | 8 | 5e-4 | latest(20) | 1,2,2 | 2 | 0,0,1 | F || 0.92821 ||
|  |
| deeplabv3_patch | 20191009_1106 | Adam | 64 | 5e-4 | dice_best | 1,2,2 | 2 | 0,0,1 | F || 0.92744 | 0.88448 |
|  |
| deeplabv3_fivesample_patch_new | 20191010_0149 | Adam | 40 | 5e-4 | dice_best | 1,2,2 | 2 | 0,0,1 | F ||  | 0.88872 | d


