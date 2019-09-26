## model result

|folder| model | log | lr策略 | batch_size | lr | train_score | val_score | submission(test_score) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unet_resnet18_lr1e-2 | best | 20190924_1939 | Adam | 16 | 5e-4 | 0.9213 | 0.9168 | 0.88101(best+30) | 
| unet_resnet18_Adam_baseline | best | 20190925_1138 | Adam | 16 | 5e-4 | 0.9539 | 0.9405 |  | 
| unet_resnet18_Adam_baseline_20-40v1  | best | 20190925_2009 | Adam | 16 | 5e-4 | 96.40 | 0.9413 |  | 
| deeplab | best | 20190925_2111 | Adam | 8 | 5e-4 |  |  |  | 
