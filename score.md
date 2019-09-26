## model result

|folder| log | lr策略 | batch_size | lr | model | train_score | val_score | submission(test_score) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unet_resnet18_lr1e-2 | 20190924_1939 | Adam | 16 | 5e-4 | best | 0.9213 | 0.9168 | 0.88101(best+30) | 
|  |
| unet_resnet18_Adam_baseline | 20190925_1138 | Adam | 16 | 5e-4 | best(18) | 0.9539 | 0.9405 |  | 
|  |
| unet_resnet18_Adam_baseline_20-40v1  | 20190925_2009 | Adam | 16 | 5e-4 | best(6) | 0.9640 | 0.9413 |  | 
|   |  |  |  |  | 20 | 97.117 | 0.9410 |  |
|  |
| unet_resnet18_Adam_baseline_40-60  | 20190926_0952 | Adam | 16 | 5e-4 | best |  | 0.9413 |  | 
|   |  |  |  |  | 20 | * | 0.9411 |  |  
|  |
| deeplab | 20190925_2111 | Adam | 8 | 5e-4 | best(20) | * | 0.9409 |  | 
