# 可配置选项

## 1. Backbone（模型主干）

| 显示名称 | 标识 | 训练参数 |
| --- | --- | --- |
| ResNet50（默认） | `resnet50` | 无额外参数 |
| ResNet50-IBN | `resnet50_ibn` | `--ibn` |
| DenseNet121 | `densenet` | `--use_dense` |
| Swin | `swin` | `--use_swin` |
| SwinV2 | `swinv2` | `--use_swinv2` |
| DINOv3 | `dino` | `--use_dino` |
| EfficientNet-B4 | `efficientnet_b4` | `--use_efficient` |
| NAS | `nas` | `--use_NAS` |
| HRNet | `hrnet` | `--use_hr` |
| ConvNeXt | `convnext` | `--use_convnext` |
| PCB (ResNet50+PCB) | `pcb` | `--PCB` |
| ResNet50-USAM | `resnet50_usam` | `--usam` |

## 2. Loss（损失函数）

| 显示名称 | 标识 | 训练参数 |
| --- | --- | --- |
| CrossEntropy（默认） | `ce` | 无额外参数 |
| Circle | `circle` | `--circle --warm_epoch 5` |
| Triplet | `triplet` | `--triplet` |
| ArcFace | `arcface` | `--arcface` |
| CosFace | `cosface` | `--cosface` |
| Contrast | `contrast` | `--contrast` |
| Instance | `instance` | `--instance` |
| Instance-ID | `instance_id` | `--instance_id` |
| Lifted | `lifted` | `--lifted` |
| Sphere | `sphere` | `--sphere` |

## 3. Dataset（数据集）

| 显示名称 | 标识 | 原始路径 | 预处理脚本 |
| --- | --- | --- | --- |
| Market-1501（默认） | `market` | `./data/Market` | `prepare.py` |
| DukeMTMC-reID | `duke` | `./data/Duke` | `prepare_Duke.py` |
| MSMT17 | `msmt` | `./data/MSMT` | `prepare_MSMT.py` |
| CUB-200-2011 | `cub` | `./data/CUB` | `prepare_CUB.py` |
| VehicleID | `vehicleid` | `./data/VehicleID` | `prepare_VehicleID.py` |
| VeRi | `veri` | `./data/VeRi` | `prepare_VeRi.py` |
| VIPeR | `viper` | `./data/VIPeR` | `prepare_viper.py` |
