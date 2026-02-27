快速开始（Market-1501）：以下命令按顺序执行，可完成环境安装、数据准备、训练、测试与评估。扩展训练命令用于快速做对比实验。

## Install Dependencies
安装项目依赖包。
```bash
pip install -r requirements.txt
```

## Prepare Dataset
将 Market-1501 原始目录整理为训练/测试可直接读取的 `pytorch` 结构。
```bash
python prepare.py
```

## Training Baseline (ResNet50)
基线训练命令（ResNet50）。
```bash
python train.py --gpu_ids 0 --name ft_ResNet50 --train_all --batchsize 32  --data_dir ./Market/pytorch
```

## Training + Random Erasing
加入随机擦除增强，常用于提升泛化能力。
```bash
python train.py --gpu_ids 0 --name ft_ResNet50_re --train_all --batchsize 32 --erasing_p 0.5 --data_dir ./Market/pytorch
```

## Training + IBN Backbone
使用 ResNet50-IBN 主干网络。
```bash
python train.py --gpu_ids 0 --name resnet50_ibn --train_all --ibn --batchsize 32 --data_dir ./Market/pytorch
```

## Training + DenseNet121 Backbone
使用 DenseNet121 主干网络。
```bash
python train.py --gpu_ids 0 --name densenet121_market --use_dense --train_all --batchsize 32 --data_dir ./Market/pytorch
```

## Training + Swin Backbone
使用 Swin Transformer 主干网络（显存需求更高）。
```bash
python train.py --gpu_ids 0 --name swin_market --use_swin --train_all --batchsize 32 --data_dir ./Market/pytorch
```

## Training + Circle Loss
在分类损失基础上加入 Circle Loss。
```bash
python train.py --gpu_ids 0 --name circle_market --train_all --circle --warm_epoch 5 --batchsize 32 --data_dir ./Market/pytorch
```

## Training + Triplet Loss
在分类损失基础上加入 Triplet Loss。
```bash
python train.py --gpu_ids 0 --name triplet_market --train_all --triplet --batchsize 32 --data_dir ./Market/pytorch
```

## Test
提取 query/gallery 特征，生成评估所需结果文件。
```bash
python test.py --gpu_ids 0 --name ft_ResNet50 --test_dir ./Market/pytorch  --batchsize 32 --which_epoch last
```

## Evaluation
计算 Rank@K 与 mAP 指标。
```bash
python evaluate_gpu.py
```

## Visualization
可视化单个 query 的检索结果。
```bash
python demo.py --query_index 777 --test_dir ./Market/pytorch
```
