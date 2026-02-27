## Install Dependencies
```bash
pip install -r requirements.txt
```

## Prepare Dataset
```bash
python prepare.py
```

## Training
```bash
python train.py --gpu_ids 0 --name ft_ResNet50 --train_all --batchsize 32  --data_dir ./Market/pytorch
```

## Test
```bash
python test.py --gpu_ids 0 --name ft_ResNet50 --test_dir ./Market/pytorch  --batchsize 32 --which_epoch last
```

## Evaluation
```bash
python evaluate_gpu.py
```

## Visualization
```bash
python demo.py --query_index 777 --test_dir ./Market/pytorch
```
