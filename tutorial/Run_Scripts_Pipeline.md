# Run Scripts Pipeline

This document explains the `train.sh` and `evaluate.sh` execution flow.

## 1) Start

```bash
git clone https://github.com/wilson-lyc/Person_reID
cd Person_reID
bash train.sh
```

## 2) Interactive Choices

`train.sh` asks user to select:
- Backbone (`ResNet50`, `ResNet50-IBN`, `DenseNet121`, `Swin`)
- Dataset (`Market (可自动下载)`, `Duke (可自动下载)`, `MSMT17`, `CUB`, `VehicleID`, `VeRi`, `VIPeR`)
- Loss (`CE`, `Circle`, `Triplet`)
- Test epoch and run name

GPU is fixed to `0`.

## 3) Dataset Prepare Stage

`train.sh` calls dataset-specific prepare script with `--path`.

Mapping:
- Market -> `python prepare.py --path ./data/Market`
- Duke -> `python prepare_Duke.py --path ./data/Duke`
- MSMT17 -> `python prepare_MSMT.py --path ./data/MSMT17`
- CUB -> `python prepare_CUB.py --path ./data/CUB`
- VehicleID -> `python prepare_VehicleID.py --path ./data/VehicleID`
- VeRi -> `python prepare_VeRi.py --path ./data/VeRi`
- VIPeR -> `python prepare_viper.py --path ./data/VIPeR`

Notes:
- If dataset is missing and selection is `Market` or `Duke`, `train.sh` will try Google Drive auto-download via `gdown`.
- For other datasets, raw path must exist before running.

## 4) Fixed Train/Test Paths

After selecting dataset, both paths are fixed by script:
- `data_dir = <raw_path>/pytorch`
- `test_dir = <raw_path>/pytorch`

User cannot override these in `train.sh`.

## 5) Execution Pipeline

After confirmation, `train.sh` executes:
1. Install dependencies: `pip install -r requirements.txt`
2. Prepare dataset via mapped `prepare*.py`
3. Train: `python train.py ... --run_id <same_id>`
4. Test: `python test.py ... --run_id <same_id>`

## 6) Failure Handling

If dataset path is missing or prepare fails, script stops and prints:
- expected raw path
- corresponding prepare command
- expected prepared path (`<raw_path>/pytorch`)

## 7) Evaluate Only

`evaluate.sh` is used when model weights already exist under `./model`.

Interactive steps:
- Select one existing model directory (detected by `./model/<run_name>/opts.yaml`)
- Select test dataset
- Set test options (gpu ids, epoch, batchsize, ms, multi-query)

Pipeline:
1. Prepare selected test dataset via mapped `prepare*.py`
2. Evaluate: `python test.py --name <selected_run_name> ...`
