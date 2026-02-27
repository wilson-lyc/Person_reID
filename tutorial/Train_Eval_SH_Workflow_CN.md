# 训练与评估脚本技术文档（train.sh / evaluate.sh）

本文档面向仓库当前版本脚本实现，重点说明 `train.sh` 与 `evaluate.sh` 的实际工作流、关键分支和产物路径。

## 1. 脚本定位

- `train.sh`: 交互式完成“依赖安装 -> 数据准备 -> 训练 -> 测试”一体流程。
- `evaluate.sh`: 交互式完成“选择已有模型 -> 准备测试集 -> 评估”流程。
- `DDP.sh`: 提供最小分布式训练入口（固定参数调用 `train_DDP.py`）。

## 2. 总体工作流

### 2.1 train.sh

1. 交互选择配置
2. 二次确认配置摘要
3. 环境/权重准备（HRNet 镜像、IBN 预训练权重）
4. 安装依赖 `pip install -r requirements.txt`
5. 数据集检查与 `prepare*.py` 处理
6. 训练：`python train.py ...`
7. 测试：`python test.py ...`
8. 输出结果位置

### 2.2 evaluate.sh

1. 扫描 `./model/**/opts.yaml` 并选择评估模型
2. 选择测试数据集与评估参数
3. 二次确认配置摘要
4. 根据模型配置自动处理 HRNet 的 Hugging Face 可达性
5. 数据集检查与 `prepare*.py` 处理
6. 评估：`python test.py ...`
7. 输出结果位置

## 3. train.sh 详细流程

### 3.1 交互输入

`train.sh` 会依次采集：
- Backbone: `resnet50`、`resnet50_ibn`、`densenet`、`swin`、`swinv2`、`dino`、`efficientnet_b4`、`nas`、`hrnet`、`convnext`、`pcb`、`resnet50_usam`
- Dataset: `market`、`duke`、`msmt17`、`cub`、`vehicleid`、`veri`、`viper`
- Loss: `ce`、`circle`、`triplet`、`arcface`、`cosface`、`contrast`、`instance`、`instance_id`、`lifted`、`sphere`
- `gpu_ids`（默认 `0`）
- `which_epoch`（默认 `last`）
- `run_name`（默认 `<backbone>_<dataset>_<loss>_<run_id>`）

其中 `run_id` 由 `python tool/run_id.py` 生成，并透传给训练与测试命令。

### 3.2 数据集映射与准备

数据集映射规则：

- `market`: `raw_data_dir=./data/Market`，`prepare.py`
- `duke`: `raw_data_dir=./data/Duke`，`prepare_Duke.py`
- `msmt17`: `raw_data_dir=./data/MSMT17`，`prepare_MSMT.py`
- `cub`: `raw_data_dir=./data/CUB`，`prepare_CUB.py`
- `vehicleid`: `raw_data_dir=./data/VehicleID`，`prepare_VehicleID.py`
- `veri`: `raw_data_dir=./data/VeRi`，`prepare_VeRi.py`
- `viper`: `raw_data_dir=./data/VIPeR`，`prepare_viper.py`

统一目标目录：`data_dir=<raw_data_dir>/pytorch`

`ensure_dataset_ready()` 负责：
- 检查原始目录结构是否满足当前数据集要求
- 调用对应 `prepare*.py --path <raw_data_dir>`
- 校验 `pytorch` 目录是否生成

当原始数据缺失/不完整时，脚本会给出手动准备提示并中止本次流程。

### 3.3 网络下载相关分支

- 选择 `hrnet` 时，脚本会询问是否设置 `HF_ENDPOINT=https://hf-mirror.com`。
- 选择 `resnet50_ibn` 时，脚本会检查 `/root/.cache/torch/hub/checkpoints/resnet50_ibn_a-d9d0bb7b.pth`，缺失则尝试下载（支持 `curl`/`wget`，可选镜像 URL）。

### 3.4 命令组装与执行顺序

训练命令骨架：

```bash
python train.py \
  --gpu_ids <gpu_ids> \
  --name <run_name> \
  --data_dir <raw_data_dir>/pytorch \
  --run_id <run_id> \
  --train_all \
  <backbone_flags> \
  <loss_flags>
```

测试命令骨架：

```bash
python test.py \
  --gpu_ids <gpu_ids> \
  --name <run_name> \
  --test_dir <raw_data_dir>/pytorch \
  --which_epoch <which_epoch> \
  --run_id <run_id>
```

固定执行阶段：

1. `python -m pip install -r requirements.txt`
2. `ensure_dataset_ready ...`
3. `train.py`
4. `test.py`

### 3.5 输出产物

- 模型目录：`./model/<run_name>`
- 结果文件：`./model/<run_name>/result.txt`
- 本次流程标识：`run_id`

## 4. evaluate.sh 详细流程

### 4.1 模型选择

脚本优先扫描：

```text
./model/<run_name>/opts.yaml
```

有扫描结果时提供编号菜单；无结果时要求手工输入 `run_name`。若 `opts.yaml` 缺失则直接退出。

### 4.2 评估参数输入

- 测试集：`market`、`duke`、`msmt17`、`cub`、`vehicleid`、`veri`、`viper`
- `gpu_ids`（默认 `0`）
- `which_epoch`（默认 `last`）
- `batchsize`（默认 `256`）
- `ms`（默认 `1`）
- 是否开启 `--multi`
- 是否开启 `--skip_eval`

`run_id` 同样由 `tool/run_id.py` 生成。

### 4.3 数据准备与自动下载

`evaluate.sh` 的 `ensure_dataset_ready()` 中：
- `market`/`duke` 缺失时，支持询问后自动下载（`gdown` + 解压归一化）
- 其余数据集需手工准备原始目录
- 之后统一执行 `prepare*.py --path <raw_data_dir>`

当前实现注意点：`evaluate.sh` 的 `dataset_has_required_structure()` 仅检查 `query`、`bounding_box_train`、`bounding_box_test` 三个目录（更偏向 Market/Duke 结构），因此在 `msmt17/cub/vehicleid/veri/viper` 上可能触发“原始目录不完整”提示，需要结合实际目录结构与脚本逻辑排查。

### 4.4 HRNet 网络可达性自适应

`configure_hf_endpoint_for_model()` 会读取 `./model/<run_name>/opts.yaml`：
- 若该模型是 HRNet（`use_hr: true`），先探测 `https://huggingface.co`
- 若不可达，则自动设置 `HF_ENDPOINT=https://hf-mirror.com`

### 4.5 命令组装与执行顺序

评估命令骨架：

```bash
python test.py \
  --gpu_ids <gpu_ids> \
  --name <run_name> \
  --test_dir <test_raw_data_dir>/pytorch \
  --which_epoch <which_epoch> \
  --batchsize <batchsize> \
  --ms <ms> \
  --run_id <run_id> \
  [--multi] \
  [--skip_eval]
```

固定执行阶段：

1. `ensure_dataset_ready ...`
2. `python test.py ...`

### 4.6 输出产物

- 模型目录：`./model/<run_name>`
- 结果文件：`./model/<run_name>/result.txt`
- 本次评估标识：`run_id`

## 5. 与 DDP.sh 的关系

`DDP.sh` 当前是最小入口：

```bash
python -m torch.distributed.launch --nproc_per_node=2 --master_port=6005 train_DDP.py
```

说明：
- 它不包含 `train.sh`/`evaluate.sh` 的交互流程、数据准备和自动下载逻辑。
- 适合已明确分布式参数且数据已准备好的场景。

## 6. 常见失败点与排查顺序

1. 原始数据目录结构不完整：先核对 `raw_data_dir` 及数据集必需子目录。
2. `prepare*.py` 执行失败：单独运行 prepare 命令定位报错。
3. `opts.yaml` 缺失：说明训练过程未完整落盘，需先完成训练。
4. HRNet/IBN 下载失败：检查网络与镜像选择，必要时手工放置权重。
5. `result.txt` 未生成：优先检查 `test.py` 阶段日志与参数。

## 7. 典型使用方式

训练并立即测试：

```bash
bash train.sh
```

仅评估已有模型：

```bash
bash evaluate.sh
```

分布式训练（简版入口）：

```bash
bash DDP.sh
```
