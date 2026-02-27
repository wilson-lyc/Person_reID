# 运行脚本流程说明

本文档说明当前 `train.sh` 与 `evaluate.sh` 的实际执行流程。

## 1) 启动方式

```bash
git clone https://github.com/wilson-lyc/Person_reID
cd Person_reID
```

训练+测试一体流程：

```bash
bash train.sh
```

仅评估已有模型：

```bash
bash evaluate.sh
```

## 2) train.sh（训练+测试一体）

### 交互项

`train.sh` 会让用户依次选择：
- Backbone（ResNet50、ResNet50-IBN、DenseNet、Swin、SwinV2、DINOv3、EfficientNet-B4、NAS、HRNet、ConvNeXt、PCB、ResNet50-USAM）
- Dataset（Market、Duke、MSMT17、CUB、VehicleID、VeRi、VIPeR）
- Loss（CE、Circle、Triplet、ArcFace、CosFace、Contrast、Instance、Instance-ID、Lifted、Sphere）
- 测试 epoch（`which_epoch`）
- 运行名（`run_name`）

说明：
- 默认 `gpu_ids=0`（脚本中固定）。
- 当前 `train.sh` 使用同一个数据集进行训练和测试，不支持在该脚本内分离训练/测试数据集。

### 数据准备映射

`train.sh` 会按数据集调用对应 prepare 脚本：
- Market -> `python prepare.py --path ./data/Market`
- Duke -> `python prepare_Duke.py --path ./data/Duke`
- MSMT17 -> `python prepare_MSMT.py --path ./data/MSMT17`
- CUB -> `python prepare_CUB.py --path ./data/CUB`
- VehicleID -> `python prepare_VehicleID.py --path ./data/VehicleID`
- VeRi -> `python prepare_VeRi.py --path ./data/VeRi`
- VIPeR -> `python prepare_viper.py --path ./data/VIPeR`

说明：
- 若选择 `Market` 或 `Duke` 且原始数据缺失，脚本会询问是否通过 `gdown` 自动下载。
- 其他数据集需要先手动准备原始目录。

### 固定路径规则

选择数据集后路径固定为：
- `data_dir = <raw_path>/pytorch`
- `test_dir = <raw_path>/pytorch`

### 执行阶段

确认后依次执行：
1. 安装依赖：`pip install -r requirements.txt`
2. 准备数据：调用对应 `prepare*.py`
3. 训练：`python train.py ... --run_id <same_id>`
4. 测试：`python test.py ... --run_id <same_id>`

## 3) evaluate.sh（仅评估）

用于“模型已训练完成、权重已在 `./model` 下”的场景。

### 模型选择

脚本会自动扫描 `./model/<run_name>/opts.yaml`，并以编号菜单让用户选择要评估的模型。

### 交互项

选择模型后，会继续询问：
- 测试数据集（Market、Duke、MSMT17、CUB、VehicleID、VeRi、VIPeR）
- `gpu_ids`
- `which_epoch`
- `batchsize`
- `ms`（多尺度）
- 是否启用 `multi-query`
- 是否 `skip_eval`（只提特征，不执行 `evaluate_gpu.py`）

### 执行阶段

确认后依次执行：
1. 准备测试数据集（调用对应 `prepare*.py`）
2. 评估：`python test.py --name <selected_run_name> ...`

## 4) 失败处理

当数据目录缺失或 prepare 失败时，脚本会输出：
- 期望的原始数据路径
- 对应 prepare 命令
- 期望的预处理后路径（`<raw_path>/pytorch`）
