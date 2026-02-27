#!/usr/bin/env bash
set -euo pipefail

ui_lang="zh"
echo "选择语言 / Select language:"
echo "  1) 中文 (默认)"
echo "  2) English"
read -r -p "请输入语言编号 [1]: " lang_choice
lang_choice="${lang_choice:-1}"
if [[ "$lang_choice" == "2" ]]; then
  ui_lang="en"
fi

txt() {
  case "$ui_lang:$1" in
    zh:select_backbone) echo "请选择 Backbone:" ;;
    en:select_backbone) echo "Select Backbone:" ;;
    zh:enter_backbone) echo "请输入 Backbone 编号 [1]: " ;;
    en:enter_backbone) echo "Enter backbone number [1]: " ;;
    zh:invalid_backbone) echo "Backbone 编号无效:" ;;
    en:invalid_backbone) echo "Invalid backbone number:" ;;
    zh:select_dataset) echo "请选择数据集:" ;;
    en:select_dataset) echo "Select Dataset:" ;;
    zh:enter_dataset) echo "请输入数据集编号 [1]: " ;;
    en:enter_dataset) echo "Enter dataset number [1]: " ;;
    zh:custom_data_dir) echo "请输入自定义 data_dir（pytorch 格式）: " ;;
    en:custom_data_dir) echo "Input custom data_dir (pytorch format): " ;;
    zh:custom_empty) echo "自定义 data_dir 不能为空。" ;;
    en:custom_empty) echo "Custom data_dir cannot be empty." ;;
    zh:invalid_dataset) echo "数据集编号无效:" ;;
    en:invalid_dataset) echo "Invalid dataset number:" ;;
    zh:train_data_dir) echo "训练 data_dir" ;;
    en:train_data_dir) echo "Train data_dir" ;;
    zh:test_data_dir) echo "测试 test_dir" ;;
    en:test_data_dir) echo "Test test_dir" ;;
    zh:select_loss) echo "请选择 Loss:" ;;
    en:select_loss) echo "Select Loss:" ;;
    zh:enter_loss) echo "请输入 Loss 编号 [1]: " ;;
    en:enter_loss) echo "Enter loss number [1]: " ;;
    zh:invalid_loss) echo "Loss 编号无效:" ;;
    en:invalid_loss) echo "Invalid loss number:" ;;
    zh:gpu_ids) echo "GPU 编号" ;;
    en:gpu_ids) echo "GPU ids" ;;
    zh:which_epoch) echo "测试 epoch" ;;
    en:which_epoch) echo "Which epoch for test" ;;
    zh:run_name) echo "运行名称" ;;
    en:run_name) echo "Run name" ;;
    zh:summary_backbone) echo "backbone" ;;
    en:summary_backbone) echo "backbone" ;;
    zh:summary_dataset) echo "dataset" ;;
    en:summary_dataset) echo "dataset" ;;
    zh:summary_loss) echo "loss" ;;
    en:summary_loss) echo "loss" ;;
    zh:summary_run_name) echo "run_name" ;;
    en:summary_run_name) echo "run_name" ;;
    zh:summary_run_id) echo "run_id" ;;
    en:summary_run_id) echo "run_id" ;;
    zh:summary_data_dir) echo "data_dir" ;;
    en:summary_data_dir) echo "data_dir" ;;
    zh:summary_test_dir) echo "test_dir" ;;
    en:summary_test_dir) echo "test_dir" ;;
    zh:summary_gpu_ids) echo "gpu_ids" ;;
    en:summary_gpu_ids) echo "gpu_ids" ;;
    zh:summary_epoch) echo "which_epoch" ;;
    en:summary_epoch) echo "which_epoch" ;;
    zh:step_train) echo "[1/2] 开始训练..." ;;
    en:step_train) echo "[1/2] Training..." ;;
    zh:step_test) echo "[2/2] 开始测试（test.py 内含评估）..." ;;
    en:step_test) echo "[2/2] Testing (includes evaluation in test.py)..." ;;
    zh:done) echo "完成。输出文件：" ;;
    en:done) echo "Done. Artifacts:" ;;
    zh:model_dir) echo "  模型目录 : ./model/${run_name}" ;;
    en:model_dir) echo "  model dir : ./model/${run_name}" ;;
    zh:result_txt) echo "  结果文件 : ./model/${run_name}/result.txt" ;;
    en:result_txt) echo "  result    : ./model/${run_name}/result.txt" ;;
    zh:runid_line) echo "  run_id    : ${run_id}" ;;
    en:runid_line) echo "  run_id    : ${run_id}" ;;
    *) echo "" ;;
  esac
}

echo "$(txt select_backbone)"
echo "  1) ResNet50 (baseline)"
echo "  2) ResNet50-IBN"
echo "  3) DenseNet121"
echo "  4) Swin"
read -r -p "$(txt enter_backbone)" backbone_choice
backbone_choice="${backbone_choice:-1}"

case "$backbone_choice" in
  1) backbone="resnet50"; backbone_flags=() ;;
  2) backbone="resnet50_ibn"; backbone_flags=(--ibn) ;;
  3) backbone="densenet121"; backbone_flags=(--use_dense) ;;
  4) backbone="swin"; backbone_flags=(--use_swin) ;;
  *)
    echo "$(txt invalid_backbone) $backbone_choice"
    exit 1
    ;;
esac

echo "$(txt select_dataset)"
echo "  1) Market-1501 (./Market/pytorch)"
echo "  2) DukeMTMC-reID (./Duke/pytorch)"
echo "  3) MSMT17 (./MSMT17/pytorch)"
echo "  4) Custom path"
read -r -p "$(txt enter_dataset)" dataset_choice
dataset_choice="${dataset_choice:-1}"

case "$dataset_choice" in
  1) dataset="market"; default_data_dir="./Market/pytorch" ;;
  2) dataset="duke"; default_data_dir="./Duke/pytorch" ;;
  3) dataset="msmt17"; default_data_dir="./MSMT17/pytorch" ;;
  4)
    dataset="custom"
    read -r -p "$(txt custom_data_dir)" default_data_dir
    if [[ -z "${default_data_dir}" ]]; then
      echo "$(txt custom_empty)"
      exit 1
    fi
    ;;
  *)
    echo "$(txt invalid_dataset) $dataset_choice"
    exit 1
    ;;
esac

read -r -p "$(txt train_data_dir) [${default_data_dir}]: " data_dir
data_dir="${data_dir:-$default_data_dir}"

read -r -p "$(txt test_data_dir) [${data_dir}]: " test_dir
test_dir="${test_dir:-$data_dir}"

echo "$(txt select_loss)"
echo "  1) CrossEntropy (baseline)"
echo "  2) Circle Loss (+CE, warm_epoch=5)"
echo "  3) Triplet Loss (+CE)"
read -r -p "$(txt enter_loss)" loss_choice
loss_choice="${loss_choice:-1}"

case "$loss_choice" in
  1) loss_name="ce"; loss_flags=() ;;
  2) loss_name="circle"; loss_flags=(--circle --warm_epoch 5) ;;
  3) loss_name="triplet"; loss_flags=(--triplet) ;;
  *)
    echo "$(txt invalid_loss) $loss_choice"
    exit 1
    ;;
esac

read -r -p "$(txt gpu_ids) [0]: " gpu_ids
gpu_ids="${gpu_ids:-0}"

read -r -p "$(txt which_epoch) [last]: " which_epoch
which_epoch="${which_epoch:-last}"

default_run_name="${backbone}_${dataset}_${loss_name}_$(date +%m%d_%H%M%S)"
read -r -p "$(txt run_name) [${default_run_name}]: " run_name
run_name="${run_name:-$default_run_name}"

run_id="$(python tool/run_id.py)"

echo "----------------------------------------"
echo "$(txt summary_backbone)   : $backbone"
echo "$(txt summary_dataset)    : $dataset"
echo "$(txt summary_loss)       : $loss_name"
echo "$(txt summary_run_name)   : $run_name"
echo "$(txt summary_run_id)     : $run_id"
echo "$(txt summary_data_dir)   : $data_dir"
echo "$(txt summary_test_dir)   : $test_dir"
echo "$(txt summary_gpu_ids)    : $gpu_ids"
echo "$(txt summary_epoch): $which_epoch"
echo "----------------------------------------"

train_cmd=(python train.py --gpu_ids "$gpu_ids" --name "$run_name" --data_dir "$data_dir" --run_id "$run_id")
train_cmd+=(--train_all)
train_cmd+=("${backbone_flags[@]}")
train_cmd+=("${loss_flags[@]}")

test_cmd=(python test.py --gpu_ids "$gpu_ids" --name "$run_name" --test_dir "$test_dir" --which_epoch "$which_epoch" --run_id "$run_id")

echo "$(txt step_train)"
"${train_cmd[@]}"

echo "$(txt step_test)"
"${test_cmd[@]}"

echo "$(txt done)"
echo "$(txt model_dir)"
echo "$(txt result_txt)"
echo "$(txt runid_line)"
