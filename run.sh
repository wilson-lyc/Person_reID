#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Copyright:
# Script built by Wilson: https://github.com/wilson-lyc
# Project codebase: https://github.com/layumi/Person_reID_baseline_pytorch

print_banner() {
cat <<'EOF'
██████╗ ███████╗██████╗ ███████╗ ██████╗ ███╗   ██╗    ██████╗ ███████╗██╗██████╗
██╔══██╗██╔════╝██╔══██╗██╔════╝██╔═══██╗████╗  ██║    ██╔══██╗██╔════╝██║██╔══██╗
██████╔╝█████╗  ██████╔╝███████╗██║   ██║██╔██╗ ██║    ██████╔╝█████╗  ██║██║  ██║
██╔═══╝ ██╔══╝  ██╔══██╗╚════██║██║   ██║██║╚██╗██║    ██╔══██╗██╔══╝  ██║██║  ██║
██║     ███████╗██║  ██║███████║╚██████╔╝██║ ╚████║    ██║  ██║███████╗██║██████╔╝
╚═╝     ╚══════╝╚═╝  ╚═╝╚══════╝ ╚═════╝ ╚═╝  ╚═══╝    ╚═╝  ╚═╝╚══════╝╚═╝╚═════╝
EOF
}

print_manual_dataset_tutorial() {
  local ds_name="$1"
  local raw_dir="$2"
  local prepared_dir="$3"
  local prepare_script="$4"

  echo
  echo "================ MANUAL DATASET PREPARATION GUIDE ================"
  echo "[Dataset] ${ds_name}"
  echo "Raw path should be: ${raw_dir}"
  echo "Run prepare command:"
  echo "  python ${prepare_script} --path \"${raw_dir}\""
  echo "Expected prepared path:"
  echo "  ${prepared_dir}"
  echo "If training/test still fails, please check folder structure under:"
  echo "  ${prepared_dir}"
  echo "=================================================================="
  echo
}

ensure_dataset_ready() {
  local ds_name="$1"
  local raw_dir="$2"
  local prepared_dir="$3"
  local prepare_script="$4"

  if [[ ! -d "$raw_dir" ]]; then
    echo "Dataset raw path not found: ${raw_dir}"
    print_manual_dataset_tutorial "$ds_name" "$raw_dir" "$prepared_dir" "$prepare_script"
    return 1
  fi

  echo "Preparing dataset by ${prepare_script} ..."
  if ! python "$prepare_script" --path "$raw_dir"; then
    echo "Prepare failed."
    print_manual_dataset_tutorial "$ds_name" "$raw_dir" "$prepared_dir" "$prepare_script"
    return 1
  fi

  if [[ ! -d "$prepared_dir" ]]; then
    echo "Prepared path not found after prepare: ${prepared_dir}"
    print_manual_dataset_tutorial "$ds_name" "$raw_dir" "$prepared_dir" "$prepare_script"
    return 1
  fi

  return 0
}

clear
print_banner
echo "Script by Wilson: https://github.com/wilson-lyc"
echo "Project codebase: https://github.com/layumi/Person_reID_baseline_pytorch"
echo

echo "Select Backbone:"
echo "  1) ResNet50 (baseline)"
echo "  2) ResNet50-IBN"
echo "  3) DenseNet121"
echo "  4) Swin"
read -r -p "Enter backbone number [1]: " backbone_choice
backbone_choice="${backbone_choice:-1}"

case "$backbone_choice" in
  1) backbone="resnet50"; backbone_flags=() ;;
  2) backbone="resnet50_ibn"; backbone_flags=(--ibn) ;;
  3) backbone="densenet121"; backbone_flags=(--use_dense) ;;
  4) backbone="swin"; backbone_flags=(--use_swin) ;;
  *)
    echo "Invalid backbone number: $backbone_choice"
    exit 1
    ;;
esac

echo "Select Dataset:"
echo "  1) Market-1501      (./data/Market)      -> prepare.py"
echo "  2) DukeMTMC-reID    (./data/Duke)        -> prepare_Duke.py"
echo "  3) MSMT17           (./data/MSMT17)      -> prepare_MSMT.py"
echo "  4) CUB-200-2011     (./data/CUB)         -> prepare_CUB.py"
echo "  5) VehicleID        (./data/VehicleID)   -> prepare_VehicleID.py"
echo "  6) VeRi             (./data/VeRi)        -> prepare_VeRi.py"
echo "  7) VIPeR            (./data/VIPeR)       -> prepare_viper.py"
read -r -p "Enter dataset number [1]: " dataset_choice
dataset_choice="${dataset_choice:-1}"

case "$dataset_choice" in
  1) dataset="market";    raw_data_dir="./data/Market";    prepare_script="prepare.py" ;;
  2) dataset="duke";      raw_data_dir="./data/Duke";      prepare_script="prepare_Duke.py" ;;
  3) dataset="msmt17";    raw_data_dir="./data/MSMT17";    prepare_script="prepare_MSMT.py" ;;
  4) dataset="cub";       raw_data_dir="./data/CUB";       prepare_script="prepare_CUB.py" ;;
  5) dataset="vehicleid"; raw_data_dir="./data/VehicleID"; prepare_script="prepare_VehicleID.py" ;;
  6) dataset="veri";      raw_data_dir="./data/VeRi";      prepare_script="prepare_VeRi.py" ;;
  7) dataset="viper";     raw_data_dir="./data/VIPeR";     prepare_script="prepare_viper.py" ;;
  *)
    echo "Invalid dataset number: $dataset_choice"
    exit 1
    ;;
esac

data_dir="${raw_data_dir}/pytorch"
test_dir="$data_dir"

echo "Select Loss:"
echo "  1) CrossEntropy (baseline)"
echo "  2) Circle Loss (+CE, warm_epoch=5)"
echo "  3) Triplet Loss (+CE)"
read -r -p "Enter loss number [1]: " loss_choice
loss_choice="${loss_choice:-1}"

case "$loss_choice" in
  1) loss_name="ce"; loss_flags=() ;;
  2) loss_name="circle"; loss_flags=(--circle --warm_epoch 5) ;;
  3) loss_name="triplet"; loss_flags=(--triplet) ;;
  *)
    echo "Invalid loss number: $loss_choice"
    exit 1
    ;;
esac

gpu_ids="0"

read -r -p "Which epoch for test [last]: " which_epoch
which_epoch="${which_epoch:-last}"

default_run_name="${backbone}_${dataset}_${loss_name}_$(date +%m%d_%H%M%S)"
read -r -p "Run name [${default_run_name}]: " run_name
run_name="${run_name:-$default_run_name}"

run_id="$(python tool/run_id.py)"

clear
print_banner
echo "Script by Wilson: https://github.com/wilson-lyc"
echo "Project codebase: https://github.com/layumi/Person_reID_baseline_pytorch"
echo

echo "----------------------------------------"
echo "backbone      : $backbone"
echo "dataset       : $dataset"
echo "prepare_script: $prepare_script"
echo "raw_data_dir  : $raw_data_dir"
echo "data_dir      : $data_dir"
echo "test_dir      : $test_dir"
echo "loss          : $loss_name"
echo "run_name      : $run_name"
echo "run_id        : $run_id"
echo "gpu_ids       : $gpu_ids"
echo "which_epoch   : $which_epoch"
echo "----------------------------------------"
read -r -p "Confirm and start run? [Y/n]: " confirm_run
confirm_run="${confirm_run:-Y}"
case "$confirm_run" in
  Y|y|yes|YES)
    ;;
  *)
    echo "Canceled."
    exit 0
    ;;
esac

train_cmd=(python train.py --gpu_ids "$gpu_ids" --name "$run_name" --data_dir "$data_dir" --run_id "$run_id")
train_cmd+=(--train_all)
train_cmd+=("${backbone_flags[@]}")
train_cmd+=("${loss_flags[@]}")

test_cmd=(python test.py --gpu_ids "$gpu_ids" --name "$run_name" --test_dir "$test_dir" --which_epoch "$which_epoch" --run_id "$run_id")

echo "[1/4] Installing dependencies from requirements.txt..."
python -m pip install -r requirements.txt

echo "[2/4] Preparing dataset..."
ensure_dataset_ready "$dataset" "$raw_data_dir" "$data_dir" "$prepare_script"

echo "[3/4] Training..."
"${train_cmd[@]}"

echo "[4/4] Testing..."
"${test_cmd[@]}"

echo "Done. Artifacts:"
echo "  model dir : ./model/${run_name}"
echo "  result    : ./model/${run_name}/result.txt"
echo "  run_id    : ${run_id}"
