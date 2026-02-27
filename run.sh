#!/usr/bin/env bash
set -euo pipefail

clear

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
echo "  1) Market-1501 (./Market/pytorch)"
echo "  2) DukeMTMC-reID (./Duke/pytorch)"
echo "  3) MSMT17 (./MSMT17/pytorch)"
echo "  4) Custom path"
read -r -p "Enter dataset number [1]: " dataset_choice
dataset_choice="${dataset_choice:-1}"

case "$dataset_choice" in
  1) dataset="market"; default_data_dir="./Market/pytorch" ;;
  2) dataset="duke"; default_data_dir="./Duke/pytorch" ;;
  3) dataset="msmt17"; default_data_dir="./MSMT17/pytorch" ;;
  4)
    dataset="custom"
    read -r -p "Input custom data_dir (pytorch format): " default_data_dir
    if [[ -z "${default_data_dir}" ]]; then
      echo "Custom data_dir cannot be empty."
      exit 1
    fi
    ;;
  *)
    echo "Invalid dataset number: $dataset_choice"
    exit 1
    ;;
esac

read -r -p "Train data_dir [${default_data_dir}]: " data_dir
data_dir="${data_dir:-$default_data_dir}"

read -r -p "Test test_dir [${data_dir}]: " test_dir
test_dir="${test_dir:-$data_dir}"

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

echo "----------------------------------------"
echo "backbone   : $backbone"
echo "dataset    : $dataset"
echo "loss       : $loss_name"
echo "run_name   : $run_name"
echo "run_id     : $run_id"
echo "data_dir   : $data_dir"
echo "test_dir   : $test_dir"
echo "gpu_ids    : $gpu_ids"
echo "which_epoch: $which_epoch"
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

echo "[1/2] Training..."
"${train_cmd[@]}"

echo "[2/2] Testing..."
"${test_cmd[@]}"

echo "Done. Artifacts:"
echo "  model dir : ./model/${run_name}"
echo "  result    : ./model/${run_name}/result.txt"
echo "  run_id    : ${run_id}"
