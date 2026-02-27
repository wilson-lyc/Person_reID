#!/usr/bin/env bash
set -euo pipefail

echo "Select run mode:"
echo "  1) baseline"
echo "  2) baseline_all"
echo "  3) pcb"
echo "  4) dense"
echo "  5) swin"
read -r -p "Enter mode number [1]: " mode_choice
mode_choice="${mode_choice:-1}"

case "$mode_choice" in
  1) mode="baseline" ;;
  2) mode="baseline_all" ;;
  3) mode="pcb" ;;
  4) mode="dense" ;;
  5) mode="swin" ;;
  *)
    echo "Invalid mode number: $mode_choice"
    exit 1
    ;;
esac

read -r -p "Train data_dir [../Market/pytorch]: " data_dir
data_dir="${data_dir:-../Market/pytorch}"

read -r -p "Test test_dir [${data_dir}]: " test_dir
test_dir="${test_dir:-$data_dir}"

read -r -p "GPU ids [0]: " gpu_ids
gpu_ids="${gpu_ids:-0}"

read -r -p "Which epoch for test [last]: " which_epoch
which_epoch="${which_epoch:-last}"

default_run_name="${mode}_$(date +%m%d_%H%M%S)"
read -r -p "Run name [${default_run_name}]: " run_name
run_name="${run_name:-$default_run_name}"

run_id="$(python tool/run_id.py)"

echo "----------------------------------------"
echo "mode       : $mode"
echo "run_name   : $run_name"
echo "run_id     : $run_id"
echo "data_dir   : $data_dir"
echo "test_dir   : $test_dir"
echo "gpu_ids    : $gpu_ids"
echo "which_epoch: $which_epoch"
echo "----------------------------------------"

train_cmd=(python train.py --gpu_ids "$gpu_ids" --name "$run_name" --data_dir "$data_dir" --run_id "$run_id")

case "$mode" in
  baseline)
    ;;
  baseline_all)
    train_cmd+=(--train_all)
    ;;
  pcb)
    train_cmd+=(--PCB --train_all --lr 0.02)
    ;;
  dense)
    train_cmd+=(--use_dense --train_all)
    ;;
  swin)
    train_cmd+=(--use_swin)
    ;;
esac

test_cmd=(python test.py --gpu_ids "$gpu_ids" --name "$run_name" --test_dir "$test_dir" --which_epoch "$which_epoch" --run_id "$run_id")

echo "[1/2] Training..."
"${train_cmd[@]}"

echo "[2/2] Testing (includes evaluation in test.py)..."
"${test_cmd[@]}"

echo "Done. Artifacts:"
echo "  model dir : ./model/${run_name}"
echo "  result    : ./model/${run_name}/result.txt"
echo "  run_id    : ${run_id}"
