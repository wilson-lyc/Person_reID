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

extract_zip() {
  local zip_path="$1"
  local dest_dir="$2"
  python -c "import zipfile; zipfile.ZipFile(r'${zip_path}').extractall(r'${dest_dir}')"
}

print_manual_dataset_tutorial() {
  local ds="$1"
  local raw_dir="$2"
  local prepared_dir="$3"
  echo
  echo "================ MANUAL DATASET PREPARATION GUIDE ================"
  echo "Auto-download from Google Drive failed (likely due to network restrictions)."
  echo "Please prepare the dataset manually, then rerun this script."
  echo
  if [[ "$ds" == "market" ]]; then
    echo "[Dataset] Market-1501"
    echo "1) Download and extract Market-1501 manually to: $raw_dir"
    echo "2) Run:"
    echo "   python prepare.py --dataset market --download_path \"$raw_dir\""
  elif [[ "$ds" == "duke" ]]; then
    echo "[Dataset] DukeMTMC-reID"
    echo "1) Download and extract DukeMTMC-reID manually to: $raw_dir"
    echo "2) Run:"
    echo "   python prepare.py --dataset duke --download_path \"$raw_dir\""
  elif [[ "$ds" == "msmt17" ]]; then
    echo "[Dataset] MSMT17"
    echo "1) Download and extract MSMT17 manually to: $raw_dir"
    echo "2) Run:"
    echo "   python prepare.py --dataset msmt17 --download_path \"$raw_dir\""
  else
    echo "[Dataset] Custom"
    echo "Please ensure your prepared dataset path includes:"
    echo "  $prepared_dir/train"
    echo "  $prepared_dir/query"
    echo "  $prepared_dir/gallery"
  fi
  echo
  echo "Expected prepared path:"
  echo "  $prepared_dir"
  echo "=================================================================="
  echo
}

ensure_dataset_ready() {
  local ds="$1"
  local prepared_dir="$2"
  local raw_dir="$3"

  if [[ -d "${prepared_dir}/train" && -d "${prepared_dir}/query" && -d "${prepared_dir}/gallery" ]]; then
    echo "Prepared dataset already exists: ${prepared_dir}"
    return 0
  fi

  if [[ "$ds" == "market" ]]; then
    if [[ ! -d "$raw_dir" && ! -d "./Market-1501-v15.09.15" ]]; then
      echo "Market-1501 not found locally. Downloading from Google Drive..."
      if ! python -m gdown "https://drive.google.com/uc?id=0B8-rUzbwVRk0c054eEozWG9COHM" -O "./Market-1501-v15.09.15.zip"; then
        print_manual_dataset_tutorial "$ds" "$raw_dir" "$prepared_dir"
        return 1
      fi
      if ! extract_zip "./Market-1501-v15.09.15.zip" "."; then
        print_manual_dataset_tutorial "$ds" "$raw_dir" "$prepared_dir"
        return 1
      fi
    fi
    echo "Preparing Market-1501..."
    if ! python prepare.py --dataset market --download_path "$raw_dir"; then
      print_manual_dataset_tutorial "$ds" "$raw_dir" "$prepared_dir"
      return 1
    fi
    return 0
  fi

  if [[ "$ds" == "duke" ]]; then
    if [[ ! -d "$raw_dir" ]]; then
      echo "DukeMTMC-reID not found locally. Downloading from Google Drive..."
      if ! python -m gdown "https://drive.google.com/uc?id=1jjE85dRCMOgRtvJ5RQV9-Afs-2_5dY3O" -O "./DukeMTMC-reID.zip"; then
        print_manual_dataset_tutorial "$ds" "$raw_dir" "$prepared_dir"
        return 1
      fi
      if ! extract_zip "./DukeMTMC-reID.zip" "."; then
        print_manual_dataset_tutorial "$ds" "$raw_dir" "$prepared_dir"
        return 1
      fi
    fi
    echo "Preparing DukeMTMC-reID..."
    if ! python prepare.py --dataset duke --download_path "$raw_dir"; then
      print_manual_dataset_tutorial "$ds" "$raw_dir" "$prepared_dir"
      return 1
    fi
    return 0
  fi

  if [[ "$ds" == "msmt17" ]]; then
    if [[ ! -d "$raw_dir" ]]; then
      echo "MSMT17 raw dataset not found at: $raw_dir"
      print_manual_dataset_tutorial "$ds" "$raw_dir" "$prepared_dir"
      return 1
    fi
    echo "Preparing MSMT17..."
    if ! python prepare.py --dataset msmt17 --download_path "$raw_dir"; then
      print_manual_dataset_tutorial "$ds" "$raw_dir" "$prepared_dir"
      return 1
    fi
    return 0
  fi

  if [[ "$ds" == "custom" ]]; then
    if [[ -d "${prepared_dir}/train" && -d "${prepared_dir}/query" && -d "${prepared_dir}/gallery" ]]; then
      return 0
    fi
    echo "Custom dataset path is missing prepared folders:"
    echo "  expected: ${prepared_dir}/train, ${prepared_dir}/query, ${prepared_dir}/gallery"
    print_manual_dataset_tutorial "$ds" "$raw_dir" "$prepared_dir"
    return 1
  fi

  echo "Unknown dataset key: $ds"
  return 1
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
echo "  1) Market-1501 (./Market/pytorch)"
echo "  2) DukeMTMC-reID (./Duke/pytorch)"
echo "  3) MSMT17 (./MSMT17/pytorch)"
echo "  4) Custom path"
read -r -p "Enter dataset number [1]: " dataset_choice
dataset_choice="${dataset_choice:-1}"

case "$dataset_choice" in
  1) dataset="market"; raw_data_dir="./Market"; default_data_dir="./Market/pytorch" ;;
  2) dataset="duke"; raw_data_dir="./DukeMTMC-reID"; default_data_dir="./DukeMTMC-reID/pytorch" ;;
  3) dataset="msmt17"; raw_data_dir="./MSMT17_V1"; default_data_dir="./MSMT17_V1/pytorch" ;;
  4)
    dataset="custom"
    raw_data_dir=""
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
if [[ "$dataset" != "custom" && "$data_dir" != "$default_data_dir" ]]; then
  raw_data_dir="${data_dir%/pytorch}"
fi

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
print_banner
echo "Script by Wilson: https://github.com/wilson-lyc"
echo "Project codebase: https://github.com/layumi/Person_reID_baseline_pytorch"
echo

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

echo "[1/4] Installing dependencies from requirements.txt..."
python -m pip install -r requirements.txt

echo "[2/4] Checking/downloading/preparing dataset..."
python -m pip install --upgrade gdown
ensure_dataset_ready "$dataset" "$data_dir" "$raw_data_dir"

echo "[3/4] Training..."
"${train_cmd[@]}"

echo "[4/4] Testing..."
"${test_cmd[@]}"

echo "Done. Artifacts:"
echo "  model dir : ./model/${run_name}"
echo "  result    : ./model/${run_name}/result.txt"
echo "  run_id    : ${run_id}"
