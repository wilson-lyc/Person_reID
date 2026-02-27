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

dataset_has_required_structure() {
  local path="$1"
  [[ -d "${path}/query" && -d "${path}/bounding_box_train" && -d "${path}/bounding_box_test" ]]
}

try_auto_download_dataset() {
  local ds_name="$1"
  local raw_dir="$2"
  local file_id=""
  local archive_name=""

  case "$ds_name" in
    market)
      file_id="0B8-rUzbwVRk0c054eEozWG9COHM"
      archive_name="Market-1501-v15.09.15.zip"
      ;;
    duke)
      file_id="1jjE85dRCMOgRtvJ5RQV9-Afs-2_5dY3O"
      archive_name="DukeMTMC-reID.zip"
      ;;
    *)
      return 1
      ;;
  esac

  local parent_dir
  parent_dir="$(dirname "$raw_dir")"
  local archive_path="${parent_dir}/${archive_name}"
  local tmp_extract_dir="${parent_dir}/.tmp_extract_${ds_name}"

  echo "Dataset missing. Trying Google Drive auto-download for ${ds_name} ..."
  mkdir -p "$parent_dir"

  echo "Installing gdown (if needed) ..."
  if ! python -m pip install gdown; then
    echo "Auto-download failed: unable to install gdown."
    return 2
  fi

  echo "Downloading archive to ${archive_path} ..."
  if ! python -m gdown --id "$file_id" --output "$archive_path"; then
    echo "Auto-download failed: cannot access Google Drive or download was blocked."
    return 2
  fi

  echo "Extracting and normalizing dataset layout ..."
  if ! python - "$raw_dir" "$archive_path" "$tmp_extract_dir" <<'PY'
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path

raw_dir = Path(sys.argv[1]).resolve()
archive_path = Path(sys.argv[2]).resolve()
tmp_extract_dir = Path(sys.argv[3]).resolve()

required = {"query", "bounding_box_train", "bounding_box_test"}

if tmp_extract_dir.exists():
    shutil.rmtree(tmp_extract_dir)
tmp_extract_dir.mkdir(parents=True, exist_ok=True)

if archive_path.suffix.lower() == ".zip":
    with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall(tmp_extract_dir)
elif archive_path.suffix.lower() in {".tar", ".gz", ".tgz", ".bz2", ".xz"}:
    with tarfile.open(archive_path, "r:*") as tf:
        tf.extractall(tmp_extract_dir)
else:
    raise RuntimeError(f"Unsupported archive format: {archive_path}")

def is_raw_root(path: Path) -> bool:
    if not path.is_dir():
        return False
    names = {p.name for p in path.iterdir() if p.is_dir()}
    return required.issubset(names)

candidate = None
if is_raw_root(tmp_extract_dir):
    candidate = tmp_extract_dir
else:
    for p in tmp_extract_dir.rglob("*"):
        if is_raw_root(p):
            candidate = p
            break

if candidate is None:
    raise RuntimeError(
        f"Failed to locate dataset root after extracting {archive_path}. "
        f"Expected folders: {sorted(required)}"
    )

raw_dir.mkdir(parents=True, exist_ok=True)
for item in candidate.iterdir():
    dst = raw_dir / item.name
    if dst.exists():
        if dst.is_dir():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    shutil.move(str(item), str(dst))

shutil.rmtree(tmp_extract_dir, ignore_errors=True)
print(f"Dataset prepared at: {raw_dir}")
PY
  then
    echo "Auto-download failed: archive extraction/normalization error."
    return 2
  fi
}

ensure_dataset_ready() {
  local ds_name="$1"
  local raw_dir="$2"
  local prepared_dir="$3"
  local prepare_script="$4"

  if ! dataset_has_required_structure "$raw_dir"; then
    if [[ "$ds_name" == "market" || "$ds_name" == "duke" ]]; then
      echo "Dataset path is missing or incomplete: ${raw_dir}"
      read -r -p "Do you want to auto-download ${ds_name} from Google Drive? [Y/n]: " auto_download_confirm
      auto_download_confirm="${auto_download_confirm:-Y}"
      case "$auto_download_confirm" in
        Y|y|yes|YES)
          if ! try_auto_download_dataset "$ds_name" "$raw_dir"; then
            echo "Google Drive auto-download is unavailable for ${ds_name}."
            echo "Please prepare dataset manually first, then rerun run.sh."
            print_manual_dataset_tutorial "$ds_name" "$raw_dir" "$prepared_dir" "$prepare_script"
            exit 1
          fi
          ;;
        *)
          echo "Auto-download canceled by user."
          echo "Please prepare dataset manually first, then rerun run.sh."
          print_manual_dataset_tutorial "$ds_name" "$raw_dir" "$prepared_dir" "$prepare_script"
          exit 1
          ;;
      esac
    else
      echo "Dataset raw path not found or incomplete: ${raw_dir}"
      print_manual_dataset_tutorial "$ds_name" "$raw_dir" "$prepared_dir" "$prepare_script"
      return 1
    fi
  fi

  if ! dataset_has_required_structure "$raw_dir"; then
    echo "Dataset raw path is still incomplete after auto-download: ${raw_dir}"
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
echo "  3) DenseNet"
echo "  4) Swin"
echo "  5) SwinV2"
echo "  6) DINOv3"
echo "  7) EfficientNet-B4"
echo "  8) NAS"
echo "  9) HRNet"
echo "  10) ConvNeXt"
echo "  11) PCB (ResNet50+PCB)"
echo "  12) ResNet50-USAM"
read -r -p "Enter backbone number [1]: " backbone_choice
backbone_choice="${backbone_choice:-1}"

case "$backbone_choice" in
  1) backbone="resnet50"; backbone_flags=() ;;
  2) backbone="resnet50_ibn"; backbone_flags=(--ibn) ;;
  3) backbone="densenet"; backbone_flags=(--use_dense) ;;
  4) backbone="swin"; backbone_flags=(--use_swin) ;;
  5) backbone="swinv2"; backbone_flags=(--use_swinv2) ;;
  6) backbone="dino"; backbone_flags=(--use_dino) ;;
  7) backbone="efficientnet_b4"; backbone_flags=(--use_efficient) ;;
  8) backbone="nas"; backbone_flags=(--use_NAS) ;;
  9) backbone="hrnet"; backbone_flags=(--use_hr) ;;
  10) backbone="convnext"; backbone_flags=(--use_convnext) ;;
  11) backbone="pcb"; backbone_flags=(--PCB) ;;
  12) backbone="resnet50_usam"; backbone_flags=(--usam) ;;
  *)
    echo "Invalid backbone number: $backbone_choice"
    exit 1
    ;;
esac

echo "Select Dataset:"
echo "  1) Market-1501 (auto-download available)"
echo "  2) DukeMTMC-reID (auto-download available)"
echo "  3) MSMT17"
echo "  4) CUB-200-2011"
echo "  5) VehicleID"
echo "  6) VeRi"
echo "  7) VIPeR"
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
echo "  4) ArcFace Loss (+CE)"
echo "  5) CosFace Loss (+CE)"
echo "  6) Contrast Loss (+CE)"
echo "  7) Instance Loss (+CE)"
echo "  8) Instance-ID Loss (+CE)"
echo "  9) Lifted Loss (+CE)"
echo "  10) Sphere Loss (+CE)"
read -r -p "Enter loss number [1]: " loss_choice
loss_choice="${loss_choice:-1}"

case "$loss_choice" in
  1) loss_name="ce"; loss_flags=() ;;
  2) loss_name="circle"; loss_flags=(--circle --warm_epoch 5) ;;
  3) loss_name="triplet"; loss_flags=(--triplet) ;;
  4) loss_name="arcface"; loss_flags=(--arcface) ;;
  5) loss_name="cosface"; loss_flags=(--cosface) ;;
  6) loss_name="contrast"; loss_flags=(--contrast) ;;
  7) loss_name="instance"; loss_flags=(--instance) ;;
  8) loss_name="instance_id"; loss_flags=(--instance_id) ;;
  9) loss_name="lifted"; loss_flags=(--lifted) ;;
  10) loss_name="sphere"; loss_flags=(--sphere) ;;
  *)
    echo "Invalid loss number: $loss_choice"
    exit 1
    ;;
esac

gpu_ids="0"

read -r -p "Which epoch for test [last]: " which_epoch
which_epoch="${which_epoch:-last}"

run_id="$(python tool/run_id.py)"
default_run_name="${backbone}_${dataset}_${loss_name}_${run_id}"
read -r -p "Run name [${default_run_name}]: " run_name
run_name="${run_name:-$default_run_name}"

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

clear
print_banner

echo "[2/4] Preparing dataset..."
ensure_dataset_ready "$dataset" "$raw_data_dir" "$data_dir" "$prepare_script"

clear
print_banner

echo "[3/4] Training..."
"${train_cmd[@]}"

clear
print_banner

echo "[4/4] Testing..."
"${test_cmd[@]}"

echo "Done. Artifacts:"
echo "  model dir : ./model/${run_name}"
echo "  result    : ./model/${run_name}/result.txt"
echo "  run_id    : ${run_id}"
