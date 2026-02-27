#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Copyright:
# Script built by Wilson: https://github.com/wilson-lyc
# Co-developed with Codex (OpenAI)
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
            echo "Please prepare dataset manually first, then rerun train.sh."
            print_manual_dataset_tutorial "$ds_name" "$raw_dir" "$prepared_dir" "$prepare_script"
            exit 1
          fi
          ;;
        *)
          echo "Auto-download canceled by user."
          echo "Please prepare dataset manually first, then rerun train.sh."
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

configure_hf_endpoint_for_hrnet() {
  local selected_backbone="$1"
  if [[ "$selected_backbone" != "hrnet" ]]; then
    return 0
  fi

  if [[ -n "${HF_ENDPOINT:-}" ]]; then
    echo "HRNet selected. HF_ENDPOINT is already set to: ${HF_ENDPOINT}"
    return 0
  fi

  echo "HRNet selected. Checking direct access to Hugging Face..."
  if python - <<'PY'
import sys
import urllib.request

url = "https://huggingface.co"
try:
    with urllib.request.urlopen(url, timeout=5) as resp:
        status = getattr(resp, "status", 200)
    # Any non-error HTTP response means endpoint is reachable.
    sys.exit(0 if 200 <= status < 500 else 1)
except Exception:
    sys.exit(1)
PY
  then
    echo "Hugging Face is reachable. Using direct endpoint."
  else
    export HF_ENDPOINT="https://hf-mirror.com"
    echo "Hugging Face is unreachable. Fallback to mirror: ${HF_ENDPOINT}"
  fi
}

ensure_ibn_checkpoint() {
  local selected_backbone="$1"
  if [[ "$selected_backbone" != "resnet50_ibn" ]]; then
    return 0
  fi

  local checkpoint_path="/root/.cache/torch/hub/checkpoints/resnet50_ibn_a-d9d0bb7b.pth"
  local checkpoint_dir
  checkpoint_dir="$(dirname "$checkpoint_path")"
  local direct_url="https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet50_ibn_a-d9d0bb7b.pth"
  local mirror_url="https://ghfast.top/?q=https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet50_ibn_a-d9d0bb7b.pth"
  local tmp_path="${checkpoint_path}.tmp"
  local use_mirror_confirm=""
  local download_url=""
  local download_source=""

  if [[ -s "$checkpoint_path" ]]; then
    echo "IBN checkpoint already exists: ${checkpoint_path}"
    return 0
  fi

  echo "ResNet50-IBN selected. Ensuring checkpoint:"
  echo "  ${checkpoint_path}"
  if ! mkdir -p "$checkpoint_dir"; then
    echo "Failed to create checkpoint directory: ${checkpoint_dir}"
    echo "Please manually place checkpoint at:"
    echo "  ${checkpoint_path}"
    return 1
  fi

  read -r -p "Use mirror URL for IBN checkpoint download? [Y/n]: " use_mirror_confirm
  use_mirror_confirm="${use_mirror_confirm:-Y}"
  case "$use_mirror_confirm" in
    Y|y|yes|YES)
      download_url="$mirror_url"
      download_source="mirror"
      ;;
    *)
      download_url="$direct_url"
      download_source="direct"
      ;;
  esac

  rm -f "$tmp_path"
  echo "Downloading IBN checkpoint via ${download_source} URL..."
  if command -v curl >/dev/null 2>&1; then
    # Show progress with curl when available.
    if curl -L --fail --progress-bar "$download_url" -o "$tmp_path"; then
      mv "$tmp_path" "$checkpoint_path"
      echo "IBN checkpoint downloaded from ${download_source} URL."
      return 0
    fi
  elif command -v wget >/dev/null 2>&1; then
    # Fallback with wget progress display.
    if wget --show-progress -O "$tmp_path" "$download_url"; then
      mv "$tmp_path" "$checkpoint_path"
      echo "IBN checkpoint downloaded from ${download_source} URL."
      return 0
    fi
  elif python - "$download_url" "$tmp_path" <<'PY'
import shutil
import sys
import urllib.request

url = sys.argv[1]
output = sys.argv[2]
request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
with urllib.request.urlopen(request, timeout=120) as resp, open(output, "wb") as f:
    shutil.copyfileobj(resp, f)
PY
  then
    echo "Downloaded by Python fallback (no progress bar shown)."
    mv "$tmp_path" "$checkpoint_path"
    echo "IBN checkpoint downloaded from ${download_source} URL."
    return 0
  else
    echo "No curl/wget available, and Python fallback download failed."
  fi

  rm -f "$tmp_path"
  echo "Failed to download IBN checkpoint from ${download_source} URL."
  echo "Please manually download and place file at:"
  echo "  ${checkpoint_path}"
  echo "Direct URL:"
  echo "  ${direct_url}"
  echo "Mirror URL:"
  echo "  ${mirror_url}"
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

resolve_dataset_config() {
  local choice="$1"
  case "$choice" in
    1) selected_dataset="market";    selected_raw_data_dir="./data/Market";    selected_prepare_script="prepare.py" ;;
    2) selected_dataset="duke";      selected_raw_data_dir="./data/Duke";      selected_prepare_script="prepare_Duke.py" ;;
    3) selected_dataset="msmt17";    selected_raw_data_dir="./data/MSMT17";    selected_prepare_script="prepare_MSMT.py" ;;
    4) selected_dataset="cub";       selected_raw_data_dir="./data/CUB";       selected_prepare_script="prepare_CUB.py" ;;
    5) selected_dataset="vehicleid"; selected_raw_data_dir="./data/VehicleID"; selected_prepare_script="prepare_VehicleID.py" ;;
    6) selected_dataset="veri";      selected_raw_data_dir="./data/VeRi";      selected_prepare_script="prepare_VeRi.py" ;;
    7) selected_dataset="viper";     selected_raw_data_dir="./data/VIPeR";     selected_prepare_script="prepare_viper.py" ;;
    *)
      echo "Invalid dataset number: $choice"
      exit 1
      ;;
  esac
}

resolve_dataset_config "$dataset_choice"
dataset="$selected_dataset"
raw_data_dir="$selected_raw_data_dir"
prepare_script="$selected_prepare_script"

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
read -r -p "GPU ids [0]: " input_gpu_ids
gpu_ids="${input_gpu_ids:-$gpu_ids}"

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
read -r -p "Confirm and start train+evaluate workflow? [Y/n]: " confirm_run
confirm_run="${confirm_run:-Y}"
case "$confirm_run" in
  Y|y|yes|YES)
    ;;
  *)
    echo "Canceled."
    exit 0
    ;;
esac

configure_hf_endpoint_for_hrnet "$backbone"
ensure_ibn_checkpoint "$backbone"

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
