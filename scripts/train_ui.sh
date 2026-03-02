#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

COMMON_UI_LIB="${SCRIPT_DIR}/common_ui.sh"
if [[ ! -f "$COMMON_UI_LIB" ]]; then
  echo "Missing global UI library: ${COMMON_UI_LIB}"
  exit 1
fi
source "$COMMON_UI_LIB"

CURRENT_SCRIPT="$(basename "$0")"
PLATFORM="unknown"
PYTHON_BIN=""

# =========================
# Runtime detection
# =========================
detect_platform() {
  local os_name
  os_name="$(uname -s 2>/dev/null || echo unknown)"
  case "$os_name" in
    Linux*)  PLATFORM="linux" ;;
    Darwin*) PLATFORM="macos" ;;
    MINGW*|MSYS*|CYGWIN*) PLATFORM="windows" ;;
    *)       PLATFORM="unknown" ;;
  esac
}

resolve_python_bin() {
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
    return 0
  fi
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
    return 0
  fi
  ui_error "Python executable not found. Please install python3/python and retry."
  return 1
}

# =========================
# Dataset preparation
# =========================
print_manual_dataset_tutorial() {
  local ds_name="$1"
  local raw_dir="$2"
  echo "1. Download the ${ds_name} dataset from internet."
  echo "2. Place the downloaded dataset at: ${raw_dir}"
  echo "3. Rerun this script after dataset is ready."
}

check_dataset_structure() {
  local ds_name="$1"
  local path="$2"

  case "$ds_name" in
    market|duke)
      [[ -d "${path}/query" && -d "${path}/bounding_box_train" && -d "${path}/bounding_box_test" ]]
      ;;
    msmt)
      [[ -d "${path}/train" && -d "${path}/test" \
         && -f "${path}/list_train.txt" && -f "${path}/list_val.txt" \
         && -f "${path}/list_query.txt" && -f "${path}/list_gallery.txt" ]]
      ;;
    cub)
      [[ -d "${path}/images" ]]
      ;;
    vehicleid)
      [[ -d "${path}/image" \
         && -f "${path}/attribute/img2vid.txt" \
         && -f "${path}/train_test_split/train_list.txt" \
         && -f "${path}/train_test_split/test_list_800.txt" \
         && -f "${path}/train_test_split/test_list_1600.txt" \
         && -f "${path}/train_test_split/test_list_2400.txt" ]]
      ;;
    veri)
      [[ -d "${path}/image_train" && -d "${path}/image_test" && -d "${path}/image_query" ]]
      ;;
    viper)
      [[ -d "${path}/cam_a" && -d "${path}/cam_b" ]]
      ;;
    *)
      return 1
      ;;
  esac
}

prepare_dataset() {
  local ds_name="$1"
  local raw_dir="$2"
  local prepared_dir="$3"
  local prepare_script="$4"

  if [[ ! -f "$prepare_script" ]]; then
    ui_error "${prepare_script} was not found. Please check your project files, then rerun this script."
    return 1
  fi

  if ! check_dataset_structure "$ds_name" "$raw_dir"; then
    ui_error "${ds_name} was not found or has unexpected structure. Please follow the instructions to prepare it."
    print_manual_dataset_tutorial "$ds_name" "$raw_dir"
    ui_tip "After preparing the dataset, rerun ${CURRENT_SCRIPT}."
    return 1
  fi

  ui_info "Preparing dataset..."
  if ! "$PYTHON_BIN" "$prepare_script" --path "$raw_dir"; then
    ui_error "Dataset preparation failed. Error while running ${prepare_script}."
    return 1
  fi

  if [[ ! -d "$prepared_dir" ]]; then
    ui_error "Dataset preparation failed. Target directory ${prepared_dir} was not found."
    return 1
  fi

  return 0
}

# =========================
# Mirror configuration
# =========================
HF_MIRROR_STATUS="no"
IBN_MIRROR_STATUS="no"
IBN_DOWNLOAD_SOURCE="direct"

mirror_config_hf() {
  local selected_backbone="$1"
  local backbone_tip
  local use_hf_mirror="no"

  case "$selected_backbone" in
    hrnet)    backbone_tip="HRNet" ;;
    convnext) backbone_tip="ConvNeXt" ;;
    swin)     backbone_tip="Swin" ;;
    swinv2)   backbone_tip="SwinV2" ;;
    dino)     backbone_tip="DINOv3" ;;
    *)
      HF_MIRROR_STATUS="no"
      return 0
      ;;
  esac

  [[ -n "${HF_ENDPOINT:-}" ]] && ui_info "Current HF_ENDPOINT: ${HF_ENDPOINT}"

  ui_yes_no use_hf_mirror \
    "${backbone_tip} may download weights from Hugging Face. Use mirror (https://hf-mirror.com)?" \
    "1"

  if [[ "$use_hf_mirror" == "yes" ]]; then
    export HF_ENDPOINT="https://hf-mirror.com"
    HF_MIRROR_STATUS="yes"
    ui_info "Using mirror endpoint: ${HF_ENDPOINT}"
  else
    unset HF_ENDPOINT 2>/dev/null
    HF_MIRROR_STATUS="no"
    ui_info "Using direct Hugging Face endpoint."
  fi
}

is_ibn_backbone() {
  local selected_backbone="$1"
  [[ "$selected_backbone" == "resnet50_ibn" || "$selected_backbone" == "resnet_ibn" ]]
}

mirror_config_ibn() {
  local selected_backbone="$1"
  local use_mirror="no"

  if ! is_ibn_backbone "$selected_backbone"; then
    IBN_MIRROR_STATUS="no"
    IBN_DOWNLOAD_SOURCE="direct"
    return 0
  fi

  ui_yes_no use_mirror "Use mirror URL for IBN checkpoint download?" "1"
  if [[ "$use_mirror" == "yes" ]]; then
    IBN_DOWNLOAD_SOURCE="mirror"
    IBN_MIRROR_STATUS="yes"
  else
    IBN_DOWNLOAD_SOURCE="direct"
    IBN_MIRROR_STATUS="no"
  fi
}

download_ibn_backbone() {
  local selected_backbone="$1"
  if ! is_ibn_backbone "$selected_backbone"; then
    return 0
  fi

  local home_dir="${HOME:-}"
  if [[ -z "$home_dir" ]]; then
    home_dir="$(cd ~ 2>/dev/null && pwd || true)"
  fi
  local torch_home="${TORCH_HOME:-${home_dir}/.cache/torch}"
  local checkpoint_path="${torch_home}/hub/checkpoints/resnet50_ibn_a-d9d0bb7b.pth"
  local checkpoint_dir
  checkpoint_dir="$(dirname "$checkpoint_path")"
  local direct_url="https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet50_ibn_a-d9d0bb7b.pth"
  local mirror_url="https://ghfast.top/?q=https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet50_ibn_a-d9d0bb7b.pth"
  local tmp_path="${checkpoint_path}.tmp"
  local download_url
  local download_source="${IBN_DOWNLOAD_SOURCE:-direct}"

  if [[ -s "$checkpoint_path" ]]; then
    ui_info "IBN checkpoint already exists: ${checkpoint_path}"
    return 0
  fi

  ui_info "ResNet50-IBN selected. Ensuring checkpoint: ${checkpoint_path}"
  if ! mkdir -p "$checkpoint_dir"; then
    ui_error "Failed to create checkpoint directory: ${checkpoint_dir}"
    ui_tip "Please manually place checkpoint at: ${checkpoint_path}"
    return 1
  fi

  if [[ "$download_source" == "mirror" ]]; then
    download_url="$mirror_url"
  else
    download_source="direct"
    download_url="$direct_url"
  fi

  rm -f "$tmp_path"
  ui_info "Downloading IBN checkpoint via ${download_source} URL..."
  if command -v curl >/dev/null 2>&1; then
    # Show progress with curl when available.
    if curl -L --fail --progress-bar "$download_url" -o "$tmp_path"; then
      mv "$tmp_path" "$checkpoint_path"
      ui_success "IBN checkpoint downloaded from ${download_source} URL."
      return 0
    fi
  elif command -v wget >/dev/null 2>&1; then
    # Fallback with wget progress display.
    if wget -O "$tmp_path" "$download_url"; then
      mv "$tmp_path" "$checkpoint_path"
      ui_success "IBN checkpoint downloaded from ${download_source} URL."
      return 0
    fi
  else
    ui_warn "No curl/wget available. Skip auto-download."
  fi

  rm -f "$tmp_path"
  ui_error "Failed to download IBN checkpoint automatically."
  ui_tip "Please download it manually and place file at: ${checkpoint_path}"
  ui_tip "Preferred URL (${download_source}): ${download_url}"
  ui_tip "Alternative URL:"
  if [[ "$download_source" == "mirror" ]]; then
    ui_tip "${direct_url}"
  else
    ui_tip "${mirror_url}"
  fi
  return 1
}

# =========================
# Interactive configuration
# =========================
clear
ui_banner "$CURRENT_SCRIPT"
detect_platform
resolve_python_bin
ui_info "Detected platform: ${PLATFORM}"
ui_info "Using Python: ${PYTHON_BIN}"

# Backbone selection
backbone_choice=""
backbone_label=""
ui_select backbone_choice backbone_label "Select Backbone:" "1" \
  "ResNet50" \
  "ResNet50 + IBN" \
  "ResNet50 + PCB" \
  "ResNet50 + USAM" \
  "DenseNet121" \
  "HRNet" \
  "ConvNeXt" \
  "Swin" \
  "SwinV2" \
  "DINOv3" \
  "EfficientNet-B4" \
  "NAS"

case "$backbone_choice" in
  1) backbone="resnet"; backbone_flags=() ;;
  2) backbone="resnet_ibn"; backbone_flags=(--ibn) ;;
  3) backbone="resnet_pcb"; backbone_flags=(--PCB) ;;
  4) backbone="resnet_usam"; backbone_flags=(--usam) ;;
  5) backbone="densenet"; backbone_flags=(--use_dense) ;;
  6) backbone="hrnet"; backbone_flags=(--use_hr) ;;
  7) backbone="convnext"; backbone_flags=(--use_convnext) ;;
  8) backbone="swin"; backbone_flags=(--use_swin) ;;
  9) backbone="swinv2"; backbone_flags=(--use_swinv2) ;;
  10) backbone="dino"; backbone_flags=(--use_dino) ;;
  11) backbone="efficientnet"; backbone_flags=(--use_efficient) ;;
  12) backbone="nas"; backbone_flags=(--use_NAS) ;;
  *)
    ui_error "Invalid backbone number: $backbone_choice"
    exit 1
    ;;
esac

# Dataset selection
dataset_choice=""
dataset_label=""
ui_select dataset_choice dataset_label "Select Dataset:" "1" \
  "Market-1501" \
  "DukeMTMC-reID" \
  "MSMT17" \
  "CUB-200-2011" \
  "VehicleID" \
  "VeRi" \
  "VIPeR"

case "$dataset_choice" in
  1) dataset="market";    raw_data_dir="./data/Market";    prepare_script="prepare.py" ;;
  2) dataset="duke";      raw_data_dir="./data/Duke";      prepare_script="prepare_Duke.py" ;;
  3) dataset="msmt";      raw_data_dir="./data/MSMT";      prepare_script="prepare_MSMT.py" ;;
  4) dataset="cub";       raw_data_dir="./data/CUB";       prepare_script="prepare_CUB.py" ;;
  5) dataset="vehicleid"; raw_data_dir="./data/VehicleID"; prepare_script="prepare_VehicleID.py" ;;
  6) dataset="veri";      raw_data_dir="./data/VeRi";      prepare_script="prepare_VeRi.py" ;;
  7) dataset="viper";     raw_data_dir="./data/VIPeR";     prepare_script="prepare_viper.py" ;;
  *)
    ui_error "Invalid dataset number: $dataset_choice"
    exit 1
    ;;
esac

data_dir="${raw_data_dir}/pytorch"

# Loss selection
loss_choice=""
loss_label=""
ui_select loss_choice loss_label "Select Loss:" "1" \
  "CrossEntropy" \
  "Circle" \
  "Triplet" \
  "ArcFace" \
  "CosFace" \
  "Contrast" \
  "Instance" \
  "Instance-ID" \
  "Lifted" \
  "Sphere"

case "$loss_choice" in
  1) loss_name="ce"; loss_flags=() ;;
  2) loss_name="circle"; loss_flags=(--circle) ;;
  3) loss_name="triplet"; loss_flags=(--triplet) ;;
  4) loss_name="arcface"; loss_flags=(--arcface) ;;
  5) loss_name="cosface"; loss_flags=(--cosface) ;;
  6) loss_name="contrast"; loss_flags=(--contrast) ;;
  7) loss_name="instance"; loss_flags=(--instance) ;;
  8) loss_name="instance_id"; loss_flags=(--instance_id) ;;
  9) loss_name="lifted"; loss_flags=(--lifted) ;;
  10) loss_name="sphere"; loss_flags=(--sphere) ;;
  *)
    ui_error "Invalid loss number: $loss_choice"
    exit 1
    ;;
esac

default_warm_epoch="5"
ui_input warm_epoch "warm_epoch" "$default_warm_epoch" '^[0-9]+$' "" "Invalid warm_epoch: please enter a non-negative integer."

default_stride="2"
ui_input stride "stride" "$default_stride" '^[1-9][0-9]*$' "" "Invalid stride: please enter a positive integer."

erasing_p="0"
ui_input erasing_p "erasing_p" "0" '^([0-9]+([.][0-9]+)?|[.][0-9]+)$' "v >= 0 && v <= 1" "Invalid erasing_p: please enter a number in [0, 1]."
color_jitter="no"
ui_yes_no color_jitter "Enable color_jitter?" "2"

default_batchsize="32"
default_lr="0.05"

ui_input batchsize "batchsize" "$default_batchsize" '^[1-9][0-9]*$' "" "Invalid batchsize: please enter a positive integer."
ui_input lr "lr" "$default_lr" '^([0-9]+([.][0-9]+)?|[.][0-9]+)$' "v > 0" "Invalid lr: please enter a positive number."

gpu_ids="0"
ui_input gpu_ids "gpu_ids" "$gpu_ids"

ui_input which_epoch "which_epoch" "last"

run_id="$("$PYTHON_BIN" tool/run_id.py)"
lr_tag="${lr//./p}"
erasing_tag="${erasing_p//./p}"
default_run_name="${run_id}_${backbone}_${dataset}_${loss_name}_w${warm_epoch}_s${stride}_b${batchsize}_lr${lr_tag}_re${erasing_tag}"
ui_input run_name "run_name" "$default_run_name"

# Setting mirror
mirror_config_hf "$backbone"
mirror_config_ibn "$backbone"

# =========================
# Run confirmation
# =========================

ui_warn "Please confirm your configuration before starting the training workflow."
echo "================ Run configuration ================"
echo "run_id        : $run_id"
echo "backbone      : $backbone"
echo "dataset       : $dataset"
echo "loss          : $loss_name"
echo "warm_epoch    : $warm_epoch"
echo "stride        : $stride"
echo "erasing_p     : $erasing_p"
echo "color_jitter  : $color_jitter"
echo "batchsize     : $batchsize"
echo "lr            : $lr"
echo "run_name      : $run_name"
echo "hf_mirror     : $HF_MIRROR_STATUS"
echo "ibn_mirror    : $IBN_MIRROR_STATUS"
echo "==================================================="
confirm_run="no"
ui_yes_no confirm_run "Confirm and start train+evaluate workflow?" "1"
if [[ "$confirm_run" != "yes" ]]; then
  ui_warn "Canceled."
  exit 0
fi

# =========================
# Runtime setup
# =========================
download_ibn_backbone "$backbone"
train_cmd=("$PYTHON_BIN" train.py --gpu_ids "$gpu_ids" --name "$run_name" --data_dir "$data_dir" --run_id "$run_id")
train_cmd+=(--train_all)
train_cmd+=(--erasing_p "$erasing_p")
if [[ "$color_jitter" == "yes" ]]; then
  train_cmd+=(--color_jitter)
fi
train_cmd+=(--warm_epoch "$warm_epoch")
train_cmd+=(--stride "$stride")
train_cmd+=(--batchsize "$batchsize")
train_cmd+=(--lr "$lr")
train_cmd+=("${backbone_flags[@]}")
train_cmd+=("${loss_flags[@]}")
test_cmd=("$PYTHON_BIN" test.py --gpu_ids "$gpu_ids" --name "$run_name" --test_dir "$data_dir" --which_epoch "$which_epoch" --run_id "$run_id")

# =========================
# Execution
# =========================
echo "[1/4] Installing dependencies..."
"$PYTHON_BIN" -m pip install -r requirements.txt

echo "[2/4] Preparing dataset..."
prepare_dataset "$dataset" "$raw_data_dir" "$data_dir" "$prepare_script"

echo "[3/4] Training..."
"${train_cmd[@]}"

echo "[4/4] Testing..."
"${test_cmd[@]}"

ui_success "Training and evaluation completed successfully!"
echo "==================== Artifacts ===================="
echo "model dir : ./model/${run_name}"
echo "result    : ./model/${run_name}/result.txt"
echo "run_id    : ${run_id}"
echo "==================================================="
