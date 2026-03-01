#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Copyright:
# Script built by Wilson: https://github.com/wilson-lyc
# Co-developed with Codex (OpenAI)
# Project codebase: https://github.com/layumi/Person_reID_baseline_pytorch

# =========================
# UI Tools
# =========================
print_banner() {
cat <<'EOF'
██████╗ ███████╗██████╗ ███████╗ ██████╗ ███╗   ██╗    ██████╗ ███████╗██╗██████╗
██╔══██╗██╔════╝██╔══██╗██╔════╝██╔═══██╗████╗  ██║    ██╔══██╗██╔════╝██║██╔══██╗
██████╔╝█████╗  ██████╔╝███████╗██║   ██║██╔██╗ ██║    ██████╔╝█████╗  ██║██║  ██║
██╔═══╝ ██╔══╝  ██╔══██╗╚════██║██║   ██║██║╚██╗██║    ██╔══██╗██╔══╝  ██║██║  ██║
██║     ███████╗██║  ██║███████║╚██████╔╝██║ ╚████║    ██║  ██║███████╗██║██████╔╝
╚═╝     ╚══════╝╚═╝  ╚═╝╚══════╝ ╚═════╝ ╚═╝  ╚═══╝    ╚═╝  ╚═╝╚══════╝╚═╝╚═════╝
EOF
  local script_name
  script_name="$(basename "$0")"
  echo "============================================================"
  echo "Designed by Wilson | Implemented by Codex"
  echo "Script: ${script_name}"
  echo "============================================================"
}

# Multi-option select menu
select_menu() {
  local __outvar="$1"
  local title="$2"
  local default_choice="$3"
  shift 3
  local options=("$@")
  local choice=""
  local selected_idx=0
  local selected_text=""
  local input_prompt="Enter choice [${default_choice}]: "
  local color_selected=""
  local color_reset=""

  if [[ -t 1 ]]; then
    color_selected="\033[1;36m"
    color_reset="\033[0m"
  fi

  echo "$title"
  for i in "${!options[@]}"; do
    printf "  %d) %s\n" "$((i + 1))" "${options[$i]}"
  done

  while true; do
    read -r -p "${input_prompt}" choice
    choice="${choice:-$default_choice}"
    if [[ "$choice" =~ ^[0-9]+$ ]] && (( choice >= 1 && choice <= ${#options[@]} )); then
      selected_idx=$((choice - 1))
      selected_text="${options[$selected_idx]}"
      break
    fi
    if [[ -t 1 ]]; then
      printf "\033[1A\r\033[2K"
    fi
    input_prompt="Invalid choice (${choice}). Enter choice [${default_choice}]: "
  done

  if [[ -t 1 ]]; then
    printf "\033[%dA" "$(( ${#options[@]} + 2 ))"
    printf "\033[J"
  fi
  printf "%s %b%s%b\n" "$title" "$color_selected" "$selected_text" "$color_reset"
  printf -v "$__outvar" '%s' "$choice"
}

# Yes/No confirmer
confirmer() {
  local prompt="$1"
  local default_choice="$2"
  local answer
  local prompt_suffix=""
  local default_answer=""
  local color_selected=""
  local color_reset=""
  local shown_answer=""

  if [[ -t 2 ]]; then
    color_selected="\033[1;36m"
    color_reset="\033[0m"
  fi

  case "$default_choice" in
    yes)
      prompt_suffix="[Y/n]"
      default_answer="Y"
      ;;
    no)
      prompt_suffix="[y/N]"
      default_answer="n"
      ;;
    *)
      echo "Invalid default option for confirmer: ${default_choice} (expected yes or no)."
      return 1
      ;;
  esac

  while true; do
    read -r -p "${prompt} ${prompt_suffix}: " answer
    answer="${answer:-$default_answer}"
    case "$answer" in
      Y|n)
        if [[ -t 2 ]]; then
          if [[ "$answer" == "Y" ]]; then
            shown_answer="Y"
          else
            shown_answer="N"
          fi
          printf "\033[1A\r\033[2K" >&2
          printf "%s: %b%s%b\n" "$prompt" "$color_selected" "$shown_answer" "$color_reset" >&2
        fi
        printf '%s\n' "$answer"
        return 0
        ;;
      *)
        echo "Invalid input. Please enter exactly 'Y' or 'n' (or press Enter for default)."
        ;;
    esac
  done
}

# Parameter inputer
inputer() {
  local __outvar="$1"
  local label="$2"
  local default_value="$3"
  local regex="${4:-}"
  local range_expr="${5:-}"
  local err_msg="${6:-Invalid input.}"
  local value=""
  local ok=1
  local color_selected=""
  local color_reset=""

  if [[ -t 1 ]]; then
    color_selected="\033[1;36m"
    color_reset="\033[0m"
  fi

  while true; do
    read -r -p "${label} [${default_value}]: " value
    value="${value:-$default_value}"
    ok=1

    if [[ -n "$regex" ]] && [[ ! "$value" =~ $regex ]]; then
      ok=0
    fi
    if [[ $ok -eq 1 && -n "$range_expr" ]] && ! awk -v v="$value" "BEGIN {exit !($range_expr)}"; then
      ok=0
    fi

    if [[ $ok -eq 1 ]]; then
      break
    fi
    echo "$err_msg"
  done

  if [[ -t 1 ]]; then
    printf "\033[1A\r\033[2K"
    printf "%s: %b%s%b\n" "$label" "$color_selected" "$value" "$color_reset"
  else
    echo "${label}: ${value}"
  fi

  printf -v "$__outvar" '%s' "$value"
}

# Mirror config status shown in run confirmation.
HF_MIRROR_STATUS="N/A"
IBN_MIRROR_STATUS="N/A"
IBN_DOWNLOAD_SOURCE="direct"

# =========================
# Dataset preparation
# =========================
print_manual_dataset_tutorial() {
  local ds_name="$1"
  local raw_dir="$2"
  local prepared_dir="$3"

  echo
  echo "================ MANUAL DATASET PREPARATION GUIDE ================"
  echo "[Dataset] ${ds_name}"
  echo "Raw path should be: ${raw_dir}"
  echo "Please prepare the raw dataset folders/files under the path above."
  echo "Expected prepared path:"
  echo "  ${prepared_dir}"
  echo "After raw data is ready, rerun train.sh."
  echo "=================================================================="
  echo
}

dataset_has_required_structure() {
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

ensure_dataset_ready() {
  local ds_name="$1"
  local raw_dir="$2"
  local prepared_dir="$3"
  local prepare_script="$4"

  if [[ ! -f "$prepare_script" ]]; then
    echo "Prepare script not found: ${prepare_script}"
    echo "Please check your project files, then rerun train.sh."
    return 1
  fi

  if ! dataset_has_required_structure "$ds_name" "$raw_dir"; then
    echo "Dataset raw path not found or incomplete: ${raw_dir}"
    echo "Please prepare this dataset manually according to its required raw folder/files."
    print_manual_dataset_tutorial "$ds_name" "$raw_dir" "$prepared_dir"
    echo "After preparing the dataset, rerun train.sh."
    return 1
  fi

  echo "Preparing dataset..."
  if ! python "$prepare_script" --path "$raw_dir"; then
    echo "Prepare failed."
    print_manual_dataset_tutorial "$ds_name" "$raw_dir" "$prepared_dir"
    return 1
  fi

  if [[ ! -d "$prepared_dir" ]]; then
    echo "Prepared path not found after prepare: ${prepared_dir}"
    print_manual_dataset_tutorial "$ds_name" "$raw_dir" "$prepared_dir"
    return 1
  fi

  return 0
}

# =========================
# Network / model source config
# =========================
mirror_config_hf() {
  local selected_backbone="$1"
  local backbone_tip
  local use_hf_mirror

  case "$selected_backbone" in
    hrnet)
      backbone_tip="HRNet"
      ;;
    convnext)
      backbone_tip="ConvNeXt"
      ;;
    swin)
      backbone_tip="Swin"
      ;;
    swinv2)
      backbone_tip="SwinV2"
      ;;
    *)
      HF_MIRROR_STATUS="N/A"
      return 0
      ;;
  esac

  if [[ -n "${HF_ENDPOINT:-}" ]]; then
    echo "Current HF_ENDPOINT: ${HF_ENDPOINT}"
  fi

  use_hf_mirror="$(confirmer "${backbone_tip} may download weights from Hugging Face. Use mirror (https://hf-mirror.com)?" "yes")"
  case "$use_hf_mirror" in
    Y)
      export HF_ENDPOINT="https://hf-mirror.com"
      HF_MIRROR_STATUS="enabled"
      echo "Using mirror endpoint: ${HF_ENDPOINT}"
      ;;
    n)
      if [[ -n "${HF_ENDPOINT:-}" ]]; then
        unset HF_ENDPOINT
      fi
      HF_MIRROR_STATUS="disabled"
      echo "Using direct Hugging Face endpoint."
      ;;
  esac
}

is_ibn_backbone() {
  local selected_backbone="$1"
  [[ "$selected_backbone" == "resnet50_ibn" || "$selected_backbone" == "resnet_ibn" ]]
}

mirror_config_ibn() {
  local selected_backbone="$1"
  local use_mirror_confirm

  if ! is_ibn_backbone "$selected_backbone"; then
    IBN_MIRROR_STATUS="N/A"
    IBN_DOWNLOAD_SOURCE="direct"
    return 0
  fi

  use_mirror_confirm="$(confirmer "Use mirror URL for IBN checkpoint download?" "yes")"
  case "$use_mirror_confirm" in
    Y)
      IBN_DOWNLOAD_SOURCE="mirror"
      IBN_MIRROR_STATUS="enabled"
      ;;
    n)
      IBN_DOWNLOAD_SOURCE="direct"
      IBN_MIRROR_STATUS="disabled"
      ;;
  esac
}

ensure_ibn_checkpoint() {
  local selected_backbone="$1"
  if ! is_ibn_backbone "$selected_backbone"; then
    return 0
  fi

  local checkpoint_path="/root/.cache/torch/hub/checkpoints/resnet50_ibn_a-d9d0bb7b.pth"
  local checkpoint_dir
  checkpoint_dir="$(dirname "$checkpoint_path")"
  local direct_url="https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet50_ibn_a-d9d0bb7b.pth"
  local mirror_url="https://ghfast.top/?q=https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet50_ibn_a-d9d0bb7b.pth"
  local tmp_path="${checkpoint_path}.tmp"
  local download_url
  local download_source="${IBN_DOWNLOAD_SOURCE:-direct}"

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

  if [[ "$download_source" == "mirror" ]]; then
    download_url="$mirror_url"
  else
    download_source="direct"
    download_url="$direct_url"
  fi

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
    if wget -O "$tmp_path" "$download_url"; then
      mv "$tmp_path" "$checkpoint_path"
      echo "IBN checkpoint downloaded from ${download_source} URL."
      return 0
    fi
  else
    echo "No curl/wget available. Skip auto-download."
  fi

  rm -f "$tmp_path"
  echo "Failed to download IBN checkpoint automatically."
  echo "Please download it manually and place file at:"
  echo "  ${checkpoint_path}"
  echo "Preferred URL (${download_source}):"
  echo "  ${download_url}"
  echo "Alternative URL:"
  if [[ "$download_source" == "mirror" ]]; then
    echo "  ${direct_url}"
  else
    echo "  ${mirror_url}"
  fi
  return 1
}

# =========================
# Interactive configuration
# =========================
clear
print_banner

# Backbone selection
select_menu backbone_choice "Select Backbone:" "1" \
  "ResNet50 (baseline)" \
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
  "NAS" \
  
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
    echo "Invalid backbone number: $backbone_choice"
    exit 1
    ;;
esac

# Dataset selection
select_menu dataset_choice "Select Dataset:" "1" \
  "Market-1501 (default)" \
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
    echo "Invalid dataset number: $dataset_choice"
    exit 1
    ;;
esac

data_dir="${raw_data_dir}/pytorch"

# Loss selection
select_menu loss_choice "Select Loss:" "1" \
  "CrossEntropy (default)" \
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
    echo "Invalid loss number: $loss_choice"
    exit 1
    ;;
esac

default_warm_epoch="5"
inputer warm_epoch "warm_epoch" "$default_warm_epoch" '^[0-9]+$' "" "Invalid warm_epoch: please enter a non-negative integer."

default_stride="2"
inputer stride "stride" "$default_stride" '^[1-9][0-9]*$' "" "Invalid stride: please enter a positive integer."

erasing_p="0"
inputer erasing_p "erasing_p" "0" '^([0-9]+([.][0-9]+)?|[.][0-9]+)$' "v >= 0 && v <= 1" "Invalid erasing_p: please enter a number in [0, 1]."
color_jitter="$(confirmer "Enable color_jitter?" "no")"

default_batchsize="32"
default_lr="0.05"

inputer batchsize "batchsize" "$default_batchsize" '^[1-9][0-9]*$' "" "Invalid batchsize: please enter a positive integer."
inputer lr "lr" "$default_lr" '^([0-9]+([.][0-9]+)?|[.][0-9]+)$' "v > 0" "Invalid lr: please enter a positive number."

gpu_ids="0"
inputer gpu_ids "gpu_ids" "$gpu_ids"

inputer which_epoch "which_epoch" "last"

run_id="$(python tool/run_id.py)"
lr_tag="${lr//./p}"
erasing_tag="${erasing_p//./p}"
default_run_name="${run_id}_${backbone}_${dataset}_${loss_name}_w${warm_epoch}_s${stride}_b${batchsize}_lr${lr_tag}_re${erasing_tag}"
inputer run_name "run_name" "$default_run_name"

# Configure mirror options before run confirmation.
mirror_config_hf "$backbone"
mirror_config_ibn "$backbone"

# =========================
# Run confirmation
# =========================

echo "----------------------------------------"
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
echo "----------------------------------------"
confirm_run="$(confirmer "Confirm and start train+evaluate workflow?" "yes")"
case "$confirm_run" in
  Y)
    ;;
  n)
    echo "Canceled."
    exit 0
    ;;
esac

# =========================
# Runtime setup and execution
# =========================
ensure_ibn_checkpoint "$backbone"

# Build train/test commands from selected options.
train_cmd=(python train.py --gpu_ids "$gpu_ids" --name "$run_name" --data_dir "$data_dir" --run_id "$run_id")
train_cmd+=(--train_all)
train_cmd+=(--erasing_p "$erasing_p")
if [[ "$color_jitter" == "Y" ]]; then
  train_cmd+=(--color_jitter)
fi
train_cmd+=(--warm_epoch "$warm_epoch")
train_cmd+=(--stride "$stride")
train_cmd+=(--batchsize "$batchsize")
train_cmd+=(--lr "$lr")
train_cmd+=("${backbone_flags[@]}")
train_cmd+=("${loss_flags[@]}")

test_cmd=(python test.py --gpu_ids "$gpu_ids" --name "$run_name" --test_dir "$data_dir" --which_epoch "$which_epoch" --run_id "$run_id")

echo "[1/4] Installing dependencies from requirements.txt..."
python -m pip install -r requirements.txt

# Validate raw dataset then run dataset-specific prepare*.py.
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
