#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Copyright:
# Script built by Wilson: https://github.com/wilson-lyc
# Co-developed with Codex (OpenAI)
# Project codebase: https://github.com/layumi/Person_reID_baseline_pytorch

# =========================
# UI / display helpers
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
}

print_project_info() {
  echo "Script by Wilson: https://github.com/wilson-lyc"
  echo "Project codebase: https://github.com/layumi/Person_reID_baseline_pytorch"
  echo
}

select_option_by_number() {
  local __outvar="$1"
  local title="$2"
  local default_choice="$3"
  shift 3
  local options=("$@")
  local choice=""
  local selected_idx=0
  local selected_text=""
  local input_prompt="Enter choice [${default_choice}]: "

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
  echo "${title} ${selected_text}"
  printf -v "$__outvar" '%s' "$choice"
}

# Prompt helper: only accept `Y` or `n` (empty input defaults to `Y`).
ask_yes_no_default_yes() {
  local prompt="$1"
  local answer
  while true; do
    read -r -p "${prompt} [Y/n]: " answer
    answer="${answer:-Y}"
    case "$answer" in
      Y|n)
        printf '%s\n' "$answer"
        return 0
        ;;
      *)
        echo "Invalid input. Please enter exactly 'Y' or 'n'."
        ;;
    esac
  done
}

dataset_display_name() {
  local ds_name="$1"
  case "$ds_name" in
    market) echo "Market-1501" ;;
    duke) echo "DukeMTMC-reID" ;;
    msmt17) echo "MSMT17" ;;
    cub) echo "CUB-200-2011" ;;
    vehicleid) echo "VehicleID" ;;
    veri) echo "VeRi" ;;
    viper) echo "VIPeR" ;;
    *) echo "$ds_name" ;;
  esac
}

backbone_display_name() {
  local bb_name="$1"
  case "$bb_name" in
    resnet50) echo "ResNet50" ;;
    resnet50_ibn) echo "ResNet50-IBN" ;;
    densenet) echo "DenseNet121" ;;
    swin) echo "Swin" ;;
    swinv2) echo "SwinV2" ;;
    dino) echo "DINOv3" ;;
    efficientnet_b4) echo "EfficientNet-B4" ;;
    nas) echo "NAS" ;;
    hrnet) echo "HRNet" ;;
    convnext) echo "ConvNeXt" ;;
    pcb) echo "PCB (ResNet50+PCB)" ;;
    resnet50_usam) echo "ResNet50-USAM" ;;
    *) echo "$bb_name" ;;
  esac
}

# =========================
# Dataset preparation
# =========================
print_manual_dataset_tutorial() {
  local ds_name="$1"
  local raw_dir="$2"
  local prepared_dir="$3"
  local ds_display
  ds_display="$(dataset_display_name "$ds_name")"

  echo
  echo "================ MANUAL DATASET PREPARATION GUIDE ================"
  echo "[Dataset] ${ds_display} (${ds_name})"
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
    msmt17)
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
configure_hf_endpoint_for_hrnet() {
  local selected_backbone="$1"
  local use_hf_mirror
  if [[ "$selected_backbone" != "hrnet" ]]; then
    return 0
  fi

  use_hf_mirror="$(ask_yes_no_default_yes "HRNet may download weights from Hugging Face. Use mirror (https://hf-mirror.com)?")"
  case "$use_hf_mirror" in
    Y)
      export HF_ENDPOINT="https://hf-mirror.com"
      echo "Using mirror endpoint: ${HF_ENDPOINT}"
      ;;
    n)
      if [[ -n "${HF_ENDPOINT:-}" ]]; then
        unset HF_ENDPOINT
      fi
      echo "Using direct Hugging Face endpoint."
      ;;
  esac
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
  local use_mirror_confirm
  local download_url
  local download_source

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

  use_mirror_confirm="$(ask_yes_no_default_yes "Use mirror URL for IBN checkpoint download?")"
  case "$use_mirror_confirm" in
    Y)
      download_url="$mirror_url"
      download_source="mirror"
      ;;
    n)
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
print_project_info

# Backbone selection
select_option_by_number backbone_choice "Select Backbone:" "1" \
  "ResNet50 (baseline, default)" \
  "ResNet50-IBN" \
  "DenseNet121" \
  "Swin" \
  "SwinV2" \
  "DINOv3" \
  "EfficientNet-B4" \
  "NAS" \
  "HRNet" \
  "ConvNeXt" \
  "PCB (ResNet50+PCB)" \
  "ResNet50-USAM"

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
backbone_display="$(backbone_display_name "$backbone")"

# Dataset selection and corresponding prepare script
select_option_by_number dataset_choice "Select Dataset:" "1" \
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
dataset_display="$(dataset_display_name "$dataset")"

data_dir="${raw_data_dir}/pytorch"

# Loss selection
select_option_by_number loss_choice "Select Loss:" "1" \
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

# =========================
# Run confirmation
# =========================
clear
print_banner
print_project_info

echo "----------------------------------------"
echo "backbone      : $backbone_display ($backbone)"
echo "dataset       : $dataset_display ($dataset)"
echo "prepare_script: $prepare_script"
echo "raw_data_dir  : $raw_data_dir"
echo "data_dir      : $data_dir"
echo "loss          : $loss_name"
echo "run_name      : $run_name"
echo "run_id        : $run_id"
echo "gpu_ids       : $gpu_ids"
echo "which_epoch   : $which_epoch"
echo "----------------------------------------"
confirm_run="$(ask_yes_no_default_yes "Confirm and start train+evaluate workflow?")"
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
# Configure network mirror for HRNet download and ensure IBN checkpoint.
configure_hf_endpoint_for_hrnet "$backbone"
ensure_ibn_checkpoint "$backbone"

# Build train/test commands from selected options.
train_cmd=(python train.py --gpu_ids "$gpu_ids" --name "$run_name" --data_dir "$data_dir" --run_id "$run_id")
train_cmd+=(--train_all)
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
