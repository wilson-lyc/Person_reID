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
██████╗ ███████╗██████╗ ███████╗ ██████╗ ███╗   ██╗    ███████╗██╗   ██╗ █████╗ ██╗
██╔══██╗██╔════╝██╔══██╗██╔════╝██╔═══██╗████╗  ██║    ██╔════╝██║   ██║██╔══██╗██║
██████╔╝█████╗  ██████╔╝███████╗██║   ██║██╔██╗ ██║    █████╗  ██║   ██║███████║██║
██╔═══╝ ██╔══╝  ██╔══██╗╚════██║██║   ██║██║╚██╗██║    ██╔══╝  ╚██╗ ██╔╝██╔══██║██║
██║     ███████╗██║  ██║███████║╚██████╔╝██║ ╚████║    ███████╗ ╚████╔╝ ██║  ██║███████╗
╚═╝     ╚══════╝╚═╝  ╚═╝╚══════╝ ╚═════╝ ╚═╝  ╚═══╝    ╚══════╝  ╚═══╝  ╚═╝  ╚═╝╚══════╝
EOF
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
  echo "If evaluation still fails, please check folder structure under:"
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
            echo "Please prepare dataset manually first, then rerun evaluate.sh."
            print_manual_dataset_tutorial "$ds_name" "$raw_dir" "$prepared_dir" "$prepare_script"
            exit 1
          fi
          ;;
        *)
          echo "Auto-download canceled by user."
          echo "Please prepare dataset manually first, then rerun evaluate.sh."
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
      echo "Invalid test dataset number: $choice"
      exit 1
      ;;
  esac
}

configure_hf_endpoint_for_model() {
  local run_name="$1"
  local config_path="./model/${run_name}/opts.yaml"
  if [[ ! -f "$config_path" ]]; then
    return 0
  fi

  if ! python - "$config_path" <<'PY'
import sys
import yaml

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)
use_hr = bool(cfg.get("use_hr", False))
sys.exit(0 if use_hr else 1)
PY
  then
    return 0
  fi

  if [[ -n "${HF_ENDPOINT:-}" ]]; then
    echo "HRNet model detected. HF_ENDPOINT is already set to: ${HF_ENDPOINT}"
    return 0
  fi

  echo "HRNet model detected. Checking direct access to Hugging Face..."
  if python - <<'PY'
import sys
import urllib.request

url = "https://huggingface.co"
try:
    with urllib.request.urlopen(url, timeout=5) as resp:
        status = getattr(resp, "status", 200)
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

clear
print_banner
echo "Evaluate script for trained models"
echo "Project codebase: https://github.com/layumi/Person_reID_baseline_pytorch"
echo

run_name=""
if [[ -d "./model" ]]; then
  mapfile -t available_runs < <(find ./model -mindepth 2 -maxdepth 2 -type f -name opts.yaml | sed 's#^\./model/\(.*\)/opts.yaml#\1#' | sort)
else
  available_runs=()
fi

if [[ "${#available_runs[@]}" -gt 0 ]]; then
  select_option_by_number run_choice "Select model to evaluate:" "1" "${available_runs[@]}"
  run_name="${available_runs[$((run_choice - 1))]}"
else
  echo "No trained model with opts.yaml found under ./model."
  read -r -p "Enter existing run name under ./model (required): " run_name
  if [[ -z "$run_name" ]]; then
    echo "run_name is required."
    exit 1
  fi
fi

config_path="./model/${run_name}/opts.yaml"
if [[ ! -f "$config_path" ]]; then
  echo "Missing training config: ${config_path}"
  echo "Please ensure this run was trained successfully."
  exit 1
fi

select_option_by_number test_dataset_choice "Select Test Dataset:" "1" \
  "Market-1501 (auto-download available)" \
  "DukeMTMC-reID (auto-download available)" \
  "MSMT17" \
  "CUB-200-2011" \
  "VehicleID" \
  "VeRi" \
  "VIPeR"

resolve_dataset_config "$test_dataset_choice"
test_dataset="$selected_dataset"
test_raw_data_dir="$selected_raw_data_dir"
test_prepare_script="$selected_prepare_script"
test_dir="${test_raw_data_dir}/pytorch"

gpu_ids="0"
read -r -p "GPU ids [0]: " input_gpu_ids
gpu_ids="${input_gpu_ids:-$gpu_ids}"

read -r -p "Which epoch for test [last]: " which_epoch
which_epoch="${which_epoch:-last}"

batchsize="256"
read -r -p "Batch size [256]: " input_batchsize
batchsize="${input_batchsize:-$batchsize}"

ms="1"
read -r -p "Multi-scale (ms) [1]: " input_ms
ms="${input_ms:-$ms}"

read -r -p "Use multi-query? [y/N]: " multi_query_confirm
read -r -p "Skip evaluate_gpu.py and only extract features? [y/N]: " skip_eval_confirm

run_id="$(python tool/run_id.py)"

test_cmd=(python test.py --gpu_ids "$gpu_ids" --name "$run_name" --test_dir "$test_dir" --which_epoch "$which_epoch" --batchsize "$batchsize" --ms "$ms" --run_id "$run_id")
case "$multi_query_confirm" in
  Y|y|yes|YES) test_cmd+=(--multi) ;;
esac
case "$skip_eval_confirm" in
  Y|y|yes|YES) test_cmd+=(--skip_eval) ;;
esac

clear
print_banner
echo "Evaluate script for trained models"
echo
echo "----------------------------------------"
echo "run_name      : $run_name"
echo "config_path   : $config_path"
echo "test_dataset  : $test_dataset"
echo "test_prepare  : $test_prepare_script"
echo "test_raw_dir  : $test_raw_data_dir"
echo "test_dir      : $test_dir"
echo "gpu_ids       : $gpu_ids"
echo "which_epoch   : $which_epoch"
echo "batchsize     : $batchsize"
echo "ms            : $ms"
echo "multi_query   : $multi_query_confirm"
echo "skip_eval     : $skip_eval_confirm"
echo "run_id        : $run_id"
echo "----------------------------------------"
read -r -p "Confirm and start evaluation? [Y/n]: " confirm_run
confirm_run="${confirm_run:-Y}"
case "$confirm_run" in
  Y|y|yes|YES)
    ;;
  *)
    echo "Canceled."
    exit 0
    ;;
esac

configure_hf_endpoint_for_model "$run_name"

echo "[1/2] Preparing test dataset..."
ensure_dataset_ready "$test_dataset" "$test_raw_data_dir" "$test_dir" "$test_prepare_script"

echo "[2/2] Evaluating..."
"${test_cmd[@]}"

echo "Done. Artifacts:"
echo "  model dir : ./model/${run_name}"
echo "  result    : ./model/${run_name}/result.txt"
echo "  run_id    : ${run_id}"
