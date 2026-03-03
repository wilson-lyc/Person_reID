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

extract_data_dir_from_opts() {
  local opts_file="$1"
  local parsed=""
  if [[ ! -f "$opts_file" ]]; then
    return 1
  fi

  if parsed="$(python - "$opts_file" <<'PY'
import sys
import yaml

opts_file = sys.argv[1]
try:
    with open(opts_file, "r", encoding="utf-8") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    data_dir = cfg.get("data_dir", "")
    if isinstance(data_dir, str):
        print(data_dir.strip())
except Exception:
    pass
PY
)"; then
    if [[ -n "$parsed" ]]; then
      printf '%s\n' "$parsed"
      return 0
    fi
  fi
  return 1
}

list_available_epochs() {
  local model_path="$1"
  local epochs=()
  local f=""
  for f in "${model_path}"/net_*.pth; do
    [[ -e "$f" ]] || continue
    local bn
    bn="$(basename "$f")"
    if [[ "$bn" =~ ^net_([0-9]+)\.pth$ ]]; then
      epochs+=("${BASH_REMATCH[1]}")
    fi
  done
  if [[ ${#epochs[@]} -eq 0 ]]; then
    return 1
  fi
  printf '%s\n' "${epochs[@]}" | sort -u
}

validate_epoch_exists() {
  local model_path="$1"
  local epoch="$2"
  if [[ "$epoch" == "last" ]]; then
    [[ -f "${model_path}/net_last.pth" ]]
    return
  fi
  [[ "$epoch" =~ ^[0-9]+$ ]] || return 1
  local padded
  padded="$(printf "%03d" "$epoch")"
  [[ -f "${model_path}/net_${padded}.pth" || -f "${model_path}/net_${epoch}.pth" ]]
}

validate_test_dir() {
  local test_dir="$1"
  [[ -d "$test_dir/query" && -d "$test_dir/gallery" ]]
}

infer_dataset_tag() {
  local test_dir="$1"
  local normalized
  normalized="$(printf '%s' "$test_dir" | tr '[:upper:]' '[:lower:]' | tr '\\' '/')"
  case "$normalized" in
    */market/*|*market/pytorch) printf 'market\n' ;;
    */duke/*|*duke/pytorch) printf 'duke\n' ;;
    */msmt/*|*msmt/pytorch) printf 'msmt\n' ;;
    */cub/*|*cub/pytorch) printf 'cub\n' ;;
    */vehicleid/*|*vehicleid/pytorch) printf 'vehicleid\n' ;;
    */veri/*|*veri/pytorch) printf 'veri\n' ;;
    */viper/*|*viper/pytorch) printf 'viper\n' ;;
    *) printf 'unknown\n' ;;
  esac
}

build_model_candidates() {
  local model_root="./model"
  local model_dir=""
  local model_name=""
  local opts_file=""
  local has_weight=0

  MODEL_NAMES=()
  MODEL_DESCS=()

  if [[ ! -d "$model_root" ]]; then
    return 1
  fi

  for model_dir in "$model_root"/*; do
    [[ -d "$model_dir" ]] || continue
    model_name="$(basename "$model_dir")"
    opts_file="${model_dir}/opts.yaml"
    has_weight=0

    [[ -f "$opts_file" ]] || continue
    if [[ -f "${model_dir}/net_last.pth" ]]; then
      has_weight=1
    elif compgen -G "${model_dir}/net_*.pth" > /dev/null; then
      has_weight=1
    fi

    [[ "$has_weight" -eq 1 ]] || continue

    MODEL_NAMES+=("$model_name")
    MODEL_DESCS+=("${model_name}")
  done

  [[ ${#MODEL_NAMES[@]} -gt 0 ]]
}

clear
ui_banner "$CURRENT_SCRIPT"

ui_info "Scanning trained models..."
MODEL_NAMES=()
MODEL_DESCS=()
if ! build_model_candidates; then
  ui_error "No trained model found. Please run train.sh first."
  exit 1
fi

default_model_idx="${#MODEL_NAMES[@]}"
model_choice=""
model_label=""
ui_select model_choice model_label "Select the model to evaluate:" "$default_model_idx" "${MODEL_DESCS[@]}"
selected_index=$((model_choice - 1))
name="${MODEL_NAMES[$selected_index]}"
model_path="./model/${name}"

opts_file="${model_path}/opts.yaml"
default_test_dir="./data/Market/pytorch"
default_dataset_choice="1"
if parsed_test_dir="$(extract_data_dir_from_opts "$opts_file")"; then
  default_test_dir="$parsed_test_dir"
fi

case "$default_test_dir" in
  "./data/Market/pytorch") default_dataset_choice="1" ;;
  "./data/Duke/pytorch") default_dataset_choice="2" ;;
  "./data/MSMT/pytorch") default_dataset_choice="3" ;;
  "./data/CUB/pytorch") default_dataset_choice="4" ;;
  "./data/VehicleID/pytorch") default_dataset_choice="5" ;;
  "./data/VeRi/pytorch") default_dataset_choice="6" ;;
  "./data/VIPeR/pytorch") default_dataset_choice="7" ;;
  *) default_dataset_choice="1" ;;
esac

dataset_choice=""
dataset_label=""
ui_select dataset_choice dataset_label "Select Evaluation Dataset:" "$default_dataset_choice" \
  "Market-1501 (default)" \
  "DukeMTMC-reID" \
  "MSMT17" \
  "CUB-200-2011" \
  "VehicleID" \
  "VeRi" \
  "VIPeR"

case "$dataset_choice" in
  1) test_dir="./data/Market/pytorch"; eval_dataset_name="Market-1501" ;;
  2) test_dir="./data/Duke/pytorch"; eval_dataset_name="DukeMTMC-reID" ;;
  3) test_dir="./data/MSMT/pytorch"; eval_dataset_name="MSMT17" ;;
  4) test_dir="./data/CUB/pytorch"; eval_dataset_name="CUB-200-2011" ;;
  5) test_dir="./data/VehicleID/pytorch"; eval_dataset_name="VehicleID" ;;
  6) test_dir="./data/VeRi/pytorch"; eval_dataset_name="VeRi" ;;
  7) test_dir="./data/VIPeR/pytorch"; eval_dataset_name="VIPeR" ;;
  *)
    ui_error "Invalid dataset number: $dataset_choice"
    exit 1
    ;;
esac

if ! validate_test_dir "$test_dir"; then
  ui_error "Invalid test_dir: ${test_dir}"
  ui_tip "Required folders are missing: ${test_dir}/query and ${test_dir}/gallery"
  exit 1
fi

ui_input gpu_ids "gpu_ids" "0"

default_epoch="last"
if [[ ! -f "${model_path}/net_last.pth" ]]; then
  if mapfile -t epochs < <(list_available_epochs "$model_path"); then
    default_epoch="${epochs[${#epochs[@]}-1]}"
    ui_warn "net_last.pth not found. Using latest numeric epoch: ${default_epoch}"
  else
    ui_error "No available model checkpoint found in ${model_path}."
    exit 1
  fi
fi
ui_input which_epoch "which_epoch" "$default_epoch" '^(last|[0-9]+)$' "" "Invalid which_epoch: expected 'last' or a non-negative integer."

if ! validate_epoch_exists "$model_path" "$which_epoch"; then
  ui_error "Checkpoint for which_epoch='${which_epoch}' not found under ${model_path}."
  echo "Available checkpoints:"
  if [[ -f "${model_path}/net_last.pth" ]]; then
    echo "  - last"
  fi
  while IFS= read -r ep; do
    echo "  - ${ep}"
  done < <(list_available_epochs "$model_path" || true)
  exit 1
fi

resolved_which_epoch="$which_epoch"
if [[ "$which_epoch" != "last" ]]; then
  epoch_padded="$(printf "%03d" "$which_epoch")"
  if [[ -f "${model_path}/net_${epoch_padded}.pth" && ! -f "${model_path}/net_${which_epoch}.pth" ]]; then
    resolved_which_epoch="$epoch_padded"
  fi
fi

eval_mode_choice=""
eval_mode_label=""
ui_select eval_mode_choice eval_mode_label "Select Evaluation Mode:" "1" \
  "Normal evaluation (evaluate_gpu.py)" \
  "Re-ranking evaluation (evaluate_rerank.py)"

case "$eval_mode_choice" in
  1) eval_mode="normal"; eval_script="evaluate_gpu.py" ;;
  2) eval_mode="rerank"; eval_script="evaluate_rerank.py" ;;
  *)
    ui_error "Invalid evaluation mode: ${eval_mode_choice}"
    exit 1
    ;;
esac

run_id="$(python tool/run_id.py)"

ui_warn "Please confirm your configuration before starting evaluation."
echo "================ Run configuration ================"
echo "model_name     : ${name}"
echo "eval_dataset   : ${eval_dataset_name}"
echo "gpu_ids        : ${gpu_ids}"
echo "which_epoch    : ${resolved_which_epoch}"
echo "eval_mode      : ${eval_mode}"
echo "run_id         : ${run_id}"
echo "==================================================="

confirm_run="no"
confirm_run_idx=""
ui_select confirm_run_idx confirm_run "Confirm and start test+evaluation workflow?" "1" "yes" "no"
if [[ "$confirm_run" != "yes" ]]; then
  ui_warn "Canceled."
  exit 0
fi

dataset_tag="$(infer_dataset_tag "$test_dir")"
result_mat="${model_path}/pytorch_result_${dataset_tag}.mat"
multi_mat="${model_path}/multi_query_${dataset_tag}.mat"

echo "[1/3] Testing..."
python test.py \
  --gpu_ids "$gpu_ids" \
  --name "$name" \
  --test_dir "$test_dir" \
  --which_epoch "$resolved_which_epoch" \
  --run_id "$run_id" \
  --skip_eval

if [[ ! -f "$result_mat" ]]; then
  ui_error "test.py finished but expected result mat was not generated: ${result_mat}"
  exit 1
fi

echo "[2/3] Evaluating..."
if [[ "$eval_mode" == "normal" ]]; then
  eval_cmd=(python "$eval_script" --result_mat "$result_mat" --run_id "$run_id")
  if [[ -f "$multi_mat" ]]; then
    eval_cmd+=(--multi_mat "$multi_mat")
  fi
else
  eval_cmd=(python "$eval_script" --result_mat "$result_mat")
fi
"${eval_cmd[@]}"

ui_success "Evaluation completed successfully."
echo "==================== Artifacts ===================="
echo "result    : ${model_path}/result.txt"
echo "result mat: ${result_mat}"
echo "model dir : ${model_path}"
echo "run_id    : ${run_id}"
echo "==================================================="
