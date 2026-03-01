#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

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

confirmer() {
  local prompt="$1"
  local default_choice="$2"
  local answer
  local prompt_suffix=""
  local default_answer=""

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
        printf '%s\n' "$answer"
        return 0
        ;;
      *)
        echo "Invalid input. Please enter exactly 'Y' or 'n' (or press Enter for default)."
        ;;
    esac
  done
}

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
print_banner

echo "Scanning trained models under ./model ..."
MODEL_NAMES=()
MODEL_DESCS=()
if ! build_model_candidates; then
  echo "No valid trained models found in ./model."
  echo "A valid model directory must contain opts.yaml and net_last.pth (or at least one net_*.pth)."
  echo "Please train first, then rerun evaluate.sh."
  exit 1
fi

default_model_idx="${#MODEL_NAMES[@]}"
select_menu model_choice "Select Model:" "$default_model_idx" "${MODEL_DESCS[@]}"
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

select_menu dataset_choice "Select Evaluation Dataset:" "$default_dataset_choice" \
  "Market-1501 (default)" \
  "DukeMTMC-reID" \
  "MSMT17" \
  "CUB-200-2011" \
  "VehicleID" \
  "VeRi" \
  "VIPeR"

case "$dataset_choice" in
  1) test_dir="./data/Market/pytorch" ;;
  2) test_dir="./data/Duke/pytorch" ;;
  3) test_dir="./data/MSMT/pytorch" ;;
  4) test_dir="./data/CUB/pytorch" ;;
  5) test_dir="./data/VehicleID/pytorch" ;;
  6) test_dir="./data/VeRi/pytorch" ;;
  7) test_dir="./data/VIPeR/pytorch" ;;
  *)
    echo "Invalid dataset number: $dataset_choice"
    exit 1
    ;;
esac

if ! validate_test_dir "$test_dir"; then
  echo "Invalid test_dir: ${test_dir}"
  echo "Required folders are missing: ${test_dir}/query and ${test_dir}/gallery"
  exit 1
fi

inputer gpu_ids "gpu_ids" "0"

default_epoch="last"
if [[ ! -f "${model_path}/net_last.pth" ]]; then
  if mapfile -t epochs < <(list_available_epochs "$model_path"); then
    default_epoch="${epochs[${#epochs[@]}-1]}"
    echo "net_last.pth not found. Defaulting which_epoch to latest numeric epoch: ${default_epoch}"
  else
    echo "No available model checkpoint found in ${model_path}."
    exit 1
  fi
fi
inputer which_epoch "which_epoch" "$default_epoch" '^(last|[0-9]+)$' "" "Invalid which_epoch: expected 'last' or a non-negative integer."

if ! validate_epoch_exists "$model_path" "$which_epoch"; then
  echo "Checkpoint for which_epoch='${which_epoch}' not found under ${model_path}."
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

select_menu eval_mode_choice "Select Evaluation Mode:" "1" \
  "Normal evaluation (evaluate_gpu.py)" \
  "Re-ranking evaluation (evaluate_rerank.py)"

case "$eval_mode_choice" in
  1) eval_mode="normal"; eval_script="evaluate_gpu.py" ;;
  2) eval_mode="rerank"; eval_script="evaluate_rerank.py" ;;
  *)
    echo "Invalid evaluation mode: ${eval_mode_choice}"
    exit 1
    ;;
esac

echo "----------------------------------------"
echo "model_name     : ${name}"
echo "eval_mode      : ${eval_mode}"
echo "----------------------------------------"

confirm_run="$(confirmer "Confirm and start test+evaluation workflow?" "yes")"
case "$confirm_run" in
  Y) ;;
  n)
    echo "Canceled."
    exit 0
    ;;
esac

result_file="${model_path}/result.txt"
timestamp="$(date '+%Y-%m-%d %H:%M:%S')"

echo "[$timestamp] ==== evaluate.sh run start ====" >> "$result_file"
echo "[$timestamp] model=${name}, which_epoch=${resolved_which_epoch}, test_dir=${test_dir}, gpu_ids=${gpu_ids}, mode=${eval_mode}" >> "$result_file"

echo "[1/2] Running test.py (feature extraction only)..."
python test.py \
  --gpu_ids "$gpu_ids" \
  --name "$name" \
  --test_dir "$test_dir" \
  --which_epoch "$resolved_which_epoch" \
  --skip_eval

if [[ ! -f "pytorch_result.mat" ]]; then
  echo "test.py finished but pytorch_result.mat was not generated."
  exit 1
fi

echo "[2/2] Running ${eval_script}..."
python "$eval_script" | tee -a "$result_file"

end_timestamp="$(date '+%Y-%m-%d %H:%M:%S')"
echo "[$end_timestamp] ==== evaluate.sh run end ====" >> "$result_file"
echo
echo "Done."
echo "Result file     : ${result_file}"
echo "Feature file    : ./pytorch_result.mat"
