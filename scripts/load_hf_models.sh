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

if [[ ! -f "load_hf_models.py" ]]; then
  ui_error "Missing file: load_hf_models.py"
  exit 1
fi

clear
ui_banner "$CURRENT_SCRIPT"
ui_info "Select backbone(s) to preload from Hugging Face."

backbone_choice=""
backbone_label=""
ui_select backbone_choice backbone_label "Select backbone to preload from Hugging Face:" "1" \
  "Swin" \
  "SwinV2" \
  "DINOv3" \
  "ConvNeXt" \
  "HRNet" \
  "All HF backbones above"

case "$backbone_choice" in
  1) backbone_flags=(--use_swin) ;;
  2) backbone_flags=(--use_swinv2) ;;
  3) backbone_flags=(--use_dino) ;;
  4) backbone_flags=(--use_convnext) ;;
  5) backbone_flags=(--use_hr) ;;
  6) backbone_flags=(--use_swin --use_swinv2 --use_dino --use_convnext --use_hr) ;;
  *)
    ui_error "Invalid backbone number: $backbone_choice"
    exit 1
    ;;
esac

use_hf_mirror="no"
use_hf_mirror_idx=""
ui_select use_hf_mirror_idx use_hf_mirror "Use Hugging Face mirror (https://hf-mirror.com)?" "1" "yes" "no"

cmd=(python load_hf_models.py "${backbone_flags[@]}")
if [[ "$use_hf_mirror" == "yes" ]]; then
  export HF_ENDPOINT="https://hf-mirror.com"
  mirror_status="yes"
else
  unset HF_ENDPOINT || true
  mirror_status="no"
fi

ui_warn "Please confirm your configuration before preloading."
echo "================ Run configuration ================"
echo "backbone      : ${backbone_label}"
echo "hf_mirror     : ${mirror_status}"
echo "==================================================="

confirm_run="no"
confirm_run_idx=""
ui_select confirm_run_idx confirm_run "Confirm and start model preload?" "1" "yes" "no"
if [[ "$confirm_run" != "yes" ]]; then
  ui_warn "Canceled."
  exit 0
fi

if [[ "$mirror_status" == "yes" ]]; then
  ui_info "Pulling model weights via mirror: ${HF_ENDPOINT}"
else
  ui_info "Pulling model weights from Hugging Face."
fi

"${cmd[@]}"
ui_success "Model preload completed successfully."
