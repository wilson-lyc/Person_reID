#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

COMMON_LIB="${SCRIPT_DIR}/common.sh"
if [[ ! -f "$COMMON_LIB" ]]; then
  echo "Missing shared shell library: ${COMMON_LIB}"
  exit 1
fi
source "$COMMON_LIB"

print_banner() {
cat <<'EOF'
██████╗ ██████╗ ███████╗██████╗  █████╗ ██████╗ ███████╗    ███╗   ███╗ ██████╗ ██████╗ ███████╗██╗
██╔══██╗██╔══██╗██╔════╝██╔══██╗██╔══██╗██╔══██╗██╔════╝    ████╗ ████║██╔═══██╗██╔══██╗██╔════╝██║
██████╔╝██████╔╝█████╗  ██████╔╝███████║██████╔╝█████╗      ██╔████╔██║██║   ██║██║  ██║█████╗  ██║
██╔═══╝ ██╔══██╗██╔══╝  ██╔═══╝ ██╔══██║██╔══██╗██╔══╝      ██║╚██╔╝██║██║   ██║██║  ██║██╔══╝  ██║
██║     ██║  ██║███████╗██║     ██║  ██║██║  ██║███████╗    ██║ ╚═╝ ██║╚██████╔╝██████╔╝███████╗███████╗
╚═╝     ╚═╝  ╚═╝╚══════╝╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝    ╚═╝     ╚═╝ ╚═════╝ ╚═════╝ ╚══════╝╚══════╝
EOF
  local script_name
  script_name="$(basename "$0")"
  echo "============================================================"
  echo "Designed by Wilson | Implemented by Codex"
  echo "Script: ${script_name}"
  echo "============================================================"
}

if [[ ! -f "load_hf_models.py" ]]; then
  echo "Missing file: load_hf_models.py"
  exit 1
fi

clear
print_banner
echo

select_menu backbone_choice "Select backbone to preload from Hugging Face:" "1" \
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
    echo "Invalid backbone number: $backbone_choice"
    exit 1
    ;;
esac

use_hf_mirror="$(confirmer "Use Hugging Face mirror (https://hf-mirror.com)?" "y")"

cmd=(python load_hf_models.py "${backbone_flags[@]}")
if [[ "$use_hf_mirror" == "y" ]]; then
  export HF_ENDPOINT="https://hf-mirror.com"
  echo "Pulling model weights from Hugging Face via mirror: ${HF_ENDPOINT} ..."
else
  unset HF_ENDPOINT || true
  echo "Pulling model weights from Hugging Face ..."
fi

"${cmd[@]}"
