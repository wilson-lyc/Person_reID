#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

print_banner() {
cat <<'EOF'
██████╗ ██████╗ ███████╗██████╗  █████╗ ██████╗ ███████╗    ███╗   ███╗ ██████╗ ██████╗ ███████╗██╗
██╔══██╗██╔══██╗██╔════╝██╔══██╗██╔══██╗██╔══██╗██╔════╝    ████╗ ████║██╔═══██╗██╔══██╗██╔════╝██║
██████╔╝██████╔╝█████╗  ██████╔╝███████║██████╔╝█████╗      ██╔████╔██║██║   ██║██║  ██║█████╗  ██║
██╔═══╝ ██╔══██╗██╔══╝  ██╔═══╝ ██╔══██║██╔══██╗██╔══╝      ██║╚██╔╝██║██║   ██║██║  ██║██╔══╝  ██║
██║     ██║  ██║███████╗██║     ██║  ██║██║  ██║███████╗    ██║ ╚═╝ ██║╚██████╔╝██████╔╝███████╗███████╗
╚═╝     ╚═╝  ╚═╝╚══════╝╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝    ╚═╝     ╚═╝ ╚═════╝ ╚═════╝ ╚══════╝╚══════╝
EOF
}

select_menu() {
  local __outvar="$1"
  local title="$2"
  local default_choice="$3"
  shift 3
  local options=("$@")
  local choice=""

  echo "$title"
  for i in "${!options[@]}"; do
    printf "  %d) %s\n" "$((i + 1))" "${options[$i]}"
  done

  while true; do
    read -r -p "Enter choice [${default_choice}]: " choice
    choice="${choice:-$default_choice}"
    if [[ "$choice" =~ ^[0-9]+$ ]] && (( choice >= 1 && choice <= ${#options[@]} )); then
      break
    fi
    echo "Invalid choice: ${choice}"
  done

  printf -v "$__outvar" '%s' "$choice"
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
      echo "Invalid default option for confirmer."
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
        echo "Invalid input. Please enter 'Y' or 'n'."
        ;;
    esac
  done
}

if [[ ! -f "load_hf_models.py" ]]; then
  echo "Missing file: load_hf_models.py"
  exit 1
fi

clear
print_banner
echo

select_menu backbone_choice "Select HF backbone to preload:" "1" \
  "Swin (swin_base_patch4_window7_224)" \
  "SwinV2 (swinv2_base_window8_256)" \
  "DINOv3 ViT-Base (vit_base_patch16_dinov3.lvd1689m)" \
  "ConvNeXt-Base (convnext_base)" \
  "HRNet-W18 (hrnet_w18)" \
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

use_hf_mirror="$(confirmer "Use Hugging Face mirror (https://hf-mirror.com)?" "yes")"

read -r -p "Input height for SwinV2/DINO [256]: " input_h
read -r -p "Input width for SwinV2/DINO [128]: " input_w
input_h="${input_h:-256}"
input_w="${input_w:-128}"

cmd=(python load_hf_models.py "${backbone_flags[@]}" --height "$input_h" --width "$input_w")
if [[ "$use_hf_mirror" == "Y" ]]; then
  cmd+=(--use-hf-mirror)
fi

echo
echo "Running:"
printf '  %q' "${cmd[@]}"
echo
echo
"${cmd[@]}"
