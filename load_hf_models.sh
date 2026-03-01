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
  local script_name
  script_name="$(basename "$0")"
  echo "+----------------------------------------------------------+"
  echo "| Designed by Wilson | Implemented by Codex               |"
  printf "| Script: %-48s |\n" "$script_name"
  echo "+----------------------------------------------------------+"
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
    printf "%s %b%s%b\n" "$title" "$color_selected" "$selected_text" "$color_reset"
  else
    printf "%s %s\n" "$title" "$selected_text"
  fi

  printf -v "$__outvar" '%s' "$choice"
}

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
      echo "Invalid default option for confirmer."
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
        else
          if [[ "$answer" == "Y" ]]; then
            shown_answer="Y"
          else
            shown_answer="N"
          fi
          printf "%s: %s\n" "$prompt" "$shown_answer" >&2
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

cmd=(python load_hf_models.py "${backbone_flags[@]}")
if [[ "$use_hf_mirror" == "Y" ]]; then
  export HF_ENDPOINT="https://hf-mirror.com"
  echo "Pulling model weights from Hugging Face via mirror: ${HF_ENDPOINT} ..."
else
  unset HF_ENDPOINT || true
  echo "Pulling model weights from Hugging Face ..."
fi

"${cmd[@]}"
