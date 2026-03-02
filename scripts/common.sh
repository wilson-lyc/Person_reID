#!/usr/bin/env bash

# Interactive selector with arrow key support.
# Usage:
#   selector out_var "Question" "default_index_1_based" "Option A" "Option B" ...
# Result:
#   - Writes selected option text into variable named by out_var.
#   - Exports SELECTOR_INDEX (1-based) and SELECTOR_VALUE.
#   - Prints confirmation line with selected value.
selector() {
  local __outvar="$1"
  local question="$2"
  local default_index="$3"
  shift 3
  local -a choices=("$@")
  local selected_idx=0
  local selected_value=""
  local key=""
  local esc_tail=""
  local rendered_lines=0
  local rendered_once=0

  if [[ ${#choices[@]} -eq 0 ]]; then
    echo "selector: requires at least one option."
    return 1
  fi

  if [[ ! "$default_index" =~ ^[0-9]+$ ]] || (( default_index < 1 || default_index > ${#choices[@]} )); then
    echo "selector: invalid default index '${default_index}', expected 1..${#choices[@]}."
    return 1
  fi

  selected_idx=$((default_index - 1))

  if [[ ! -t 0 ]]; then
    local input_choice=""
    echo "$question"
    local i
    for i in "${!choices[@]}"; do
      printf "  %d) %s\n" "$((i + 1))" "${choices[$i]}"
    done
    while true; do
      read -r -p "Enter choice [${default_index}]: " input_choice || input_choice=""
      input_choice="${input_choice:-$default_index}"
      if [[ "$input_choice" =~ ^[0-9]+$ ]] && (( input_choice >= 1 && input_choice <= ${#choices[@]} )); then
        selected_idx=$((input_choice - 1))
        break
      fi
      echo "Invalid choice: ${input_choice}. Please input 1..${#choices[@]}."
    done
  else
    while true; do
      if (( rendered_once == 1 )); then
        # Move back to selector start and clear old menu block before re-render.
        printf "\033[%dA\033[J" "$rendered_lines"
      fi
      echo "$question"
      local idx
      for idx in "${!choices[@]}"; do
        if (( idx == selected_idx )); then
          printf "> %s\n" "${choices[$idx]}"
        else
          printf "  %s\n" "${choices[$idx]}"
        fi
      done
      rendered_lines=$((1 + ${#choices[@]}))
      rendered_once=1

      IFS= read -rsn1 key
      if [[ "$key" == $'\x1b' ]]; then
        esc_tail=""
        IFS= read -rsn2 -t 0.05 esc_tail || true
        key+="$esc_tail"
      fi

      case "$key" in
        $'\x1b[A'|k)
          if (( selected_idx > 0 )); then
            selected_idx=$((selected_idx - 1))
          fi
          ;;
        $'\x1b[B'|j)
          if (( selected_idx < ${#choices[@]} - 1 )); then
            selected_idx=$((selected_idx + 1))
          fi
          ;;
        ""|$'\n'|$'\r')
          break
          ;;
      esac
    done
  fi

  selected_value="${choices[$selected_idx]}"
  SELECTOR_INDEX=$((selected_idx + 1))
  SELECTOR_VALUE="$selected_value"
  printf -v "$__outvar" '%s' "$selected_value"

  if [[ -t 0 && -t 1 && (( rendered_once == 1 )) ]]; then
    # Remove selector-rendered block first, then echo final selected value.
    printf "\r\033[%dA\033[J" "$rendered_lines"
    # Echo the final selected value after confirming with Enter.
    printf "%s\n" "$selected_value"
  fi
  return 0
}

# Multi-option select menu wrapper based on selector.
# Usage:
#   select_menu out_var "Title" "default_index_1_based" "Option A" ...
# Result:
#   Writes selected 1-based index into out_var (compatible with legacy callers).
select_menu() {
  local __outvar="$1"
  local title="$2"
  local default_choice="$3"
  shift 3
  local options=("$@")
  local selected_text=""

  selector selected_text "$title" "$default_choice" "${options[@]}" || return 1
  printf -v "$__outvar" '%s' "$SELECTOR_INDEX"
}

# Shared yes/no confirmer for interactive scripts, powered by selector.
# Usage: confirmer "Prompt text" "y|n"
confirmer() {
  local prompt="$1"
  local default_choice="$2"
  local default_index="1"
  local selected=""

  case "$default_choice" in
    y) default_index="1" ;;
    n) default_index="2" ;;
    *)
      echo "Invalid default option for confirmer: ${default_choice} (expected y or n)."
      return 1
      ;;
  esac

  selector selected "$prompt" "$default_index" "y" "n" >&2 || return 1
  printf '%s\n' "$selected"
}
