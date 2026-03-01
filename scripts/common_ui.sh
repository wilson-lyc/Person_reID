#!/usr/bin/env bash

# Standalone UI helpers for interactive shell scripts.
# This file is intentionally independent from scripts/common.sh.

ui_use_color() {
  [[ -t 1 ]]
}

ui_badge() {
  local kind="$1"
  local text="$2"
  if ui_use_color; then
    case "$kind" in
      info)    printf '\033[1;44m INFO \033[0m %s\n' "$text" ;;
      tip)     printf '\033[1;46m TIP  \033[0m %s\n' "$text" ;;
      success) printf '\033[1;42m OK   \033[0m %s\n' "$text" ;;
      warn)    printf '\033[1;43m WARN \033[0m %s\n' "$text" ;;
      error)   printf '\033[1;41m ERR  \033[0m %s\n' "$text" ;;
      *)       printf '%s\n' "$text" ;;
    esac
  else
    case "$kind" in
      info)    printf '[INFO] %s\n' "$text" ;;
      tip)     printf '[TIP] %s\n' "$text" ;;
      success) printf '[OK] %s\n' "$text" ;;
      warn)    printf '[WARN] %s\n' "$text" ;;
      error)   printf '[ERR] %s\n' "$text" ;;
      *)       printf '%s\n' "$text" ;;
    esac
  fi
}

ui_info() { ui_badge info "$*"; }
ui_tip() { ui_badge tip "$*"; }
ui_success() { ui_badge success "$*"; }
ui_warn() { ui_badge warn "$*"; }
ui_error() { ui_badge error "$*"; }

# Validated input prompt.
# Usage:
#   ui_input out_var "label" "default" [regex] [range_expr] [err_msg]
ui_input() {
  local __outvar="$1"
  local label="$2"
  local default_value="${3:-}"
  local regex="${4:-}"
  local range_expr="${5:-}"
  local err_msg="${6:-Invalid input.}"
  local value=""
  local ok=1

  while true; do
    read -r -p "${label} [${default_value}]: " value || value=""
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
    ui_warn "$err_msg"
  done

  if ui_use_color; then
    printf '\033[1A\r\033[2K'
    printf '%s: \033[1;36m%s\033[0m\n' "$label" "$value"
  else
    printf '%s: %s\n' "$label" "$value"
  fi

  printf -v "$__outvar" '%s' "$value"
}

# Enhanced selector with page support and cancel key.
# Usage:
#   ui_select out_idx_var out_value_var "Title" "default_index_1_based" "Option A" ...
# Result:
#   - Writes 1-based index to out_idx_var and option text to out_value_var.
#   - Also exports UI_SELECT_INDEX / UI_SELECT_VALUE.
#   - Returns 130 when user cancels with q/Q (TTY mode).
ui_select() {
  local __out_idx_var="$1"
  local __out_val_var="$2"
  local title="$3"
  local default_index="$4"
  shift 4
  local -a options=("$@")
  local selected_idx=0
  local key=""
  local esc_tail=""
  local input_choice=""
  local page_size=8
  local start=0
  local end=0
  local rendered_lines=0
  local rendered_once=0
  local tty_mode=0

  if [[ ${#options[@]} -eq 0 ]]; then
    ui_error "ui_select: requires at least one option."
    return 1
  fi

  if [[ ! "$default_index" =~ ^[0-9]+$ ]] || (( default_index < 1 || default_index > ${#options[@]} )); then
    ui_error "ui_select: invalid default index '${default_index}', expected 1..${#options[@]}."
    return 1
  fi
  selected_idx=$((default_index - 1))

  if [[ -t 0 && -t 1 ]]; then
    tty_mode=1
  fi

  if (( tty_mode == 0 )); then
    echo "$title"
    local i
    for i in "${!options[@]}"; do
      printf "  %d) %s\n" "$((i + 1))" "${options[$i]}"
    done
    while true; do
      read -r -p "Enter choice [${default_index}]: " input_choice || input_choice=""
      input_choice="${input_choice:-$default_index}"
      if [[ "$input_choice" =~ ^[0-9]+$ ]] && (( input_choice >= 1 && input_choice <= ${#options[@]} )); then
        selected_idx=$((input_choice - 1))
        break
      fi
      ui_warn "Invalid choice: ${input_choice}. Please input 1..${#options[@]}."
    done
  else
    local term_lines
    term_lines="$(tput lines 2>/dev/null || echo 24)"
    page_size=$((term_lines - 6))
    if (( page_size < 5 )); then
      page_size=5
    fi

    _ui_clear_select_block() {
      local lines="$1"
      local j
      for ((j = 0; j < lines; j++)); do
        printf "\r\033[1A\033[2K"
      done
    }

    while true; do
      if (( rendered_once == 1 )); then
        _ui_clear_select_block "$rendered_lines"
      fi

      if (( selected_idx < start )); then
        start=$selected_idx
      fi
      if (( selected_idx >= start + page_size )); then
        start=$((selected_idx - page_size + 1))
      fi
      end=$((start + page_size - 1))
      if (( end >= ${#options[@]} )); then
        end=$((${#options[@]} - 1))
      fi

      printf "%s\n" "$title"
      printf "  (Up/Down or k/j, Enter confirm, q cancel)\n"
      local idx
      for ((idx = start; idx <= end; idx++)); do
        if (( idx == selected_idx )); then
          if ui_use_color; then
            printf "\033[1;34m> %s\033[0m\n" "${options[$idx]}"
          else
            printf "> %s\n" "${options[$idx]}"
          fi
        else
          printf "  %s\n" "${options[$idx]}"
        fi
      done
      if (( ${#options[@]} > page_size )); then
        printf "  [%d-%d / %d]\n" "$((start + 1))" "$((end + 1))" "${#options[@]}"
      fi

      rendered_lines=$((3 + end - start + 1))
      if (( ${#options[@]} > page_size )); then
        rendered_lines=$((rendered_lines + 1))
      fi
      rendered_once=1

      IFS= read -rsn1 key
      if [[ "$key" == $'\x1b' ]]; then
        esc_tail=""
        IFS= read -rsn2 -t 0.05 esc_tail || true
        key+="$esc_tail"
      fi

      case "$key" in
        $'\x1b[A'|k|w)
          if (( selected_idx > 0 )); then
            selected_idx=$((selected_idx - 1))
          fi
          ;;
        $'\x1b[B'|j|s)
          if (( selected_idx < ${#options[@]} - 1 )); then
            selected_idx=$((selected_idx + 1))
          fi
          ;;
        q|Q)
          _ui_clear_select_block "$rendered_lines"
          ui_warn "Operation canceled."
          return 130
          ;;
        ""|$'\n'|$'\r')
          break
          ;;
      esac
    done

    _ui_clear_select_block "$rendered_lines"
    if ui_use_color; then
      printf "\033[1;36m%s\033[0m\n" "${options[$selected_idx]}"
    else
      printf "%s\n" "${options[$selected_idx]}"
    fi
  fi

  UI_SELECT_INDEX=$((selected_idx + 1))
  UI_SELECT_VALUE="${options[$selected_idx]}"
  printf -v "$__out_idx_var" '%s' "$UI_SELECT_INDEX"
  printf -v "$__out_val_var" '%s' "$UI_SELECT_VALUE"
}

# Boolean confirm built on ui_select.
# Usage:
#   ui_confirm "Prompt?" "y|n" ["yes_label"] ["no_label"]
# Output:
#   prints y or n
ui_confirm() {
  local prompt="$1"
  local default_choice="${2:-y}"
  local yes_label="${3:-yes}"
  local no_label="${4:-no}"
  local default_index=1
  local idx=""
  local val=""

  case "$default_choice" in
    y|Y) default_index=1 ;;
    n|N) default_index=2 ;;
    *)
      ui_error "ui_confirm: default must be y or n."
      return 1
      ;;
  esac

  ui_select idx val "$prompt" "$default_index" "$yes_label" "$no_label" >&2 || return $?
  if [[ "$idx" == "1" ]]; then
    printf 'y\n'
  else
    printf 'n\n'
  fi
}
