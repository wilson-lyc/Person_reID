#!/usr/bin/env bash

# Purpose:
#   Standalone UI helpers for interactive shell scripts.
# Agent Notes:
#   - Keep this file dependency-free except basic POSIX/bash tools.
#   - Do not assume scripts/common.sh is sourced.
#   - Prefer stable function signatures; other scripts call these directly.

# Purpose:
#   Detect whether colored output is appropriate for current stdout.
# Inputs:
#   None.
# Outputs:
#   None (boolean via exit code).
# Exit Codes:
#   0: color-capable terminal.
#   1: non-interactive output.
ui_use_color() {
  [[ -t 1 ]]
}

# Purpose:
#   Print a one-line badge message with optional color.
# Inputs:
#   $1 kind: info|tip|success|warn|error|other
#   $2 text: message body
# Outputs:
#   Writes formatted text to stdout.
# Side Effects:
#   Emits terminal color escapes when tty is detected.
ui_badge() {
  local kind="$1"
  local text="$2"
  if ui_use_color; then
    case "$kind" in
      info)    printf '\033[1;94m[INFO]\033[0m %s\n' "$text" ;;
      tip)     printf '\033[1;96m[TIP]\033[0m %s\n' "$text" ;;
      success) printf '\033[1;92m[OK]\033[0m %s\n' "$text" ;;
      warn)    printf '\033[1;93m[WARN]\033[0m %s\n' "$text" ;;
      error)   printf '\033[1;91m[ERR]\033[0m %s\n' "$text" ;;
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

# Purpose: Wrapper of ui_badge(kind=info).
ui_info() { ui_badge info "$*"; }
# Purpose: Wrapper of ui_badge(kind=tip).
ui_tip() { ui_badge tip "$*"; }
# Purpose: Wrapper of ui_badge(kind=success).
ui_success() { ui_badge success "$*"; }
# Purpose: Wrapper of ui_badge(kind=warn).
ui_warn() { ui_badge warn "$*"; }
# Purpose: Wrapper of ui_badge(kind=error).
ui_error() { ui_badge error "$*"; }

# Purpose:
#   Prompt a yes/no selection and return normalized 0/1 value.
# Inputs:
#   $1 out_var: destination variable name (written via printf -v).
#   $2 prompt: question shown to user.
#   $3 default_index (optional): 1 for yes, 2 for no; defaults to 1.
# Outputs:
#   Stores yes or no into out_var.
# Exit Codes:
#   0 on successful selection.
ui_yes_no() {
  local __outvar="$1"
  local prompt="$2"
  local default_index="${3:-1}"
  local yn_idx="" yn_val=""

  ui_select yn_idx yn_val "$prompt" "$default_index" "yes" "no"
  if [[ "$yn_idx" == "1" ]]; then
    printf -v "$__outvar" 'yes'
  else
    printf -v "$__outvar" 'no'
  fi
}

# Purpose:
#   Print project banner and script metadata.
# Inputs:
#   $1 script_name (optional): displayed script name, defaults to basename "$0".
# Outputs:
#   Banner and metadata lines to stdout.
ui_banner() {
  local script_name="${1:-$(basename "$0")}"
  cat <<'EOF'
██████╗ ███████╗██████╗ ███████╗ ██████╗ ███╗   ██╗    ██████╗ ███████╗██╗██████╗
██╔══██╗██╔════╝██╔══██╗██╔════╝██╔═══██╗████╗  ██║    ██╔══██╗██╔════╝██║██╔══██╗
██████╔╝█████╗  ██████╔╝███████╗██║   ██║██╔██╗ ██║    ██████╔╝█████╗  ██║██║  ██║
██╔═══╝ ██╔══╝  ██╔══██╗╚════██║██║   ██║██║╚██╗██║    ██╔══██╗██╔══╝  ██║██║  ██║
██║     ███████╗██║  ██║███████║╚██████╔╝██║ ╚████║    ██║  ██║███████╗██║██████╔╝
╚═╝     ╚══════╝╚═╝  ╚═╝╚══════╝ ╚═════╝ ╚═╝  ╚═══╝    ╚═╝  ╚═╝╚══════╝╚═╝╚═════╝
EOF
  printf '%s\n' "============================================================"
  printf '%s\n' "Designed by Wilson | Implemented by Codex"
  printf 'Script: %s\n' "$script_name"
  printf '%s\n' "============================================================"
}

# Purpose:
#   Prompt for scalar input with validation and store into caller variable.
# Inputs:
#   $1 out_var: destination variable name (written via printf -v).
#   $2 label: prompt/field label.
#   $3 default: fallback when user presses Enter.
#   $4 regex (optional): bash regex condition.
#   $5 range_expr (optional): awk boolean expression using variable v.
#   $6 err_msg (optional): message shown on invalid input.
# Outputs:
#   Prints a normalized confirmation line after accepted input.
# Exit Codes:
#   0: value stored in out_var.
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

# Purpose:
#   Interactive selector with TTY navigation and non-TTY numeric fallback.
# Inputs:
#   $1 out_idx_var: destination variable for selected 1-based index.
#   $2 out_value_var: destination variable for selected option text.
#   $3 title: selector title/prompt.
#   $4 default_index: default 1-based index.
#   $5... options: selectable option labels.
# Outputs:
#   Sets caller vars via printf -v and exports UI_SELECT_INDEX/UI_SELECT_VALUE.
# Exit Codes:
#   0: selection confirmed.
#   1: invalid arguments.
#   130: canceled or interrupted.
# Agent Notes:
#   - Non-TTY mode prints numbered options and reads numeric choice.
#   - TTY mode supports arrows, k/j, Enter confirm, q cancel.
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
  local tty_mode=0
  local default_tag=" (default)"

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
    local option_text
    for i in "${!options[@]}"; do
      option_text="${options[$i]}"
      if (( i + 1 == default_index )); then
        option_text+="$default_tag"
      fi
      printf "  %d) %s\n" "$((i + 1))" "$option_text"
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
    local has_tput=0
    local use_alt_screen=0
    local canceled=0
    local interrupted=0
    local idx
    if command -v tput >/dev/null 2>&1; then
      has_tput=1
    fi

    if (( has_tput == 1 )); then
      term_lines="$(tput lines 2>/dev/null || echo 24)"
    else
      term_lines=24
    fi
    page_size=$((term_lines - 6))
    if (( page_size < 5 )); then
      page_size=5
    fi

    # Purpose:
    #   Render current selector page in interactive TTY mode.
    # Side Effects:
    #   Clears screen region and prints title/options/page info.
    _ui_select_draw() {
      if (( has_tput == 1 )); then
        tput cup 0 0 2>/dev/null || true
        tput ed 2>/dev/null || true
      else
        clear
      fi

      printf "%s\n" "$title"
      printf "  (Up/Down or k/j, Enter confirm, q cancel)\n"
      local display_text
      for ((idx = start; idx <= end; idx++)); do
        display_text="${options[$idx]}"
        if (( idx + 1 == default_index )); then
          display_text+="$default_tag"
        fi
        if (( idx == selected_idx )); then
          if ui_use_color; then
            printf "\033[1;34m> %s\033[0m\n" "$display_text"
          else
            printf "> %s\n" "$display_text"
          fi
        else
          printf "  %s\n" "$display_text"
        fi
      done
      if (( ${#options[@]} > page_size )); then
        printf "  [%d-%d / %d]\n" "$((start + 1))" "$((end + 1))" "${#options[@]}"
      fi
    }

    # Purpose:
    #   Restore terminal state after interactive selector.
    # Side Effects:
    #   Re-enables cursor and exits alt screen when enabled.
    _ui_select_restore() {
      if (( has_tput == 1 )); then
        tput cnorm 2>/dev/null || true
        if (( use_alt_screen == 1 )); then
          tput rmcup 2>/dev/null || true
        fi
      fi
    }

    # Purpose:
    #   Handle INT/TERM while selector is active.
    # Exit Codes:
    #   130 to match shell interrupt/cancel semantics.
    _ui_select_on_interrupt() {
      interrupted=1
      _ui_select_restore
      trap - INT TERM
      return 130
    }

    if (( has_tput == 1 )); then
      tput smcup 2>/dev/null && use_alt_screen=1 || use_alt_screen=0
      tput civis 2>/dev/null || true
    fi
    trap '_ui_select_on_interrupt' INT TERM

    while true; do
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

      _ui_select_draw

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
          canceled=1
          break
          ;;
        ""|$'\n'|$'\r')
          break
          ;;
      esac
    done

    trap - INT TERM
    _ui_select_restore
    if (( interrupted == 1 )); then
      return 130
    fi
    if (( canceled == 1 )); then
      ui_warn "Operation canceled."
      return 130
    fi
    if (( use_alt_screen == 0 )); then
      # Notes:
      #   Clear selector menu first, then print the final selected value.
      if (( has_tput == 1 )); then
        tput cup 0 0 2>/dev/null || true
        tput ed 2>/dev/null || true
      else
        clear
      fi
      if ui_use_color; then
        printf "\033[1;36m%s\033[0m\n" "${options[$selected_idx]}"
      else
        printf "%s\n" "${options[$selected_idx]}"
      fi
    fi
  fi

  UI_SELECT_INDEX=$((selected_idx + 1))
  UI_SELECT_VALUE="${options[$selected_idx]}"
  printf -v "$__out_idx_var" '%s' "$UI_SELECT_INDEX"
  printf -v "$__out_val_var" '%s' "$UI_SELECT_VALUE"
}
