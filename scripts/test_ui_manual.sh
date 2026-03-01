#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMON_LIB="${SCRIPT_DIR}/common_ui.sh"

if [[ ! -f "$COMMON_LIB" ]]; then
  echo "Missing shared shell library: ${COMMON_LIB}"
  exit 1
fi
source "$COMMON_LIB"

if [[ ! -t 0 || ! -t 1 ]]; then
  echo "Manual UI test requires a TTY."
  echo "Run in a terminal:"
  echo "  bash scripts/test_ui_manual.sh"
  exit 1
fi

print_header() {
  clear
  cat <<'EOF'
=========================================
 Person_reID UI Manual Test Playground
=========================================
EOF
  echo "This script only tests UI utilities in scripts/common_ui.sh."
  echo "It does not execute training/evaluation workflows."
  echo
}

demo_badges() {
  ui_info "Info badge example"
  ui_tip "Tip badge example"
  ui_success "Success badge example"
  ui_warn "Warn badge example"
  ui_error "Error badge example"
}

demo_input() {
  local epochs=""
  local lr=""
  local run_name=""

  echo
  ui_info "Validated input demo"
  ui_input epochs "warm_epoch" "5" '^[0-9]+$' "" "warm_epoch must be a non-negative integer."
  ui_input lr "lr" "0.01" '^([0-9]+([.][0-9]+)?|[.][0-9]+)$' "v > 0" "lr must be a positive number."
  ui_input run_name "run_name" "demo_run"
  ui_success "Collected: warm_epoch=${epochs}, lr=${lr}, run_name=${run_name}"
}

demo_paged_select() {
  local i
  local -a options=()
  local idx=""
  local value=""

  for ((i = 1; i <= 24; i++)); do
    options+=("Dataset candidate #${i}")
  done

  echo
  ui_info "Paged selector demo (try Up/Down, k/j, Enter, q)"
  if ui_select idx value "Select one candidate:" "12" "${options[@]}"; then
    ui_success "Selected index=${idx}, value='${value}'"
  else
    local rc=$?
    if [[ $rc -eq 130 ]]; then
      ui_warn "Selection canceled by user."
    else
      ui_error "Selection failed with rc=${rc}."
    fi
  fi
}

main() {
  local action=""

  while true; do
    print_header
    action=""
    if ! ui_select _menu_idx action "Choose a UI test action:" "1" \
      "Show badge styles" \
      "Run validated input demo" \
      "Run paged selector demo" \
      "Run all demos" \
      "Exit"; then
      break
    fi

    case "$action" in
      "Show badge styles")
        demo_badges
        ;;
      "Run validated input demo")
        demo_input
        ;;
      "Run paged selector demo")
        demo_paged_select
        ;;
      "Run all demos")
        demo_badges
        demo_input
        demo_paged_select
        ;;
      "Exit")
        break
        ;;
    esac

    echo
    read -r -p "Press Enter to return to menu..."
  done

  echo "Manual UI test finished."
}

main
