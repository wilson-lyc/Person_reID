#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMON_LIB="${SCRIPT_DIR}/common.sh"

if [[ ! -f "$COMMON_LIB" ]]; then
  echo "Missing shared shell library: ${COMMON_LIB}"
  exit 1
fi
source "$COMMON_LIB"

pass_count=0
fail_count=0

pass() {
  local msg="$1"
  pass_count=$((pass_count + 1))
  echo "[PASS] ${msg}"
}

fail() {
  local msg="$1"
  fail_count=$((fail_count + 1))
  echo "[FAIL] ${msg}"
}

assert_eq() {
  local actual="$1"
  local expected="$2"
  local msg="$3"
  if [[ "$actual" == "$expected" ]]; then
    pass "$msg"
  else
    fail "${msg} (expected='${expected}', actual='${actual}')"
  fi
}

run_auto_tests() {
  local selected=""
  local status=0

  echo "Running non-interactive selector tests..."

  selected=""
  SELECTOR_INDEX=""
  SELECTOR_VALUE=""
  selector selected "Color" "2" "Red" "Green" "Blue" <<< "" >/dev/null
  assert_eq "$selected" "Green" "Default selection returns option text"
  assert_eq "$SELECTOR_INDEX" "2" "Default selection keeps 1-based index"
  assert_eq "$SELECTOR_VALUE" "Green" "Default selection exports SELECTOR_VALUE"

  selected=""
  SELECTOR_INDEX=""
  SELECTOR_VALUE=""
  selector selected "Color" "2" "Red" "Green" "Blue" <<< "3" >/dev/null
  assert_eq "$selected" "Blue" "Explicit numeric choice returns option text"
  assert_eq "$SELECTOR_INDEX" "3" "Explicit numeric choice updates SELECTOR_INDEX"
  assert_eq "$SELECTOR_VALUE" "Blue" "Explicit numeric choice updates SELECTOR_VALUE"

  selected=""
  SELECTOR_INDEX=""
  SELECTOR_VALUE=""
  selector selected "Color" "1" "Red" "Green" "Blue" <<< $'99\n\n' >/dev/null
  assert_eq "$selected" "Red" "Invalid then Enter falls back to default"
  assert_eq "$SELECTOR_INDEX" "1" "Fallback default index is correct"

  selected=""
  status=0
  if selector selected "Color" "0" "Red" "Green" "Blue" >/dev/null 2>&1; then
    status=0
  else
    status=$?
  fi
  if [[ $status -ne 0 ]]; then
    pass "Invalid default index returns non-zero"
  else
    fail "Invalid default index should return non-zero"
  fi

  echo
  echo "Auto test summary: pass=${pass_count}, fail=${fail_count}"
  if [[ $fail_count -ne 0 ]]; then
    return 1
  fi
}

run_interactive_demo() {
  if [[ ! -t 0 || ! -t 1 ]]; then
    echo "Interactive demo requires a TTY."
    return 1
  fi
  local selected=""
  echo "Interactive demo:"
  echo "  Use Up/Down arrow (or k/j) to move, Enter to confirm."
  selector selected "Dataset" "2" "Market-1501" "DukeMTMC-reID" "MSMT17" "CUB-200-2011"
  echo "Returned value: ${selected}"
  echo "Returned index: ${SELECTOR_INDEX}"
}

main() {
  local mode="${1:---auto}"
  case "$mode" in
    --auto)
      run_auto_tests
      ;;
    --interactive)
      run_interactive_demo
      ;;
    --all)
      run_auto_tests
      run_interactive_demo
      ;;
    *)
      echo "Usage: $0 [--auto|--interactive|--all]"
      return 1
      ;;
  esac
}

main "${1:---auto}"
