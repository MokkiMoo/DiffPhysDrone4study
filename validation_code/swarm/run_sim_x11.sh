#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
display_file="$script_dir/.x11_display"
display_to_use="${DIFFPHYS_DISPLAY:-}"

if [[ -z "$display_to_use" && -f "$display_file" ]]; then
  display_to_use="$(<"$display_file")"
fi

if [[ -z "$display_to_use" ]]; then
  display_to_use="${1:-${DISPLAY:-}}"
fi

if [[ -z "$display_to_use" ]]; then
  echo "DISPLAY is empty. Connect with X11 forwarding first, then rerun this script." >&2
  exit 2
fi

export DISPLAY="$display_to_use"
export XAUTHORITY="${XAUTHORITY:-$HOME/.Xauthority}"

if command -v xdpyinfo >/dev/null 2>&1; then
  timeout 5s xdpyinfo >/dev/null
fi

exec "$script_dir/LinuxNoEditor/Blocks.sh" \
  -ResX=896 \
  -ResY=504 \
  -windowed \
  -WinX=512 \
  -WinY=304 \
  -settings="$script_dir/settings.json"
