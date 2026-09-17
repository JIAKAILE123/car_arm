#!/usr/bin/env bash
# Run vendored cobridge (no apt install required).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../../third_party/cobridge_ros" && pwd)"
if [[ ! -x "${ROOT}/lib/cobridge/cobridge" ]]; then
  # fallback: relative to install share → workspace third_party
  WS="$(cd "$(dirname "$0")/../../../../.." && pwd)"
  ROOT="${WS}/third_party/cobridge_ros"
fi
export LD_LIBRARY_PATH="${ROOT}/lib:${LD_LIBRARY_PATH:-}"
export AMENT_PREFIX_PATH="${ROOT}:${AMENT_PREFIX_PATH:-}"
PORT="${PORT:-21274}"
USE_SIM_TIME="${USE_SIM_TIME:-true}"
exec "${ROOT}/lib/cobridge/cobridge" --ros-args \
  -r __node:=cobridge \
  -p "port:=${PORT}" \
  -p address:=0.0.0.0 \
  -p "use_sim_time:=${USE_SIM_TIME}" \
  -p send_buffer_limit:=10000000
