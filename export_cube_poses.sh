#!/usr/bin/env bash
# Export zone A/B/C + 10 cubes via Gazebo Classic `gz model -p`.
# Usage:  bash ~/dev_ws/yzbot/src/export_cube_poses.sh
set -euo pipefail
OUT="${1:-$HOME/dev_ws/yzbot/cube_poses_export.yaml}"
export GAZEBO_MASTER_URI="${GAZEBO_MASTER_URI:-http://127.0.0.1:11345}"

if ! gz model -m zone_a -p >/dev/null 2>&1; then
  echo "ERROR: Gazebo not reachable. Is gazebo_world2 running?"
  echo "  test: gz model -m zone_a -p"
  exit 1
fi

{
  echo "# Auto-exported $(date '+%F %T')"
  echo "zones:"
} >"$OUT"

echo "===== ZONE_POSES ====="
for name in zone_a zone_b zone_c; do
  read -r x y z _rest <<<"$(gz model -m "$name" -p 2>/dev/null | tr -d '\r')"
  z=0.01
  printf "  %s: {x: %.4f, y: %.4f, z: %.4f}\n" "$name" "$x" "$y" "$z" >>"$OUT"
  letter=${name: -1}
  letter=$(echo "$letter" | tr '[:lower:]' '[:upper:]')
  printf "  zone_%s  (%.2f, %.2f, %.3f)\n" "$letter" "$x" "$y" "$z"
done

echo "cubes:" >>"$OUT"
echo "===== CUBE_POSES ====="
for c in red blue; do
  for i in 1 2 3 4 5; do
    name="${c}_cube_${i}"
    read -r x y z _rest <<<"$(gz model -m "$name" -p 2>/dev/null | tr -d '\r')"
    z=0.015
    printf "  %s: {x: %.4f, y: %.4f, z: %.4f}\n" "$name" "$x" "$y" "$z" >>"$OUT"
    printf "  %-12s  (%.2f, %.2f, %.3f)\n" "$name" "$x" "$y" "$z"
  done
done
echo "WROTE $OUT"
echo "Apply into code/world:"
echo "  ros2 run yzbot_task export_cube_poses --ros-args -p once:=true -p apply:=true"
