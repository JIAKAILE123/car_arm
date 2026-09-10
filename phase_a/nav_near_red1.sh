#!/usr/bin/env bash
# 导航到 red_cube_1 附近（方块约在 -1.5, 1.0；停在稍外侧避免撞桌）
set -eo pipefail
set +u
source /opt/ros/humble/setup.bash
source "${HOME}/dev_ws/yzbot/install/setup.bash"
set -u

X="${1:--1.15}"
Y="${2:-1.0}"
YAW="${3:-3.14}"

read -r QZ QW < <(python3 - <<PY
import math
yaw=float("${YAW}")
print(math.sin(yaw/2.0), math.cos(yaw/2.0))
PY
)

echo "navigate_to_pose -> (${X}, ${Y}, yaw=${YAW})"
ros2 action send_goal /navigate_to_pose nav2_msgs/action/NavigateToPose "{
  pose: {
    header: {frame_id: 'map'},
    pose: {
      position: {x: ${X}, y: ${Y}, z: 0.0},
      orientation: {x: 0.0, y: 0.0, z: ${QZ}, w: ${QW}}
    }
  }
}"
