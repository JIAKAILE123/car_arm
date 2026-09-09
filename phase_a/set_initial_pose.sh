#!/usr/bin/env bash
# 发布初始位姿给 AMCL（默认地图原点附近；不对就在 RViz 里手点更准）
set -eo pipefail
set +u
source /opt/ros/humble/setup.bash
source "${HOME}/dev_ws/install/setup.bash"
set -u

X="${1:-0.0}"
Y="${2:-0.0}"
YAW="${3:-0.0}"

# yaw -> quaternion (z,w)，假定 roll=pitch=0
# qz = sin(yaw/2), qw = cos(yaw/2)
read -r QZ QW < <(python3 - <<PY
import math
yaw=float("${YAW}")
print(math.sin(yaw/2.0), math.cos(yaw/2.0))
PY
)

echo "发布 initialpose: x=${X} y=${Y} yaw=${YAW}"
ros2 topic pub --once /initialpose geometry_msgs/msg/PoseWithCovarianceStamped "{
  header: {frame_id: 'map'},
  pose: {
    pose: {
      position: {x: ${X}, y: ${Y}, z: 0.0},
      orientation: {x: 0.0, y: 0.0, z: ${QZ}, w: ${QW}}
    },
    covariance: [0.25, 0.0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.25, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0, 0.0, 0.0, 0.068]
  }
}"
