#!/usr/bin/env bash
# 单次吸放验收：臂下探 → 夹爪开 → Attach → 抬起 → Detach
# 用法: bash demo_pick_one.sh [red|blue] [1-5]
set -eo pipefail

COLOR="${1:-red}"
ID="${2:-1}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

set +u
source /opt/ros/humble/setup.bash
source "${HOME}/dev_ws/yzbot/install/setup.bash"
set -u

echo "=== demo_pick_one: ${COLOR} ${ID} ==="
echo "1) 臂下探到预标定抓取姿态..."
bash "${ROOT}/send_goal_pick.sh"

echo "2) 吸附 ${COLOR}_cube_${ID} ..."
bash "${ROOT}/cube_control.sh" on "${COLOR}" "${ID}"
sleep 1

echo "3) 抬臂（回零附近）..."
ros2 action send_goal /arm_controller/follow_joint_trajectory control_msgs/action/FollowJointTrajectory "{
  trajectory: {
    joint_names: [joint1, joint2, joint3, joint4, joint5, joint6],
    points: [
      { positions: [0.0, 1.2, 1.17, 0.0, -0.3, 0.0], time_from_start: { sec: 0, nanosec: 500 } },
      { positions: [0.0, 0.6, 0.8, 0.0, -0.2, 0.0], time_from_start: { sec: 3, nanosec: 0 } },
      { positions: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], time_from_start: { sec: 6, nanosec: 0 } }
    ]
  }
}"

echo "4) 松开..."
bash "${ROOT}/cube_control.sh" off "${COLOR}" "${ID}"

echo "完成。请在 Gazebo 确认方块是否随臂抬起再落下。"
echo "若吸附时方块没动：先把底盘开到方块旁，再重跑本脚本。"
