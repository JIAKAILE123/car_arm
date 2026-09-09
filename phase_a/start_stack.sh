#!/usr/bin/env bash
# 阶段 A：一键开 4 个终端标签（Gazebo / MoveIt / Nav2 / rosbridge）
set -euo pipefail

WS="${HOME}/dev_ws"
SRC="${WS}/src/yzbot"
SETUP="source /opt/ros/humble/setup.bash && source ${WS}/install/setup.bash"

if [[ ! -f "${WS}/install/setup.bash" ]]; then
  echo "找不到 ${WS}/install/setup.bash，请先 colcon build"
  exit 1
fi

if ! command -v gnome-terminal >/dev/null 2>&1; then
  echo "没有 gnome-terminal。请按 phase_a/README.md 手动开 4 个终端。"
  exit 1
fi

run_tab() {
  local title="$1"
  local cmd="$2"
  gnome-terminal --tab --title="${title}" -- bash -lc "${SETUP}; echo '=== ${title} ==='; ${cmd}; exec bash"
}

run_tab "1-Gazebo" "ros2 launch mybot gazebo_world2.launch.py"
sleep 2
run_tab "2-MoveIt" "ros2 launch mybot my_moveit_rviz.launch.py"
sleep 2
run_tab "3-Nav2" "ros2 launch bot_navigation nav_bringup_gazebo2.launch.py"
sleep 2
run_tab "4-rosbridge" "ros2 launch rosbridge_server rosbridge_websocket_launch.xml"

echo
echo "已尝试打开 4 个终端标签。"
echo "下一步：等 Gazebo/Nav2 起来后执行："
echo "  cd ${SRC}/phase_a && bash check_ready.sh"
echo "然后在 Nav2 RViz 做 2D Pose Estimate，再跑 demo_pick_one.sh"
