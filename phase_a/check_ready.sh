#!/usr/bin/env bash
# 阶段 A：检查控制器 / 吸附服务 / 导航相关是否就绪
set -eo pipefail
# ROS setup.bash 会读未定义变量，不能开 nounset
set +u
source /opt/ros/humble/setup.bash
source "${HOME}/dev_ws/install/setup.bash"
set -u

ok=0
fail=0

check() {
  local name="$1"
  shift
  if "$@" >/dev/null 2>&1; then
    echo "[OK]   ${name}"
    ok=$((ok + 1))
  else
    echo "[FAIL] ${name}"
    fail=$((fail + 1))
  fi
}

echo "=== 阶段 A 就绪检查 ==="

# 控制器
if ros2 control list_controllers 2>/dev/null | grep -q 'arm_controller.*active'; then
  echo "[OK]   arm_controller active"
  ok=$((ok + 1))
else
  echo "[FAIL] arm_controller active"
  fail=$((fail + 1))
fi

if ros2 control list_controllers 2>/dev/null | grep -q 'gripper_controller.*active'; then
  echo "[OK]   gripper_controller active"
  ok=$((ok + 1))
else
  echo "[FAIL] gripper_controller active"
  fail=$((fail + 1))
fi

if ros2 control list_controllers 2>/dev/null | grep -q 'joint_state_broadcaster.*active'; then
  echo "[OK]   joint_state_broadcaster active"
  ok=$((ok + 1))
else
  echo "[FAIL] joint_state_broadcaster active"
  fail=$((fail + 1))
fi

# 吸附
check "服务 /ATTACHLINK" ros2 service type /ATTACHLINK
check "服务 /DETACHLINK" ros2 service type /DETACHLINK

# 导航相关
check "话题 /cmd_vel" ros2 topic type /cmd_vel
check "话题 /scan 或激光" bash -c 'ros2 topic list | grep -Eq "/scan|/laser"'
check "action /navigate_to_pose" bash -c 'ros2 action list | grep -q navigate_to_pose'

# rosbridge（可选，未装包时只提示）
if ros2 pkg list 2>/dev/null | grep -q rosbridge_server; then
  if ros2 node list 2>/dev/null | grep -qi rosbridge; then
    echo "[OK]   rosbridge 节点在跑"
    ok=$((ok + 1))
  else
    echo "[WARN] rosbridge 包已装，但节点未启动（可开第 4 终端）"
  fi
else
  echo "[WARN] 未安装 ros-humble-rosbridge-suite（可视化接线前需安装）"
fi

echo
echo "通过 ${ok} 项，失败 ${fail} 项"
if [[ "${fail}" -gt 0 ]]; then
  echo "提示：先确认 Gazebo 已加载完控制器；Nav2 已启动；工作空间已 source。"
  exit 1
fi

echo "下一步：在 Nav2 RViz 做 2D Pose Estimate，或 bash set_initial_pose.sh"
exit 0
