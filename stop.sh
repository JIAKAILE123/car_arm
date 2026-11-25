#!/usr/bin/env bash
# 一键清场：停掉 run.sh 起的全部进程（Gazebo / Nav2 / viz / task_sm / llm / 出题）
# 用法： ./stop.sh          （默认杀掉本工作空间相关进程 + coStudio）
#        KEEP_COSTUDIO=1 ./stop.sh   （保留 coStudio 窗口）

KEEP_COSTUDIO="${KEEP_COSTUDIO:-0}"

echo "=== 正在结束仿真与任务进程 ==="

# 1) Gazebo
killall -9 gzserver gzclient 2>/dev/null && echo "  · gazebo 已结束"

# 2) ROS2 launch / 节点
pkill -9 -f "ros2 launch" 2>/dev/null && echo "  · ros2 launch 已结束"
for n in task_sm llm_solver exam_feeder practice_exam gt_localization route_planner \
         dyn_obs_guard dyn_obs_predict arm_js_filter; do
  pkill -9 -f "yzbot_task/(${n}|${n}:main)" 2>/dev/null
  pkill -9 -f "ros2 run yzbot_task ${n}" 2>/dev/null
done
echo "  · yzbot_task 节点已结束"

# 3) Nav2 / 定位 / 桥接
pkill -9 -f "nav_bringup_gazebo2" 2>/dev/null
pkill -9 -f "component_container" 2>/dev/null
pkill -9 -f "amcl|planner_server|controller_server|bt_navigator|behavior_server|velocity_smoother|collision_monitor" 2>/dev/null
pkill -9 -f "rosbridge|cobridge" 2>/dev/null
echo "  · Nav2 / 可视化转发已结束"

# 4) coStudio
if [[ "${KEEP_COSTUDIO}" != "1" ]]; then
  pkill -9 -f "coStudio/honeybee" 2>/dev/null && echo "  · coStudio 已结束"
fi

sleep 1
left=$(ps -eo args | grep -E "gzserver|task_sm|llm_solver|component_container|honeybee" | grep -v grep | wc -l)
echo "=== 清场完成（残留进程 ${left} 个）==="