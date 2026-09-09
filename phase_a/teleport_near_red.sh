#!/usr/bin/env bash
# [调试用] 瞬移会破坏 AMCL 定位，正式流程请不要用。
# 请改用：RViz 2D Pose Estimate → 让车慢慢开过去（task_sm / Nav2 Goal）。
echo "[WARN] teleport_near_red.sh 已停用：瞬移会让定位偏掉，导航乱飞。"
echo "正式流程："
echo "  1) 确认 Gazebo world2 + Nav2 gazebo2 已开"
echo "  2) RViz 用 2D Pose Estimate 对准 Gazebo 里车的真实位置/朝向"
echo "  3) ros2 run yzbot_task task_sm"
echo "  4) ros2 topic pub --once /task_command std_msgs/msg/String \"{data: '[red,1,A]'}\""
echo "车会自己慢慢开到红块旁，不要瞬移。"
exit 1
