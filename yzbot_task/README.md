# 阶段 B（暂不接云端 API）

包：`yzbot_task`

## 做什么

1. **本地解题**：应用题 → `x,y`（ceil(/5)）→ 按赛前映射得到 `[red,4,A;blue,1,C]`
2. **解析器**：吃固定格式任务串
3. **状态机**：按序 导航到颜色区 → 臂下探 → Attach → 抬起 → 导航放置区 → Detach

云端 API 以后接到 `/exam_question` 或直接往 `/task_command` 发字符串即可。

## 话题

| 话题 | 方向 | 内容 |
|------|------|------|
| `/exam_question` | in | 出题原文（给 local_solver） |
| `/task_command` | in | `[red,4,A;blue,1,C]` |
| `/task_status` | out | 进度字符串 |
| `/task_info` | out | 解析后的任务摘要 |

## 编译

```bash
cd ~/dev_ws/yzbot
source /opt/ros/humble/setup.bash
colcon build --packages-select yzbot_task --symlink-install
source install/setup.bash
```

## 先测解析（不用仿真）

```bash
source /opt/ros/humble/setup.bash
source ~/dev_ws/yzbot/install/setup.bash
python3 -m pytest ~/dev_ws/yzbot/src/yzbot_task/test -q
```

## 干跑状态机（只打印步骤）

```bash
# 终端 A
ros2 run yzbot_task task_sm --ros-args -p dry_run:=true

# 终端 B：发任务（推荐 topic pub，避免参数把 [] 当成数组）
ros2 topic pub --once /task_command std_msgs/msg/String "{data: '[red,1,A]'}"
```

## 仿真里跑 1 个红块到 A（官方 v2：Gazebo world2 + Nav2 gazebo2 + 定位）

```bash
# 终端1
ros2 launch mybot gazebo_world2.launch.py
# 终端2
ros2 launch bot_navigation nav_bringup_gazebo2.launch.py
# 终端3：状态机
ros2 run yzbot_task task_sm
# 终端4
ros2 topic pub --once /task_command std_msgs/msg/String "{data: '[red,1,A]'}"
```

放置区分三间（隔墙、开阔）：A 南厅 (5.4,-11.8)、B 中厅 (0.5,-4.2)、C 西侧 (-8.3,-6.5)。

若导航不稳，可先瞬移再只测臂+吸附：

```bash
cd ~/dev_ws/yzbot/src/phase_a
bash teleport_near_red.sh
bash set_initial_pose.sh -0.7 1.0 3.1416
ros2 run yzbot_task task_sm --ros-args -p skip_nav:=true
ros2 topic pub --once /task_command std_msgs/msg/String "{data: '[red,1,A]'}"
```

## 本地解题节点（仍无 API）

```bash
ros2 run yzbot_task local_solver
ros2 topic pub --once /exam_question std_msgs/msg/String "{data: '题目: 每个衣柜有5件衣物;有T恤、裤子两种衣柜;小帅、小丽都需要7件T恤;小娜需要2件T恤,1条裤子;小杰需要1条裤子。设T恤衣柜数量为x,裤子衣柜数量为y,计算x、y分别为多少。'}"
ros2 topic echo /task_command --once
```
