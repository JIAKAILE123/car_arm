# 阶段 A — 基线可复现

目标：冷启动后，**导航到点 + 臂到位 + 吸放 1 个方块**可重复成功。不做 LLM、不做状态机。

## 本阶段用什么「算法 / 方案」

| 环节 | 选用 | 说明 |
|------|------|------|
| 定位 | Nav2 **AMCL** | 粒子滤波；开局必须 2D Pose Estimate |
| 导航 | Nav2 现成栈（`originbot_nav2.yaml`） | 全局规划 + 局部控制器，阶段 A 不改参 |
| 臂运动 | **开环关节轨迹** `FollowJointTrajectory` | 绕开 MoveIt / `left_wheel_joint` 不匹配 |
| 抓取 | Gazebo **LinkAttacher** | 仿真假吸合；末端靠近后再 `cube_control on` |
| 可视化接线 | **rosbridge** + 稍后 coStudio | 阶段 A 先保证 bridge 能起；面板放到阶段 D |

阶段 B 再谈：LLM 提示词与解析、色块检测、任务状态机。

## 启动顺序（必按序 · 官方 v2 地图）

开 4 个终端（或直接跑 `./start_stack.sh`）：

1. Gazebo 全车：`ros2 launch mybot gazebo_world2.launch.py`（`offic_room` + mapn3）
2. MoveIt/RViz（可选）：`ros2 launch mybot my_moveit_rviz.launch.py`
3. Nav2：`ros2 launch bot_navigation nav_bringup_gazebo2.launch.py`
4. rosbridge：`ros2 launch rosbridge_server rosbridge_websocket_launch.xml`

> 旧的 `gazebo_world.launch.py` / `nav_bringup_gazebo.launch.py`（room+map.yaml）仅作早期练习，赛场对齐 **world2**。

每个终端先：

```bash
source /opt/ros/humble/setup.bash
source ~/dev_ws/yzbot/install/setup.bash
```

## 开局清单（控制器 + 定位）

```bash
cd ~/dev_ws/yzbot/src/phase_a
bash check_ready.sh
```

通过后：

1. 在 **Nav2 的 RViz** 里点 **2D Pose Estimate**（机器人大致位姿与朝向）
2. 或：`bash set_initial_pose.sh`（默认原点朝向，不对再改参数重发）

## 单次吸放验收

红块约在 `(-1.5, 1.0, 0.75)`，车默认在原点，**必须先到红块旁再吸**。

推荐（不依赖导航，阶段 A 更快）：

```bash
bash teleport_near_red.sh
bash set_initial_pose.sh -0.7 1.0 3.1416
bash demo_pick_one.sh red 1
```

或用导航：`bash nav_near_red1.sh` 后再 `demo_pick_one.sh`。

Gazebo 找不到红块时：左侧 **Models** 点 `red_cube_1`，窗口会高亮该模型。

成功标准：Gazebo 里红块 1 随末端抬起，松开后落下。

## 常见问题

| 现象 | 处理 |
|------|------|
| `map->odom` 没有 / 导航不动 | 先做 2D Pose Estimate |
| `invalid choice: control` | 装 `ros-humble-ros2controlcli` |
| `/ATTACHLINK` 不存在 | 确认用的是 `room.world`（已含 link_attacher 插件）且 `ros2_linkattacher` 已编译进工作空间 |
| MoveIt Plan 报 `left_wheel_joint` | 已知问题；阶段 A 用脚本控臂即可 |
