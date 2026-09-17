# car_arm / yzbot — 大模型技术创新赛

---

## 比赛走一遍（按顺序开终端）

**每个新终端先执行这两行：**

```bash
source /opt/ros/humble/setup.bash
source ~/dev_ws/yzbot/install/setup.bash
```

然后按下面开（共 6 个终端）：

```bash
# ① Gazebo
ros2 launch mybot gazebo_world2.launch.py

# ② Nav2（起来后在 RViz 点一次 2D Pose Estimate）
ros2 launch bot_navigation nav_bringup_gazebo2.launch.py

# ③ coStudio 转发（测试阶段带总用时；正式比赛加 show_elapsed:=false）
# cobridge：优先用系统包；没有则用仓库内 third_party/cobridge_ros（已随仓库提供）
# 可选系统安装：sudo apt install ros-humble-cobridge（需先加 apt.coscene.cn 源）
ros2 launch yzbot_task rosbridge_viz.launch.py
# 打开 coStudio：/opt/coStudio/honeybee
# 打开连接 → ws://127.0.0.1:21274（不要用 9090）
# 导入布局：~/dev_ws/yzbot/src/costudio_yzbot.json
# 右侧 /viz/* 红字 = viz_bridge 没在发；确认本终端 launch 里有 viz_bridge ready

# ④ 任务状态机
ros2 run yzbot_task task_sm

# ⑤ 大模型解题
ros2 run yzbot_task llm_solver

# ⑥ 官方出题（跑一次就出题；也可反复再跑）
ros2 run yzbot_task exam_feeder
```

到这里：出题 → DeepSeek 解题 → `/task_command` → `task_sm` 抓放；coStudio 上看题目/数量/顺序/状态。

---

## 常用补充

**手动发一题 / 发任务（调试）：**

```bash
ros2 topic pub --once /exam_question std_msgs/msg/String \
  "{data: '题目: 每个衣柜有5件衣物;有T恤、裤子两种衣柜;小美需要2件T恤,1条裤子;小帅、小丽都需要7件T恤;小娜需要2件T恤,1条裤子;小杰需要1条裤子。设T恤衣柜数量为x,裤子衣柜数量为y,计算x、y分别为多少。'}"

ros2 topic pub --once /task_command std_msgs/msg/String "{data: '[red,1,A]'}"
```

**赛前多测解题（12 题，本地+DeepSeek）：**

```bash
ros2 run yzbot_task practice_exam --ros-args -p backend:=both
```

**改代码后编译：**

```bash
cd ~/dev_ws/yzbot
colcon build --packages-select yzbot_task mybot mybot_description bot_navigation --symlink-install
source install/setup.bash
```

**DeepSeek 密钥（只需配一次）：**

```bash
mkdir -p ~/.config/yzbot
echo 'sk-你的密钥' > ~/.config/yzbot/deepseek_api_key
```

**卡住时清进程：**

```bash
killall -9 gzserver gzclient 2>/dev/null
pkill -9 -f 'ros2 launch|task_sm|llm_solver|viz_bridge|rosbridge|nav2|amcl' 2>/dev/null
```

---

## 话题速查

| 话题 | 作用 |
|------|------|
| `/exam_question` | 出题 |
| `/llm_xy` | x,y |
| `/task_command` | `[red,4,A;blue,1,C]` |
| `/task_plan` | 抓取顺序 |
| `/task_status` | 进度 |
| `/viz/*` `/chat` | coStudio 面板 |

默认映射：`x→红`，`y→蓝`；数量 ≥3 → A，否则 → C。
