# car_arm / yzbot — 大模型技术创新赛

技术文档：[`docs/技术文档.md`](docs/技术文档.md)

---

## 一键启动

```bash
cd ~/dev_ws/yzbot
./run.sh
```

会自动打开多个终端：Gazebo、Nav2、coStudio 转发、task_sm、llm_solver、出题，并尝试启动 coStudio。  
`run.sh` 会自动避开 conda，改用系统 `/usr/bin/python3`（否则常见 `No module named 'numpy'`，小车也 spawn 不出来）。

启动后：
1. Nav2 的 RViz 里点一次 **2D Pose Estimate**
2. coStudio → 打开连接 → `ws://127.0.0.1:21274`
3. 导入布局：`~/dev_ws/yzbot/src/costudio_yzbot.json`（桌面也有一份）
4. 切到 **6-Exam** 终端，按 **空格** 出题（可多次）

正式比赛隐藏总用时：`SHOW_ELAPSED=false ./run.sh`  
不开出题终端：`RUN_EXAM=false ./run.sh`

---

## 改代码后编译

```bash
cd ~/dev_ws/yzbot
colcon build --packages-select yzbot_task mybot mybot_description bot_navigation --symlink-install
source install/setup.bash
```

## DeepSeek 密钥（只需配一次）

```bash
mkdir -p ~/.config/yzbot
echo 'sk-你的密钥' > ~/.config/yzbot/deepseek_api_key
```

## 卡住时清进程

```bash
killall -9 gzserver gzclient 2>/dev/null
pkill -9 -f 'ros2 launch|task_sm|llm_solver|viz_bridge|rosbridge|nav2|amcl' 2>/dev/null
```

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
