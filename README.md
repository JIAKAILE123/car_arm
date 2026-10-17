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
1. 定位已用 Gazebo 真值（`gt_localization`），**不用**再点 2D Pose Estimate
2. coStudio → 打开连接 → `ws://127.0.0.1:21274`
3. **重新导入布局**：`~/dev_ws/yzbot/src/costudio_yzbot.json`
4. **两种出题方式（任选其一，都走大模型解题）**：
   - **① 空格出题**：切到「6-Exam」终端，按【空格】发下一题
   - **② 粘贴解题**：右侧「②粘贴题目解题」贴入赛题 → 点 **提交题目→大模型解题**

正式比赛隐藏总用时：`SHOW_ELAPSED=false ./run.sh`  
不开出题终端：`RUN_EXAM=false ./run.sh`

### 多套题练习（官方 bin 只有 2 题）

```bash
# 验解题：本地秒过全部边界题（ceil / 都需要 / 区映射 AA·AC·CA·CC）
ros2 run yzbot_task practice_exam --ros-args -p mode:=check -p backend:=local

# 整机按空格出下一套自然语言题（走大模型）
EXAM_BANK=practice ./run.sh

# 整机按空格直接发 /task_command（跳过解题，专测导航取放组合）
EXAM_BANK=robot ./run.sh
```

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

正式赛约束：`x≥1`、`y≥1` 且 `x+y=5`。  
默认映射：`x→红`，`y→蓝`；数量 **≥3 → A**，否则 → C（门槛是 ≥3，不是 >0 / >1）。


