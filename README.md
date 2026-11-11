# yzbot — 大模型技术创新赛

ROS2 Humble + Gazebo + Nav2 + OpenCV + DeepSeek。一键启动整机取放。

## 编译

请用系统 Python（不要在 conda 环境里编）：

```bash
cd ~/dev_ws/yzbot
# 若已进 conda：conda deactivate
colcon build --symlink-install --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3
source install/setup.bash
```

`third_party/cobridge` 是预编译二进制，已放 `COLCON_IGNORE`，不要拿它当源码包编译。

## 启动

```bash
./run.sh
```

正式赛隐藏用时：`SHOW_ELAPSED=false ./run.sh`

启动后：

1. 定位使用 Gazebo 真值，无需 2D Pose Estimate  
2. coStudio 连接 `ws://127.0.0.1:21274`  
3. 导入布局：`src/costudio_yzbot.json`  
4. 出题：在「6-Exam」终端输入参数后回车（比赛模式 `11470507`），或在面板粘贴题目提交  

## DeepSeek 密钥

```bash
mkdir -p ~/.config/yzbot
echo 'sk-你的密钥' > ~/.config/yzbot/deepseek_api_key
```

## 主要包

| 包 | 作用 |
| --- | --- |
| `yzbot_task` | 解题、主控、视觉、预测、可视化 |
| `mybot` / `mybot_description` | 车体、世界、控制器 |
| `bot_navigation` | Nav2 |
| `yzbot_gazebo_plugins` | 动障插件 |
| `IFRA_LinkAttacher` | 吸附 / 区垫固定 |

上传前请先在本机重新编译：`colcon build --symlink-install && source install/setup.bash`。
