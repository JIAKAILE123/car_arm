#!/usr/bin/env bash
# 比赛一键启动：Gazebo + Nav2 + coStudio转发 + task_sm + llm_solver + 出题
set -euo pipefail

WS="${HOME}/dev_ws/yzbot"
COSTUDIO_LAYOUT="${WS}/src/costudio_yzbot.json"
COSTUDIO_BIN="/opt/coStudio/honeybee"

if [[ ! -f "${WS}/install/setup.bash" ]]; then
  echo "找不到 ${WS}/install/setup.bash，请先："
  echo "  cd ${WS} && colcon build --symlink-install && source install/setup.bash"
  exit 1
fi

SHOW_ELAPSED="${SHOW_ELAPSED:-true}"
# 出题终端默认开启；需在该终端按【空格】才出题
RUN_EXAM="${RUN_EXAM:-true}"
# official=官方2题bin；practice=多套练习库(feed)；robot=直接发task_command测路径
EXAM_BANK="${EXAM_BANK:-official}"

# 每个终端先甩掉 conda，强制用系统 Python3.10（ROS Humble 需要；否则会报没有 numpy）
ros_env() {
  cat <<'EOS'
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER CONDA_PYTHON_EXE CONDA_SHLVL CONDA_EXE _CE_M _CE_CONDA
export PATH="/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin"
hash -r
source /opt/ros/humble/setup.bash
source "$HOME/dev_ws/yzbot/install/setup.bash"
export PATH="/usr/bin:$PATH"
hash -r
python3 -c "import numpy" >/dev/null 2>&1 || {
  echo "[错误] 系统 python3 缺少 numpy。请执行: sudo apt install -y python3-numpy"
  exit 1
}
EOS
}

open_tab() {
  local title="$1"
  local cmd="$2"
  if command -v gnome-terminal >/dev/null 2>&1; then
    gnome-terminal --tab --title="${title}" -- bash --noprofile --norc -lc \
      "$(ros_env); echo '=== ${title} ==='; echo \"python3=\$(which python3)\"; ${cmd}; echo; echo '[${title}] 已退出，按回车关闭'; read"
  else
    echo "[WARN] 无 gnome-terminal，请手动开终端跑：${cmd}"
  fi
}

echo "=========================================="
echo " yzbot 一键启动 (./run.sh)"
echo " 工作空间: ${WS}"
echo " 说明: 已避开 conda，使用 /usr/bin/python3"
echo " 总用时显示: ${SHOW_ELAPSED}  (正式赛: SHOW_ELAPSED=false ./run.sh)"
echo " 出题终端: ${RUN_EXAM}  (在 6-Exam 终端按【空格】出题)"
echo "=========================================="

# 本脚本自身先检查系统 python（勿在 set -u 下 source ROS setup）
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER CONDA_PYTHON_EXE CONDA_SHLVL CONDA_EXE || true
export PATH="/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin"
hash -r
if ! /usr/bin/python3 -c "import numpy" >/dev/null 2>&1; then
  echo "[错误] 系统 python3 缺少 numpy。请执行: sudo apt install -y python3-numpy"
  exit 1
fi
echo "当前 python3: $(/usr/bin/which python3)  ($(/usr/bin/python3 -V 2>&1))"

# ① Gazebo（等世界加载 + spawn 小车后再开后续）
open_tab "1-Gazebo" "ros2 launch mybot gazebo_world2.launch.py"
echo "等待 Gazebo 加载世界并生成小车 …"; sleep 25

# ② Nav2（GT 定位自动对齐，无需 2D Pose Estimate）
open_tab "2-Nav2" "ros2 launch bot_navigation nav_bringup_gazebo2.launch.py"
echo "等待 Nav2 + GT 定位 …"; sleep 12

# ③ coStudio 转发 + viz
open_tab "3-Viz" "ros2 launch yzbot_task rosbridge_viz.launch.py show_elapsed:=${SHOW_ELAPSED}"
echo "等待 viz …"; sleep 5

# ④ 任务状态机
open_tab "4-TaskSM" "ros2 run yzbot_task task_sm"
echo "等待 task_sm …"; sleep 3

# ⑤ 大模型解题
open_tab "5-LLM" "ros2 run yzbot_task llm_solver"
echo "等待 llm_solver …"; sleep 3

# ⑥ 出题：等栈起来后开终端，按空格才出题
if [[ "${RUN_EXAM}" == "true" ]]; then
  case "${EXAM_BANK}" in
    practice|bank|multi)
      open_tab "6-Exam" "ros2 run yzbot_task practice_exam --ros-args -p mode:=feed"
      ;;
    robot|cmd|commands)
      open_tab "6-Exam" "ros2 run yzbot_task practice_exam --ros-args -p mode:=robot"
      ;;
    *)
      open_tab "6-Exam" "ros2 run yzbot_task exam_feeder --ros-args -p auto_once:=false"
      ;;
  esac
fi

sleep 2
if [[ -x "${COSTUDIO_BIN}" ]]; then
  echo "正在打开 coStudio …"
  nohup "${COSTUDIO_BIN}" >/tmp/costudio_yzbot.log 2>&1 &
else
  echo "[提示] 未找到 ${COSTUDIO_BIN}，请手动打开 coStudio"
fi

echo
echo "已启动各终端。接下来："
echo "  1) 定位已用 Gazebo 真值（gt_localization），一般不用再点 2D Pose Estimate"
echo "  2) coStudio → 打开连接 → ws://127.0.0.1:21274"
echo "  3) 导入布局：${COSTUDIO_LAYOUT}"
echo "  4) 切到「6-Exam」终端，按【空格】出题"
echo "     多套练习：EXAM_BANK=practice ./run.sh"
echo "     只测取放路径：EXAM_BANK=robot ./run.sh"
echo "     只验解题：ros2 run yzbot_task practice_exam --ros-args -p mode:=check -p backend:=local"
echo
echo "清场（卡住时）："
echo "  killall -9 gzserver gzclient 2>/dev/null; pkill -9 -f 'ros2 launch|task_sm|llm_solver|viz_bridge|rosbridge|nav2|amcl|exam_feeder|practice_exam' 2>/dev/null"
echo
