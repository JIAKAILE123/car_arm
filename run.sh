#!/usr/bin/env bash
# 比赛一键启动（4 个终端标签 + 本窗口出题）
#   1-Gazebo  世界 + 小车
#   2-Nav2    Nav2 导航栈（GT 定位自动对齐）
#   3-Viz     可视化（coStudio 转发 cobridge + viz_bridge）与 llm_solver
#   4-TaskSM  任务状态机
#   本窗口     出题（出完即退回，不用再留一个终端）
set -euo pipefail

WS="${HOME}/dev_ws/yzbot"
COSTUDIO_BIN="/opt/coStudio/honeybee"

if [[ ! -f "${WS}/install/setup.bash" ]]; then
  echo "找不到 ${WS}/install/setup.bash，请先："
  echo "  cd ${WS} && colcon build --symlink-install && source install/setup.bash"
  exit 1
fi

# 布局：练习版(多 抓取顺序/总用时) / 正式版(严格按评分项：指令解析 7 项 + 状态反馈 3 项 + 移动路径 + 相机)
SHOW_ELAPSED="${SHOW_ELAPSED:-true}"
if [[ "${SHOW_ELAPSED}" == "true" ]]; then
  COSTUDIO_LAYOUT="${WS}/src/costudio_yzbot_practice.json"
  LAYOUT_NAME="练习版（正式 12 项 + 抓取顺序 / 总用时）"
else
  COSTUDIO_LAYOUT="${WS}/src/costudio_yzbot.json"
  LAYOUT_NAME="正式版（严格 12 项，无测试用面板）"
fi

# 国赛映射规则（裁判赛前从三条中随机挑一条）：
#   RULE_ID=1  x=红、y=蓝；数量≥3→A区，否则→C区
#   RULE_ID=2  数值≥3为红、否则蓝；x→B区、y→A区
#   RULE_ID=3  数值大者蓝、小者红；蓝→B区、红→C区
# 也可 RULE_TEXT="<规则原文>" 复制粘贴，自动识别编号（RULE_ID 显式指定时优先）
RULE_ID="${RULE_ID:-1}"
RULE_TEXT="${RULE_TEXT:-}"

RUN_EXAM="${RUN_EXAM:-true}"
EXAM_ARG="${EXAM_ARG:-}"
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
      "$(ros_env); echo '=== ${title} ==='; ${cmd}; echo; echo '[${title}] 已退出，按回车关闭'; read"
  else
    echo "[WARN] 无 gnome-terminal，请手动开终端跑：${cmd}"
  fi
}

echo "=========================================="
echo " yzbot 一键启动 (./run.sh)"
echo " 工作空间: ${WS}   python: /usr/bin/python3 (已避开 conda)"
echo " 可视化布局: ${LAYOUT_NAME}"
echo "             ${COSTUDIO_LAYOUT}"
echo " 终端: 1-Gazebo  2-Nav2  3-Viz+LLM  4-TaskSM   （出题就在本窗口）"
echo " 正式赛命令: SHOW_ELAPSED=false ./run.sh"
echo "=========================================="

# 本脚本自身先检查系统 python（勿在 set -u 下 source ROS setup）
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER CONDA_PYTHON_EXE CONDA_SHLVL CONDA_EXE || true
export PATH="/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin"
hash -r
if ! /usr/bin/python3 -c "import numpy" >/dev/null 2>&1; then
  echo "[错误] 系统 python3 缺少 numpy。请执行: sudo apt install -y python3-numpy"
  exit 1
fi

# ① Gazebo（等世界加载 + spawn 小车后再开后续）
open_tab "1-Gazebo" "ros2 launch mybot gazebo_world2.launch.py"
echo "[1/4] 等待 Gazebo 加载世界并生成小车 …"; sleep 25

# ② Nav2（GT 定位自动对齐，无需 2D Pose Estimate）
open_tab "2-Nav2" "ros2 launch bot_navigation nav_bringup_gazebo2.launch.py"
echo "[2/4] 等待 Nav2 + GT 定位 …"; sleep 12

# ③ 可视化转发 cobridge + viz_bridge，同一个标签里再起 llm_solver
open_tab "3-Viz+LLM" "ros2 launch yzbot_task rosbridge_viz.launch.py show_elapsed:=${SHOW_ELAPSED} & sleep 3; ros2 run yzbot_task llm_solver --ros-args -p rule_id:=${RULE_ID} -p rule_text:=\"${RULE_TEXT}\""
echo "[3/4] 等待可视化与大模型解题 …"; sleep 5

# ④ 任务状态机
open_tab "4-TaskSM" "ros2 run yzbot_task task_sm"
echo "[4/4] 等待任务状态机 …"; sleep 3

# 打开 coStudio（可视化客户端）
if [[ -x "${COSTUDIO_BIN}" ]]; then
  echo "正在打开 coStudio …"
  mkdir -p "${HOME}/dev_ws/.roslog"
  nohup "${COSTUDIO_BIN}" >"${HOME}/dev_ws/.roslog/costudio.log" 2>&1 &
else
  echo "[提示] 未找到 ${COSTUDIO_BIN}，请手动打开 coStudio"
fi

echo
echo "=========================================="
echo " 先在 coStudio 里连 ws://127.0.0.1:21274 ，再导入布局："
echo "   ${COSTUDIO_LAYOUT}"
echo " 评分 12 项：题目种子 / 规则题目 / 计算题目 / 红色数量 / 红色区域 /"
echo "            蓝色数量 / 蓝色区域 / 移动路径 / 任务进度 / 当前任务 /"
echo "            当前状态 / 相机画面"
echo " 赛前输入面板：映射规则输入 / 出题参数输入 / 题目输入 / 题目种子输入"
echo "=========================================="

# ⑤ 出题：直接在本窗口出，出完就退回，不再单独占一个终端
if [[ "${RUN_EXAM}" == "true" ]]; then
  source /opt/ros/humble/setup.bash
  source "${WS}/install/setup.bash"
  export PATH="/usr/bin:$PATH"
  case "${EXAM_BANK}" in
    practice|bank|multi)
      echo "→ 练习题库出题（mode=feed）"
      ros2 run yzbot_task practice_exam --ros-args -p mode:=feed || true
      ;;
    robot|cmd|commands)
      echo "→ 直接下发任务指令测试路径（mode=robot）"
      ros2 run yzbot_task practice_exam --ros-args -p mode:=robot || true
      ;;
    *)
      ARG="${EXAM_ARG}"
      if [[ -n "${ARG}" ]]; then
        # 预置参数（EXAM_ARG=11470507 ./run.sh）：直接出题，不等面板
        ros2 run yzbot_task exam_feeder --ros-args -p exam_arg:="${ARG}" || true
      else
        # 面板模式：出题参数在可视化面板里输入/粘贴后点按钮（比赛 11470507；
        # 留空=练习模式；可反复发送重新出题），节点后台常驻
        echo
        echo "→ 面板模式出题（exam_feeder 后台常驻）："
        echo "    在 coStudio 面板「出题参数输入」里填参数后点按钮："
        echo "      比赛模式 → 11470507（题目每 30 分钟变更一次）"
        echo "      练习模式 → 留空（清空后再点一次）"
        echo "    日志：~/.roslog/exam_feeder.log"
        nohup ros2 run yzbot_task exam_feeder --ros-args -p panel_mode:=true \
          >"${HOME}/dev_ws/.roslog/exam_feeder.log" 2>&1 &
      fi
      ;;
  esac
  echo "→ 出题完成，任务已在 4-TaskSM 开始执行。"
else
  echo "[提示] RUN_EXAM=false：未自动出题。可手动执行："
  echo "   ros2 run yzbot_task exam_feeder --ros-args -p panel_mode:=true   # 面板输入"
  echo "   ros2 run yzbot_task exam_feeder --ros-args -p exam_arg:=11470507  # 比赛模式"
  echo "   映射规则: RULE_ID=2 ./run.sh（或 RULE_TEXT=\"<规则原文>\" 自动识别；"
  echo "             也可在面板「映射规则输入」里粘贴规则原文）"
fi

echo
echo "收工/清场： ./stop.sh"