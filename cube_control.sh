#!/bin/bash

# 方块抓取与释放控制脚本
# 用法: ./cube_control.sh [on|off] [red|blue] [1-5]

if [ $# -ne 3 ]; then
    echo "错误: 参数数量不正确!"
    echo "用法: $0 [on|off] [red|blue] [1-5]"
    exit 1
fi

ACTION=$1
COLOR=$2
ID=$3

# 验证方块颜色
if [[ "$COLOR" != "red" && "$COLOR" != "blue" ]]; then
    echo "错误: 颜色参数必须是 'red' 或 'blue'"
    exit 1
fi

# 验证方块ID
if ! [[ "$ID" =~ ^[1-5]$ ]]; then
    echo "错误: 方块ID必须是 1-5 的数字"
    exit 1
fi

# 构造方块名称
CUBE_NAME="${COLOR}_cube_${ID}"

# 执行服务调用
case $ACTION in
    on)
        SERVICE="/ATTACHLINK"
        SRT_TYPE="linkattacher_msgs/srv/AttachLink"
        REQUEST_DATA="{model1_name: 'six_arm', link1_name: 'link6', model2_name: '$CUBE_NAME', link2_name: 'link'}"
        ;;
    off)
        SERVICE="/DETACHLINK"
        SRT_TYPE="linkattacher_msgs/srv/DetachLink"
        REQUEST_DATA="{model1_name: 'six_arm', link1_name: 'link6', model2_name: '$CUBE_NAME', link2_name: 'link'}"
        ;;
    *)
        echo "错误: 操作类型必须是 'on' 或 'off'"
        exit 1
        ;;
esac

echo "执行: $ACTION $COLOR 方块 $ID"
ros2 service call $SERVICE $SRT_TYPE "$REQUEST_DATA"
