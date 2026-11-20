#!/usr/bin/env python3
"""viz_bridge + coBridge WebSocket for coStudio (ws://127.0.0.1:21274).

Prefer system `ros-humble-cobridge` if installed; otherwise use vendored binary at
~/dev_ws/yzbot/third_party/cobridge_ros (no apt required).
"""

from __future__ import annotations

import os
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _cobridge_root() -> Path | None:
    env = os.environ.get("YZBOT_COBRIDGE_ROOT", "").strip()
    if env:
        p = Path(env)
        if (p / "lib" / "cobridge" / "cobridge").is_file():
            return p
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        cand = parent / "third_party" / "cobridge_ros"
        if (cand / "lib" / "cobridge" / "cobridge").is_file():
            return cand
    home = Path.home() / "dev_ws" / "yzbot" / "third_party" / "cobridge_ros"
    if (home / "lib" / "cobridge" / "cobridge").is_file():
        return home
    return None


def _system_cobridge_available() -> bool:
    """Only treat apt/system installs as 'system' (ignore leftover /tmp overlays)."""
    try:
        from ament_index_python.packages import get_package_prefix

        prefix = Path(get_package_prefix("cobridge")).resolve()
        return str(prefix).startswith("/opt/ros/")
    except Exception:
        return False


def _launch_setup(context, *args, **kwargs):
    use_sim_time = LaunchConfiguration("use_sim_time")
    port = LaunchConfiguration("port")
    show_elapsed = LaunchConfiguration("show_elapsed")

    actions = []

    if _system_cobridge_available():
        actions.append(
            Node(
                package="cobridge",
                executable="cobridge",
                name="cobridge",
                output="screen",
                parameters=[
                    {
                        "use_sim_time": use_sim_time,
                        "port": port,
                        "address": "0.0.0.0",
                        "send_buffer_limit": 10000000,
                    }
                ],
            )
        )
    else:
        root = _cobridge_root()
        if root is None:
            raise RuntimeError(
                "package 'cobridge' not found and vendored binary missing.\n"
                "  要么: sudo apt install ros-humble-cobridge\n"
                "  要么: 确认存在 ~/dev_ws/yzbot/third_party/cobridge_ros/lib/cobridge/cobridge"
            )
        bin_path = root / "lib" / "cobridge" / "cobridge"
        port_val = port.perform(context)
        sim_val = use_sim_time.perform(context)
        ld = f"{root / 'lib'}:{os.environ.get('LD_LIBRARY_PATH', '')}"
        ament = f"{root}:{os.environ.get('AMENT_PREFIX_PATH', '')}"
        actions.append(
            ExecuteProcess(
                cmd=[
                    str(bin_path),
                    "--ros-args",
                    "-r",
                    "__node:=cobridge",
                    "-p",
                    f"port:={port_val}",
                    "-p",
                    "address:=0.0.0.0",
                    "-p",
                    f"use_sim_time:={sim_val}",
                    "-p",
                    "send_buffer_limit:=10000000",
                ],
                additional_env={
                    "LD_LIBRARY_PATH": ld,
                    "AMENT_PREFIX_PATH": ament,
                },
                output="screen",
                name="cobridge",
            )
        )

    actions.append(
        Node(
            package="yzbot_task",
            executable="viz_bridge",
            name="viz_bridge",
            output="screen",
            parameters=[
                {
                    "use_sim_time": use_sim_time,
                    "show_elapsed": ParameterValue(
                        show_elapsed, value_type=bool
                    ),
                }
            ],
        )
    )
    return actions


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "use_sim_time",
                default_value="true",
                description="Match Gazebo /clock",
            ),
            DeclareLaunchArgument(
                "port",
                default_value="21274",
                description="coBridge WebSocket port for coStudio",
            ),
            DeclareLaunchArgument(
                "show_elapsed",
                default_value="true",
                description="测试总用时面板；正式比赛改 false",
            ),
            OpaqueFunction(function=_launch_setup),
        ]
    )
