from setuptools import setup
from setuptools import find_packages
import os
from glob import glob

package_name = "yzbot_task"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "config"), glob("config/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="jiakaile",
    maintainer_email="jiakaile@todo.todo",
    description="Task parse + local solve + state machine",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "mock_command = yzbot_task.mock_command:main",
            "local_solver = yzbot_task.local_solver_node:main",
            "task_sm = yzbot_task.task_sm:main",
            "cam_fpv = yzbot_task.cam_fpv:main",
            "arm_js_filter = yzbot_task.arm_js_filter:main",
            "dyn_obstacles = yzbot_task.dyn_obstacles:main",
            "waypoint_probe = yzbot_task.waypoint_probe:main",
            "costmap_ghost_clear = yzbot_task.costmap_ghost_clear:main",
            "export_cube_poses = yzbot_task.export_cube_poses:main",
            "route_planner = yzbot_task.route_planner:main",
        ],
    },
)
