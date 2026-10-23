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
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="jiakaile",
    maintainer_email="jiakaile@todo.todo",
    description="Task parse + LLM solve + state machine + viz",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "llm_solver = yzbot_task.llm_solver_node:main",
            "exam_feeder = yzbot_task.exam_feeder:main",
            "practice_exam = yzbot_task.practice_exam:main",
            "viz_bridge = yzbot_task.viz_bridge:main",
            "task_sm = yzbot_task.task_sm:main",
            "arm_js_filter = yzbot_task.arm_js_filter:main",
            "dyn_obs_predict = yzbot_task.dyn_obs_predict:main",
            "gt_localization = yzbot_task.gt_localization:main",
            "route_planner = yzbot_task.route_planner:main",
        ],
    },
)
