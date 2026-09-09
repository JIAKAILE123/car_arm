"""MoveIt IK (compute_ik) + arm FollowJointTrajectory execute.

Uses arm-only MoveIt model; wheels never enter the request.
Falls back clearly when move_group /compute_ik is down.
"""

from __future__ import annotations

import time
from typing import List, Optional, Sequence

from geometry_msgs.msg import PoseStamped
from moveit_msgs.srv import GetPositionIK
from moveit_msgs.msg import PositionIKRequest, RobotState, MoveItErrorCodes
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration as MsgDuration
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.node import Node


ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]


class MoveItArm:
    def __init__(self, node: Node, arm_action: ActionClient):
        self.node = node
        self.arm = arm_action
        self.ik_cli = node.create_client(GetPositionIK, "/compute_ik")
        self._last_js: Optional[JointState] = None
        node.create_subscription(
            JointState, "/yzbot/arm_joint_states", self._on_js, 10
        )
        # also accept raw in case filter not up yet
        node.create_subscription(JointState, "/joint_states", self._on_js_raw, 10)

    def _on_js(self, msg: JointState):
        self._last_js = msg

    def _on_js_raw(self, msg: JointState):
        if self._last_js is None:
            self._last_js = msg

    def wait_ready(self, timeout: float = 15.0) -> bool:
        ok = self.ik_cli.wait_for_service(timeout_sec=timeout)
        if not ok:
            self.node.get_logger().error(
                "/compute_ik unavailable — start: ros2 launch mybot move_group_headless.launch.py"
            )
        return ok

    def _spin_until(self, future, timeout_sec: float) -> bool:
        start = time.time()
        while not future.done():
            if time.time() - start > timeout_sec:
                return False
            time.sleep(0.05)
        return True

    def _seed_state(self) -> RobotState:
        rs = RobotState()
        rs.is_diff = True
        if self._last_js is not None:
            js = JointState()
            for name in ARM_JOINTS:
                if name in self._last_js.name:
                    i = self._last_js.name.index(name)
                    js.name.append(name)
                    if i < len(self._last_js.position):
                        js.position.append(self._last_js.position[i])
            rs.joint_state = js
            rs.is_diff = False
        return rs

    def compute_ik(
        self,
        pose: PoseStamped,
        timeout: float = 5.0,
        *,
        try_orientations: bool = True,
    ) -> Optional[List[float]]:
        from copy import deepcopy
        from yzbot_task.vision_pose import _quat_from_rpy
        import math

        candidates = [pose]
        if try_orientations:
            for pitch in (math.pi / 2.0, -math.pi / 2.0, math.pi / 2.0 + 0.3):
                for yaw in (0.0, 0.4, -0.4, 0.8, -0.8):
                    p = deepcopy(pose)
                    p.pose.orientation = _quat_from_rpy(0.0, pitch, yaw)
                    candidates.append(p)

        last_code = None
        for p in candidates:
            req = GetPositionIK.Request()
            ik = PositionIKRequest()
            ik.group_name = "arm"
            ik.ik_link_name = "grasping_frame"
            ik.pose_stamped = p
            ik.pose_stamped.header.frame_id = p.header.frame_id or "arm_base_link"
            ik.avoid_collisions = False
            ik.timeout.sec = 0
            ik.timeout.nanosec = 200000000  # 0.2 s
            ik.robot_state = self._seed_state()
            req.ik_request = ik

            fut = self.ik_cli.call_async(req)
            if not self._spin_until(fut, timeout):
                continue
            res = fut.result()
            if res is None or res.error_code.val != MoveItErrorCodes.SUCCESS:
                last_code = getattr(res.error_code, "val", None) if res else None
                continue
            names = list(res.solution.joint_state.name)
            pos = list(res.solution.joint_state.position)
            out = []
            ok = True
            for j in ARM_JOINTS:
                if j not in names:
                    ok = False
                    break
                out.append(float(pos[names.index(j)]))
            if ok:
                return out
        self.node.get_logger().warn(f"IK failed code={last_code} (tried {len(candidates)} orients)")
        return None

    def execute_joints(
        self, joints: Sequence[float], duration_sec: float = 4.0, timeout: float = 60.0
    ) -> bool:
        traj = JointTrajectory()
        traj.joint_names = list(ARM_JOINTS)
        # start from current if known
        if self._last_js is not None:
            cur = []
            ok = True
            for j in ARM_JOINTS:
                if j not in self._last_js.name:
                    ok = False
                    break
                i = self._last_js.name.index(j)
                cur.append(float(self._last_js.position[i]))
            if ok:
                pt0 = JointTrajectoryPoint()
                pt0.positions = cur
                pt0.time_from_start = MsgDuration(sec=0, nanosec=100000000)
                traj.points.append(pt0)
        pt = JointTrajectoryPoint()
        pt.positions = [float(x) for x in joints]
        pt.time_from_start = MsgDuration(sec=int(duration_sec), nanosec=0)
        traj.points.append(pt)

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj
        send = self.arm.send_goal_async(goal)
        if not self._spin_until(send, 30.0):
            return False
        handle = send.result()
        if not handle.accepted:
            return False
        result = handle.get_result_async()
        return self._spin_until(result, timeout)

    def move_to_pose(
        self, pose: PoseStamped, duration_sec: float = 4.0
    ) -> bool:
        """IK + execute (MoveIt planning via compute_ik)."""
        self.node.get_logger().info(
            f"MoveIt IK → ({pose.pose.position.x:.3f},"
            f"{pose.pose.position.y:.3f},{pose.pose.position.z:.3f}) "
            f"in {pose.header.frame_id}"
        )
        q = self.compute_ik(pose)
        if q is None:
            return False
        return self.execute_joints(q, duration_sec=duration_sec)
