"""Publish short-horizon predicted dyn_obs footprints → costmap keepout.

Gazebo plugin moves bars with known sine law; laser only sees *now*, so the
robot can plan through a gap that the bar will occupy a second later.
This node marks current + future poses (~2.5s) on /yzbot/dyn_obs_predict.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header


# Must match yzbot_gazebo_plugins/src/dyn_obs_mover.cpp
# axis 'x': x = mid + amp*sin, y fixed
# axis 'y': y = mid + amp*sin, x fixed
# tuple: (axis, fixed, mid, amp, phase, period, half_x, half_y)
# 上杆东端贴近 +X 墙：预测 footprint 勿画满到墙缝，否则转弯卡死
_OBS = (
    ("x", -1.35, -0.80, 1.95, 0.0, 31.0, 0.22, 0.22),
    ("x", -6.15, -0.65, 1.65, 1.2, 33.67, 0.22, 0.22),
    ("y", -2.00, 0.00, 2.50, 2.4, 45.0, 0.22, 0.22),
)
# 上杆(y=-1.35)东侧墙约 x=1.80：预测中心 x 上限留缝给转弯
_OBS1_X_CLIP = 0.90


class DynObsPredict(Node):
    def __init__(self) -> None:
        super().__init__("dyn_obs_predict")
        # 仅加强杆预测；不改 DWB/膨胀（避免物块旁空转）
        self.declare_parameter("horizon_sec", 1.6)
        self.declare_parameter("dt_sec", 0.22)
        self.declare_parameter("rate_hz", 10.0)
        self.declare_parameter("step_m", 0.08)
        self._pub = self.create_publisher(PointCloud2, "/yzbot/dyn_obs_predict", 1)
        hz = max(2.0, float(self.get_parameter("rate_hz").value))
        self.create_timer(1.0 / hz, self._tick)
        self.get_logger().info(
            f"dyn_obs_predict → /yzbot/dyn_obs_predict "
            f"horizon={float(self.get_parameter('horizon_sec').value):.1f}s "
            f"n_obs={len(_OBS)}"
        )

    def _sim_t(self) -> float:
        return float(self.get_clock().now().nanoseconds) * 1e-9

    def _footprint(
        self, cx: float, cy: float, hx: float, hy: float, step: float
    ) -> List[Tuple[float, float, float]]:
        pts: List[Tuple[float, float, float]] = []
        x = cx - hx
        while x <= cx + hx + 1e-9:
            y = cy - hy
            while y <= cy + hy + 1e-9:
                pts.append((float(x), float(y), 0.08))
                y += step
            x += step
        return pts

    def _tick(self) -> None:
        t0 = self._sim_t()
        horizon = max(0.5, float(self.get_parameter("horizon_sec").value))
        dt = max(0.1, float(self.get_parameter("dt_sec").value))
        step = max(0.06, float(self.get_parameter("step_m").value))
        pts: List[Tuple[float, float, float]] = []
        n = int(horizon / dt) + 1
        for oi, (axis, fixed, mid, amp, phase, period, hx, hy) in enumerate(
            _OBS
        ):
            for i in range(n):
                t = t0 + i * dt
                s = mid + amp * math.sin(
                    2.0 * math.pi * t / float(period) + phase
                )
                if axis == "y":
                    x, y = float(fixed), float(s)
                else:
                    x, y = float(s), float(fixed)
                    # 上杆：别把预测画进东墙缝，留转弯通道
                    if oi == 0:
                        x = min(x, float(_OBS1_X_CLIP))
                pts.extend(self._footprint(x, y, hx, hy, step))
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "map"
        msg = point_cloud2.create_cloud(
            header,
            [
                PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            ],
            pts,
        )
        self._pub.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DynObsPredict()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
