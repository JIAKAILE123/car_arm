"""Global shortest-path task planner for random LLM task strings.

Policy: freely reorder pick→place legs across colors/zones as long as each
required (color count → zone) is fulfilled. Assigns cube ids + visit order to
minimize Euclidean tour:

  start → [door vias] → cube → zone → cube → zone → …

Door vias for red#1 → A/B match task_sm.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from yzbot_task.command_parser import PickTask, format_command, parse_command
from yzbot_task.vision_pose import known_cube_xy

Point = Tuple[float, float]
Leg = Tuple[str, int, str]  # color, cube_id, zone


@dataclass
class JobSpec:
    color: str
    zone: str
    fixed_id: Optional[int] = None


@dataclass
class PlanResult:
    legs: List[Leg]
    total_m: float
    start: Point
    detail: List[str]

    def as_command(self) -> str:
        tasks = [
            PickTask(color=c, count=1, zone=z, cube_id=i) for c, i, z in self.legs
        ]
        return format_command(tasks)

    def summary(self) -> str:
        lines = [
            f"ROUTE_PLAN total≈{self.total_m:.1f}m "
            f"start=({self.start[0]:.2f},{self.start[1]:.2f})",
            f"  cmd: {self.as_command()}",
        ]
        for i, (c, cid, z) in enumerate(self.legs, 1):
            extra = self.detail[i - 1] if i - 1 < len(self.detail) else ""
            lines.append(f"  {i}. {c}#{cid} → {z}  {extra}")
        return "\n".join(lines)


def _dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _load_zones(poses_yaml: Optional[Path] = None) -> Dict[str, Point]:
    zones = {"A": (5.40, -11.80), "B": (0.50, -4.20), "C": (-0.77, 2.71)}
    path = poses_yaml or (
        Path(__file__).resolve().parents[1] / "config" / "poses.yaml"
    )
    if path.is_file():
        try:
            import yaml

            data = yaml.safe_load(path.read_text()) or {}
            for letter in ("A", "B", "C"):
                key = f"zone_{letter.lower()}"
                if key in data and "x" in data[key]:
                    zones[letter] = (float(data[key]["x"]), float(data[key]["y"]))
        except Exception:
            pass
    return zones


def _cube_xy(color: str, cid: int) -> Point:
    x, y, _z = known_cube_xy(color, cid)
    return (float(x), float(y))


def _door_vias() -> List[Point]:
    return [(-4.2, 0.0), (-3.2, 0.0)]


def _leg_cost(
    pos: Point,
    color: str,
    cid: int,
    zone: str,
    zones: Dict[str, Point],
    *,
    use_door_via: bool = True,
) -> Tuple[float, Point, str]:
    cube = _cube_xy(color, cid)
    zxy = zones[zone.upper()]
    pts: List[Point] = [pos]
    note = ""
    if (
        use_door_via
        and color.lower() == "red"
        and int(cid) == 1
        and zone.upper() in ("A", "B")
    ):
        vias = list(_door_vias())
        dx, dy = cube[0] - pos[0], cube[1] - pos[1]
        L = math.hypot(dx, dy) or 1.0

        def _along(v: Point) -> float:
            return ((v[0] - pos[0]) * dx + (v[1] - pos[1]) * dy) / L

        pts.extend(sorted(vias, key=_along))
        note = "+door_via"
    pts.append(cube)
    pts.append(zxy)
    cost = sum(_dist(a, b) for a, b in zip(pts, pts[1:]))
    return cost, zxy, note


def expand_jobs(tasks: Sequence[PickTask], max_cubes: int = 5) -> List[JobSpec]:
    jobs: List[JobSpec] = []
    used_fixed = {"red": set(), "blue": set()}
    for t in tasks:
        color = t.color.lower()
        zone = t.zone.upper()
        if t.cube_id is not None:
            cid = int(t.cube_id)
            if cid in used_fixed[color]:
                continue
            used_fixed[color].add(cid)
            jobs.append(JobSpec(color=color, zone=zone, fixed_id=cid))
            continue
        for _ in range(int(t.count)):
            jobs.append(JobSpec(color=color, zone=zone, fixed_id=None))
    need = {"red": 0, "blue": 0}
    for j in jobs:
        need[j.color] += 1
    for c in ("red", "blue"):
        if need[c] > max_cubes:
            raise ValueError(f"need {need[c]} {c} cubes but only {max_cubes} exist")
    return jobs


def _assign_for_order(
    order: Sequence[JobSpec],
    start: Point,
    zones: Dict[str, Point],
    fixed: Dict[str, set],
    max_cubes: int,
    use_door_via: bool,
) -> Optional[Tuple[float, List[Leg], List[str]]]:
    used_walk = {"red": set(), "blue": set()}
    pos = start
    total = 0.0
    legs: List[Leg] = []
    detail: List[str] = []
    for j in order:
        if j.fixed_id is not None:
            cid = j.fixed_id
            if cid in used_walk[j.color]:
                return None
        else:
            best_cid = None
            best_d = float("inf")
            for cid in range(1, max_cubes + 1):
                if cid in used_walk[j.color] or cid in fixed[j.color]:
                    continue
                d, _, _ = _leg_cost(
                    pos, j.color, cid, j.zone, zones, use_door_via=use_door_via
                )
                if d < best_d:
                    best_d, best_cid = d, cid
            if best_cid is None:
                return None
            cid = best_cid
        d, end, note = _leg_cost(
            pos, j.color, cid, j.zone, zones, use_door_via=use_door_via
        )
        total += d
        used_walk[j.color].add(cid)
        legs.append((j.color, cid, j.zone))
        cx, cy = _cube_xy(j.color, cid)
        detail.append(f"Δ{d:.1f}m cube=({cx:.2f},{cy:.2f})→{j.zone}{note}")
        pos = end
    return total, legs, detail


def _nearest_job_order(
    jobs: List[JobSpec],
    start: Point,
    zones: Dict[str, Point],
    fixed: Dict[str, set],
    max_cubes: int,
    use_door_via: bool,
) -> List[JobSpec]:
    """Greedy construction when N is too large for full permutation."""
    remaining = list(jobs)
    built: List[JobSpec] = []
    pos = start
    used = {"red": set(), "blue": set()}
    while remaining:
        best_i = None
        best_d = float("inf")
        best_cid = None
        for i, j in enumerate(remaining):
            if j.fixed_id is not None:
                cands = [j.fixed_id]
            else:
                cands = [
                    c
                    for c in range(1, max_cubes + 1)
                    if c not in used[j.color] and c not in fixed[j.color]
                ]
            for cid in cands:
                if cid in used[j.color]:
                    continue
                d, _, _ = _leg_cost(
                    pos, j.color, cid, j.zone, zones, use_door_via=use_door_via
                )
                if d < best_d:
                    best_d, best_i, best_cid = d, i, cid
        if best_i is None or best_cid is None:
            # append rest as-is
            built.extend(remaining)
            break
        j = remaining.pop(best_i)
        built.append(j)
        used[j.color].add(best_cid)
        _, pos, _ = _leg_cost(
            pos, j.color, best_cid, j.zone, zones, use_door_via=use_door_via
        )
    return built


def plan_route(
    command: str,
    *,
    start: Point = (0.0, 0.0),
    max_cubes: int = 5,
    use_door_via: bool = True,
    poses_yaml: Optional[Path] = None,
) -> PlanResult:
    """Parse command → globally reorder legs for shortest Euclidean tour."""
    tasks = parse_command(command, max_cubes=max_cubes)
    jobs = expand_jobs(tasks, max_cubes=max_cubes)
    if not jobs:
        raise ValueError("no jobs to plan")
    zones = _load_zones(poses_yaml)
    fixed = {"red": set(), "blue": set()}
    for j in jobs:
        if j.fixed_id is not None:
            fixed[j.color].add(j.fixed_id)

    best: Optional[Tuple[float, List[Leg], List[str]]] = None
    n = len(jobs)

    if n <= 8:
        for perm in itertools.permutations(jobs):
            got = _assign_for_order(
                perm, start, zones, fixed, max_cubes, use_door_via
            )
            if got is None:
                continue
            if best is None or got[0] < best[0]:
                best = got
    else:
        order = _nearest_job_order(
            jobs, start, zones, fixed, max_cubes, use_door_via
        )
        best = _assign_for_order(
            order, start, zones, fixed, max_cubes, use_door_via
        )

    if best is None:
        raise RuntimeError("route planning failed (cube conflict?)")

    total, legs, detail = best
    return PlanResult(legs=legs, total_m=total, start=start, detail=detail)


def plan_to_queue(result: PlanResult) -> List[Leg]:
    return list(result.legs)


def main(argv: Optional[Iterable[str]] = None) -> None:
    import sys

    # Quick non-ROS: python3 -c "from yzbot_task.route_planner import plan_route; print(...)"
    # When launched via console_scripts / ros2 run, use ROS params only.
    if argv is not None:
        argv = list(argv)
    elif len(sys.argv) > 1 and sys.argv[1].startswith("["):
        cmd = sys.argv[1]
        start = (0.0, 0.0)
        if len(sys.argv) >= 4:
            start = (float(sys.argv[2]), float(sys.argv[3]))
        print(plan_route(cmd, start=start).summary())
        return

    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String

    class RoutePlannerNode(Node):
        def __init__(self):
            super().__init__("route_planner")
            self.declare_parameter("command", "[red,2,A;blue,1,C]")
            self.declare_parameter("start_x", 0.0)
            self.declare_parameter("start_y", 0.0)
            self.declare_parameter("once", True)
            self.declare_parameter("use_door_via", True)
            cmd = str(self.get_parameter("command").value)
            sx = float(self.get_parameter("start_x").value)
            sy = float(self.get_parameter("start_y").value)
            use_via = bool(self.get_parameter("use_door_via").value)
            pub = self.create_publisher(String, "/task_plan", 10)
            try:
                result = plan_route(cmd, start=(sx, sy), use_door_via=use_via)
            except Exception as exc:  # noqa: BLE001
                self.get_logger().error(f"plan failed: {exc}")
                raise SystemExit(1) from exc
            for line in result.summary().splitlines():
                self.get_logger().info(line)
            msg = String()
            msg.data = result.as_command()
            pub.publish(msg)
            self.get_logger().info(f"published /task_plan: {msg.data}")
            if bool(self.get_parameter("once").value):
                raise SystemExit(0)

    rclpy.init()
    try:
        node = RoutePlannerNode()
        rclpy.spin(node)
    except SystemExit:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
