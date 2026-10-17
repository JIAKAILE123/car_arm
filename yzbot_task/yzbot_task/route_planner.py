"""Global shortest-path task planner for random LLM task strings.

Optimizes BOTH:
  1) which cube id goes to which zone job
  2) visit order of pick→place legs

so total Euclidean length is minimal:

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
    zones = {"A": (4.20, -1.80), "B": (0.60, -3.40), "C": (-6.00, -2.90)}
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
    """Cost of one pick→place: pos → (vias) → cube → zone. End pose = zone."""
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


def _tour_cost(
    legs: Sequence[Leg],
    start: Point,
    zones: Dict[str, Point],
    use_door_via: bool,
) -> Tuple[float, List[str]]:
    pos = start
    total = 0.0
    detail: List[str] = []
    for color, cid, zone in legs:
        d, end, note = _leg_cost(
            pos, color, cid, zone, zones, use_door_via=use_door_via
        )
        total += d
        cx, cy = _cube_xy(color, cid)
        detail.append(f"Δ{d:.1f}m cube=({cx:.2f},{cy:.2f})→{zone}{note}")
        pos = end
    return total, detail


def _materialize_assignments(
    jobs: List[JobSpec], max_cubes: int
) -> List[List[Leg]]:
    """All ways to bind free jobs to unused cube ids (fixed ids locked)."""
    fixed = {"red": set(), "blue": set()}
    for j in jobs:
        if j.fixed_id is not None:
            fixed[j.color].add(int(j.fixed_id))

    free_idx = {"red": [], "blue": []}
    for i, j in enumerate(jobs):
        if j.fixed_id is None:
            free_idx[j.color].append(i)

    avail = {
        c: [i for i in range(1, max_cubes + 1) if i not in fixed[c]]
        for c in ("red", "blue")
    }

    # Choose which free ids to use, then assign them to free job slots
    red_need = len(free_idx["red"])
    blue_need = len(free_idx["blue"])
    if red_need > len(avail["red"]) or blue_need > len(avail["blue"]):
        return []

    out: List[List[Leg]] = []
    for red_pick in itertools.combinations(avail["red"], red_need):
        for blue_pick in itertools.combinations(avail["blue"], blue_need):
            for red_perm in itertools.permutations(red_pick):
                for blue_perm in itertools.permutations(blue_pick):
                    legs: List[Optional[Leg]] = [None] * len(jobs)
                    ri = bi = 0
                    ok = True
                    for i, j in enumerate(jobs):
                        if j.fixed_id is not None:
                            legs[i] = (j.color, int(j.fixed_id), j.zone)
                        elif j.color == "red":
                            legs[i] = (j.color, int(red_perm[ri]), j.zone)
                            ri += 1
                        else:
                            legs[i] = (j.color, int(blue_perm[bi]), j.zone)
                            bi += 1
                    if not ok or any(x is None for x in legs):
                        continue
                    out.append([x for x in legs if x is not None])  # type: ignore
    return out


def _best_order(
    legs: List[Leg],
    start: Point,
    zones: Dict[str, Point],
    use_door_via: bool,
) -> Tuple[float, List[Leg], List[str]]:
    """Shortest visit order for a fully assigned set of legs."""
    n = len(legs)
    if n == 0:
        return 0.0, [], []
    if n == 1:
        total, detail = _tour_cost(legs, start, zones, use_door_via)
        return total, list(legs), detail

    best_total = float("inf")
    best_legs: List[Leg] = list(legs)
    best_detail: List[str] = []

    if n <= 8:
        for perm in itertools.permutations(legs):
            total, detail = _tour_cost(perm, start, zones, use_door_via)
            if total < best_total:
                best_total = total
                best_legs = list(perm)
                best_detail = detail
        return best_total, best_legs, best_detail

    # Greedy NN fallback for large N
    remaining = list(legs)
    built: List[Leg] = []
    pos = start
    while remaining:
        best_i = 0
        best_d = float("inf")
        for i, (c, cid, z) in enumerate(remaining):
            d, _, _ = _leg_cost(pos, c, cid, z, zones, use_door_via=use_door_via)
            if d < best_d:
                best_d, best_i = d, i
        leg = remaining.pop(best_i)
        built.append(leg)
        _, pos, _ = _leg_cost(
            pos, leg[0], leg[1], leg[2], zones, use_door_via=use_door_via
        )
    total, detail = _tour_cost(built, start, zones, use_door_via)
    return total, built, detail


def plan_route(
    command: str,
    *,
    start: Point = (0.0, 0.0),
    max_cubes: int = 5,
    use_door_via: bool = True,
    poses_yaml: Optional[Path] = None,
) -> PlanResult:
    """Parse command → assign cubes + reorder for shortest Euclidean tour."""
    tasks = parse_command(command, max_cubes=max_cubes)
    jobs = expand_jobs(tasks, max_cubes=max_cubes)
    if not jobs:
        raise ValueError("no jobs to plan")
    zones = _load_zones(poses_yaml)

    assignments = _materialize_assignments(jobs, max_cubes)
    if not assignments:
        raise RuntimeError("route planning failed (cube conflict?)")

    # Cap combinatorial blow-up: if too many assignments, sample via greedy ids
    # then only optimize order. Typical contest: ≤5 legs → fine.
    MAX_ASSIGN = 1200
    if len(assignments) > MAX_ASSIGN:
        # Prefer assignments that use cubes nearest to their zone (cheap filter)
        def _assign_score(legs: List[Leg]) -> float:
            s = 0.0
            for c, cid, z in legs:
                s += _dist(_cube_xy(c, cid), zones[z.upper()])
            return s

        assignments = sorted(assignments, key=_assign_score)[:MAX_ASSIGN]

    best_total = float("inf")
    best_legs: List[Leg] = []
    best_detail: List[str] = []
    for legs in assignments:
        total, ordered, detail = _best_order(
            legs, start, zones, use_door_via
        )
        if total < best_total:
            best_total = total
            best_legs = ordered
            best_detail = detail

    if not best_legs:
        raise RuntimeError("route planning failed (empty)")

    return PlanResult(
        legs=best_legs, total_m=best_total, start=start, detail=best_detail
    )


def plan_to_queue(result: PlanResult) -> List[Leg]:
    return list(result.legs)


def main(argv: Optional[Iterable[str]] = None) -> None:
    import sys

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
