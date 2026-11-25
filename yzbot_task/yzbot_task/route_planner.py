"""Global shortest-path task planner for random LLM task strings.

Optimizes BOTH:
  1) which cube id goes to which zone job
  2) visit order of pick→place legs

so total travel length is minimal:

  start → park(cube) → place slot → park(cube) → place slot → …

代价用**真实可行驶里程**（在 mapn3 静态图上按车体内切半径膨胀后做 A*），
不再用欧氏直线：门洞/隔墙把场地切成几块，同一对点欧氏 8m 实际绕 28m
（典型：C 放货区 ↔ 南口袋 red#5，欧氏 8.6m / 真实 13.8m 单程）。
16:19 实测该命令欧氏排法真实走 64.9m，真实里程最优 ≈51.4m。
取图失败时自动退回欧氏（+door vias），行为与旧版一致。
"""

from __future__ import annotations

import heapq
import itertools
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from yzbot_task.command_parser import PickTask, format_command, parse_command
from yzbot_task.vision_pose import approach_xy_for_cube, known_cube_xy

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
    cost_mode: str = "euclid"

    def as_command(self) -> str:
        tasks = [
            PickTask(color=c, count=1, zone=z, cube_id=i) for c, i, z in self.legs
        ]
        return format_command(tasks)

    def summary(self) -> str:
        lines = [
            f"ROUTE_PLAN total≈{self.total_m:.1f}m "
            f"start=({self.start[0]:.2f},{self.start[1]:.2f}) "
            f"cost={self.cost_mode}",
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


# ---------------------------------------------------------------------------
# 真实可行驶里程表（静态地图 + 车体内切半径膨胀 + 8 邻域 A*）
# ---------------------------------------------------------------------------

# 放块槽位（相对区心），与 task_sm._place_slot_xy 的表保持一致
_SLOT_OFFSETS: Tuple[Point, ...] = (
    (0.0, 0.0),
    (0.30, 0.17),
    (-0.30, -0.17),
    (0.30, -0.17),
    (-0.30, 0.17),
)


def _default_map_yaml() -> Optional[Path]:
    try:
        from ament_index_python.packages import get_package_share_directory

        p = Path(get_package_share_directory("bot_navigation")) / "maps" / "mapn3.yaml"
        if p.is_file():
            return p
    except Exception:  # noqa: BLE001
        pass
    root = Path(__file__).resolve().parents[2]  # …/yzbot/src
    for cand in (
        root / "bot_navigation" / "maps" / "mapn3.yaml",
        root / "bot_navigation" / "maps" / "map.yaml",
        Path(__file__).resolve().parents[1] / "config" / "mapn3.yaml",
    ):
        if cand.is_file():
            return cand
    return None


def _read_map_meta(yaml_path: Path) -> Tuple[Path, float, Point]:
    """最小化解析 map.yaml 的 image/resolution/origin（不依赖 PyYAML）。"""
    text = yaml_path.read_text(errors="ignore")
    image = "map.pgm"
    res = 0.05
    origin: Point = (0.0, 0.0)
    m = re.search(r"^\s*image\s*:\s*(\S+)", text, re.M)
    if m:
        image = m.group(1).strip()
    m = re.search(r"^\s*resolution\s*:\s*([-+0-9.eE]+)", text, re.M)
    if m:
        res = float(m.group(1))
    m = re.search(
        r"^\s*origin\s*:\s*\[\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)", text, re.M
    )
    if m:
        origin = (float(m.group(1)), float(m.group(2)))
    img = Path(image)
    if not img.is_absolute():
        img = yaml_path.parent / img
    return img, res, origin


def _load_pgm(path: Path) -> Optional["np.ndarray"]:
    try:
        import numpy as np
    except Exception:  # noqa: BLE001
        return None
    try:
        data = path.read_bytes()
        idx = 0
        fields: List[bytes] = []
        while len(fields) < 4:
            while idx < len(data) and data[idx:idx + 1].isspace():
                idx += 1
            if data[idx:idx + 1] == b"#":
                while idx < len(data) and data[idx:idx + 1] not in (b"\n", b""):
                    idx += 1
                continue
            start = idx
            while idx < len(data) and not data[idx:idx + 1].isspace():
                idx += 1
            fields.append(data[start:idx])
        w, h = int(fields[1]), int(fields[2])
        idx += 1
        raw = np.frombuffer(data[idx:idx + w * h], dtype=np.uint8)
        return raw.reshape(h, w)
    except Exception:  # noqa: BLE001
        return None


class _RoadGrid:
    """静态地图上的可行驶里程查询器（惰性 A* + 缓存，单次查询 1~3ms）。"""

    def __init__(
        self,
        yaml_path: Path,
        *,
        downsample: int = 2,
        inscribed: float = 0.40,
        blocked_extra: Sequence[Tuple[float, float, float]] = (),
    ) -> None:
        import numpy as np

        img_path, res, origin = _read_map_meta(yaml_path)
        img = _load_pgm(img_path)
        if img is None or res <= 0:
            raise RuntimeError(f"road map unreadable: {img_path}")
        ds = max(1, int(downsample))
        h, w = img.shape
        occ = img < 165
        o = occ[: h // ds * ds, : w // ds * ds].reshape(
            h // ds, ds, w // ds, ds
        ).any(axis=(1, 3))
        self.res = res * ds
        self.h, self.w = o.shape
        self.origin = origin
        blocked = o.copy()
        for cx, cy, r in blocked_extra:
            rr = int(math.ceil(max(float(r), self.res) / self.res))
            r0 = self.h - 1 - int((cy - origin[1]) / self.res)
            c0 = int((cx - origin[0]) / self.res)
            for dr in range(-rr, rr + 1):
                for dc in range(-rr, rr + 1):
                    if dr * dr + dc * dc > (max(float(r), self.res) / self.res) ** 2:
                        continue
                    row, col = r0 + dr, c0 + dc
                    if 0 <= row < self.h and 0 <= col < self.w:
                        blocked[row, col] = True
        # 按车体内切半径膨胀：障碍格扩成圆盘，圆盘外仍可通行
        r_cells = float(inscribed) / self.res
        rr = int(math.floor(r_cells))
        for dr in range(-rr, rr + 1):
            for dc in range(-rr, rr + 1):
                if dr * dr + dc * dc <= r_cells * r_cells + 1e-9:
                    blocked |= np.roll(np.roll(o, dr, axis=0), dc, axis=1)
        self.blocked = blocked
        self._cache: Dict[Tuple[int, int, int, int], Optional[float]] = {}
        self._free: Dict[Tuple[int, int], Tuple[int, int]] = {}

    def cell(self, p: Point) -> Tuple[int, int]:
        r = self.h - 1 - int((p[1] - self.origin[1]) / self.res)
        c = int((p[0] - self.origin[0]) / self.res)
        return min(max(r, 0), self.h - 1), min(max(c, 0), self.w - 1)

    def _nearest_free(self, r: int, c: int) -> Optional[Tuple[int, int]]:
        key = (r, c)
        if key in self._free:
            return self._free[key]
        out: Optional[Tuple[int, int]] = None
        for rad in range(0, 31):
            best = None
            for dr in range(-rad, rad + 1):
                for dc in range(-rad, rad + 1):
                    if max(abs(dr), abs(dc)) != rad:
                        continue
                    rr, cc = r + dr, c + dc
                    if not (0 <= rr < self.h and 0 <= cc < self.w):
                        continue
                    if not self.blocked[rr, cc]:
                        d2 = dr * dr + dc * dc
                        if best is None or d2 < best[0]:
                            best = (d2, rr, cc)
            if best is not None:
                out = (best[1], best[2])
                break
        self._free[key] = out  # type: ignore[assignment]
        return out

    def dist(self, a: Point, b: Point) -> Optional[float]:
        ca = self._nearest_free(*self.cell(a))
        cb = self._nearest_free(*self.cell(b))
        if ca is None or cb is None:
            return None
        key = (ca[0], ca[1], cb[0], cb[1])
        if key in self._cache:
            return self._cache[key]
        val = self._astar(ca, cb)
        self._cache[key] = val
        return val

    def _astar(
        self, src: Tuple[int, int], dst: Tuple[int, int]
    ) -> Optional[float]:
        (sr, sc), (gr, gc) = src, dst
        R = self.res
        SQ2 = math.sqrt(2.0)
        hh, ww = self.h, self.w
        br = self.blocked

        def hcost(r: int, c: int) -> float:
            dr = abs(r - gr)
            dc = abs(c - gc)
            return (dr + dc + (SQ2 - 2.0) * min(dr, dc)) * R

        INF = float("inf")
        g: Dict[Tuple[int, int], float] = {(sr, sc): 0.0}
        pq: List[Tuple[float, float, int, int]] = [(hcost(sr, sc), 0.0, sr, sc)]
        while pq:
            _f, gcst, r, c = heapq.heappop(pq)
            if (r, c) == (gr, gc):
                return gcst
            if gcst > g.get((r, c), INF):
                continue
            for dr, dc in (
                (1, 0), (-1, 0), (0, 1), (0, -1),
                (1, 1), (1, -1), (-1, 1), (-1, -1),
            ):
                nr, nc = r + dr, c + dc
                if not (0 <= nr < hh and 0 <= nc < ww) or br[nr, nc]:
                    continue
                step = R if (dr == 0 or dc == 0) else R * SQ2
                ng = gcst + step
                if ng < g.get((nr, nc), INF):
                    g[(nr, nc)] = ng
                    heapq.heappush(pq, (ng + hcost(nr, nc), ng, nr, nc))
        return None


_ROAD_CACHE: Dict[str, Optional[_RoadGrid]] = {}


def _get_road_grid(
    map_yaml: Optional[Path] = None,
    *,
    downsample: int = 2,
    inscribed: float = 0.40,
) -> Optional[_RoadGrid]:
    """按 map 路径缓存图对象；任何异常都返回 None（调用方退回欧氏）。"""
    path = Path(map_yaml) if map_yaml is not None else _default_map_yaml()
    if path is None or not Path(path).is_file():
        return None
    key = f"{Path(path).resolve()}|{downsample}|{inscribed:.3f}"
    if key not in _ROAD_CACHE:
        try:
            _ROAD_CACHE[key] = _RoadGrid(
                Path(path), downsample=downsample, inscribed=inscribed
            )
        except Exception:  # noqa: BLE001
            _ROAD_CACHE[key] = None
    return _ROAD_CACHE[key]


class _CostModel:
    """单腿代价：真实里程(pos→park) + 真实里程(park→放块槽位)。

    节点用**抓取停靠位 park**（视觉停位，不是块本身的坐标）和**放块槽位**，
    这样才和机器人真正要走的里程对齐；取图失败时退回欧氏直线。
    """

    def __init__(
        self,
        zones: Dict[str, Point],
        *,
        use_door_via: bool = False,
        road: Optional[_RoadGrid] = None,
        standoff_m: float = 0.55,
    ) -> None:
        self.zones = zones
        self.use_door_via = bool(use_door_via)
        self.road = road
        self.standoff_m = float(standoff_m)
        self._park: Dict[Tuple[str, int], Point] = {}
        self._cube: Dict[Tuple[str, int], Point] = {}
        # 节点集很小（start + ≤10 park + ≤15 槽位），按点对缓存真实里程，
        # 让 5! 全排列穷举仍是纯字典查找（≤1.5s）。
        self._dcache: Dict[Tuple[Point, Point], Optional[float]] = {}

    def cube_xy(self, color: str, cid: int) -> Point:
        return self._cube_xy(color, cid)

    def _cube_xy(self, color: str, cid: int) -> Point:
        key = (str(color).lower(), int(cid))
        got = self._cube.get(key)
        if got is None:
            got = _cube_xy(key[0], key[1])
            self._cube[key] = got
        return got

    def _road_dist(self, a: Point, b: Point) -> Optional[float]:
        key = (a, b)
        if key in self._dcache:
            return self._dcache[key]
        val = self.road.dist(a, b) if self.road is not None else None
        self._dcache[key] = val
        return val

    def park_xy(self, color: str, cid: int) -> Point:
        key = (str(color).lower(), int(cid))
        got = self._park.get(key)
        if got is None:
            try:
                ax, ay, _yaw = approach_xy_for_cube(
                    key[0], key[1], standoff_m=self.standoff_m
                )
                got = (float(ax), float(ay))
            except Exception:  # noqa: BLE001
                got = _cube_xy(*key)
            self._park[key] = got
        return got

    def slot_xy(self, zone: str, slot_index: int) -> Point:
        zx, zy = self.zones[str(zone).upper()]
        dx, dy = _SLOT_OFFSETS[int(slot_index) % len(_SLOT_OFFSETS)]
        return (zx + dx, zy + dy)

    def leg(
        self,
        pos: Point,
        color: str,
        cid: int,
        zone: str,
        slot_index: int = 0,
    ) -> Tuple[float, Point, str]:
        """Cost of one pick→place leg; end pose = 放块槽位。"""
        cube = self._cube_xy(color, cid)
        park = self.park_xy(color, cid)
        target = self.slot_xy(zone, slot_index)
        if self.road is not None:
            d1 = self._road_dist(pos, park)
            d2 = self._road_dist(park, target)
            if d1 is not None and d2 is not None:
                return (
                    d1 + d2,
                    target,
                    f"+road park=({park[0]:.2f},{park[1]:.2f})",
                )
            note = "+road_fallback"
        else:
            note = ""
        # ---- 欧氏兜底（保留旧行为）----
        pts: List[Point] = [pos]
        if (
            self.use_door_via
            and str(color).lower() == "red"
            and int(cid) == 1
            and str(zone).upper() in ("A", "B")
        ):
            vias = list(_door_vias())
            dx, dy = cube[0] - pos[0], cube[1] - pos[1]
            L = math.hypot(dx, dy) or 1.0
            pts.extend(
                sorted(
                    vias,
                    key=lambda v: ((v[0] - pos[0]) * dx + (v[1] - pos[1]) * dy) / L,
                )
            )
            note += "+door_via"
        pts.append(cube)
        pts.append(target)
        return sum(_dist(a, b) for a, b in zip(pts, pts[1:])), target, note


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
    model: Optional[_CostModel] = None,
    *,
    want_detail: bool = True,
) -> Tuple[float, List[str]]:
    model = model or _CostModel(zones, use_door_via=use_door_via)
    pos = start
    total = 0.0
    detail: List[str] = []
    placed: Dict[str, int] = {}
    for color, cid, zone in legs:
        zkey = str(zone).upper()
        d, end, note = model.leg(pos, color, cid, zkey, placed.get(zkey, 0))
        placed[zkey] = placed.get(zkey, 0) + 1
        total += d
        if want_detail:
            cx, cy = model.cube_xy(color, cid)
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
    model: Optional[_CostModel] = None,
) -> Tuple[float, List[Leg], List[str]]:
    """Shortest visit order for a fully assigned set of legs."""
    model = model or _CostModel(zones, use_door_via=use_door_via)
    n = len(legs)
    if n == 0:
        return 0.0, [], []
    if n == 1:
        total, detail = _tour_cost(legs, start, zones, use_door_via, model)
        return total, list(legs), detail

    best_total = float("inf")
    best_legs: List[Leg] = list(legs)

    if n <= 8:
        for perm in itertools.permutations(legs):
            total, _ = _tour_cost(
                perm, start, zones, use_door_via, model, want_detail=False
            )
            if total < best_total:
                best_total = total
                best_legs = list(perm)
        _, best_detail = _tour_cost(
            best_legs, start, zones, use_door_via, model
        )
        return best_total, best_legs, best_detail

    # Greedy NN fallback for large N
    remaining = list(legs)
    built: List[Leg] = []
    pos = start
    placed: Dict[str, int] = {}
    while remaining:
        best_i = 0
        best_d = float("inf")
        for i, (c, cid, z) in enumerate(remaining):
            d, _, _ = model.leg(pos, c, cid, z, placed.get(z.upper(), 0))
            if d < best_d:
                best_d, best_i = d, i
        leg = remaining.pop(best_i)
        built.append(leg)
        zkey = leg[2].upper()
        _, pos, _ = model.leg(pos, leg[0], leg[1], zkey, placed.get(zkey, 0))
        placed[zkey] = placed.get(zkey, 0) + 1
    total, detail = _tour_cost(built, start, zones, use_door_via, model)
    return total, built, detail


def plan_route(
    command: str,
    *,
    start: Point = (0.0, 0.0),
    max_cubes: int = 5,
    use_door_via: bool = True,
    poses_yaml: Optional[Path] = None,
    use_road_cost: bool = True,
    road_map: Optional[Path] = None,
    grasp_standoff_m: float = 0.55,
) -> PlanResult:
    """Parse command → assign cubes + reorder for shortest **drivable** tour."""
    tasks = parse_command(command, max_cubes=max_cubes)
    jobs = expand_jobs(tasks, max_cubes=max_cubes)
    if not jobs:
        raise ValueError("no jobs to plan")
    zones = _load_zones(poses_yaml)

    road = _get_road_grid(road_map) if use_road_cost else None
    model = _CostModel(
        zones,
        use_door_via=use_door_via,
        road=road,
        standoff_m=grasp_standoff_m,
    )
    cost_mode = "road" if road is not None else "euclid"

    assignments = _materialize_assignments(jobs, max_cubes)
    if not assignments:
        raise RuntimeError("route planning failed (cube conflict?)")

    # _materialize_assignments 会产出"同一组 (color,id,zone) 的不同排列"，
    # 而 _best_order 本身要穷举排列 —— 先去重，搜索量直接降一个量级
    # （[red,2,C;blue,3,C] 从 1200 组 × 120 排列 → 100 组 × 120 排列）。
    seen: set = set()
    uniq_assign: List[List[Leg]] = []
    for legs in assignments:
        key = tuple(sorted((str(c), int(cid), str(z).upper()) for c, cid, z in legs))
        if key in seen:
            continue
        seen.add(key)
        uniq_assign.append(list(legs))
    assignments = uniq_assign

    # Cap combinatorial blow-up: if too many assignments, sample via greedy ids
    # then only optimize order. Typical contest: ≤5 legs → fine.
    MAX_ASSIGN = 1200
    if len(assignments) > MAX_ASSIGN:
        # Prefer assignments whose legs are cheap under the same cost model
        def _assign_score(legs: List[Leg]) -> float:
            pos: Point = start
            s = 0.0
            placed: Dict[str, int] = {}
            for c, cid, z in legs:
                zkey = z.upper()
                d, pos, _ = model.leg(pos, c, cid, zkey, placed.get(zkey, 0))
                placed[zkey] = placed.get(zkey, 0) + 1
                s += d
            return s

        assignments = sorted(assignments, key=_assign_score)[:MAX_ASSIGN]

    best_total = float("inf")
    best_legs: List[Leg] = []
    best_detail: List[str] = []
    for legs in assignments:
        total, ordered, detail = _best_order(
            legs, start, zones, use_door_via, model
        )
        if total < best_total:
            best_total = total
            best_legs = ordered
            best_detail = detail

    if not best_legs:
        raise RuntimeError("route planning failed (empty)")

    return PlanResult(
        legs=best_legs,
        total_m=best_total,
        start=start,
        detail=best_detail,
        cost_mode=cost_mode,
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
            self.declare_parameter("use_road_cost", True)
            cmd = str(self.get_parameter("command").value)
            sx = float(self.get_parameter("start_x").value)
            sy = float(self.get_parameter("start_y").value)
            use_via = bool(self.get_parameter("use_door_via").value)
            pub = self.create_publisher(String, "/task_plan", 10)
            try:
                result = plan_route(
                    cmd,
                    start=(sx, sy),
                    use_door_via=use_via,
                    use_road_cost=bool(
                        self.get_parameter("use_road_cost").value
                    ),
                )
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
