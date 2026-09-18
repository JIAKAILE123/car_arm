"""Read live Gazebo cube poses → log + YAML (optional apply into code/world).

Gazebo Classic often does NOT publish /gazebo/model_states even with
libgazebo_ros_state.so loaded. This tool prefers `gz model -m NAME -p`.

Usage (Gazebo already running):

  ros2 run yzbot_task export_cube_poses --ros-args -p once:=true

  # dump + write vision_pose + offic_room.world
  ros2 run yzbot_task export_cube_poses --ros-args -p once:=true -p apply:=true

Gazebo: Translation Mode → drag cubes → run once:=true.
"""

from __future__ import annotations

import os
import re
import subprocess
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import rclpy
from rclpy.node import Node
from gazebo_msgs.msg import ModelStates

CUBE_NAMES = [f"{c}_cube_{i}" for c in ("red", "blue") for i in range(1, 6)]
ZONE_NAMES = ["zone_a", "zone_b", "zone_c"]
ALL_NAMES = ZONE_NAMES + CUBE_NAMES


def _ws_paths() -> Tuple[Path, Path, Path, Path]:
    """(export_yaml, vision_pose.py, offic_room.world, poses.yaml)"""
    here = Path(__file__).resolve()
    pkg = here.parents[1]
    src = here.parents[2]
    yzbot = here.parents[3]
    export = pkg / "config" / "cube_poses_export.yaml"
    vision = pkg / "yzbot_task" / "vision_pose.py"
    poses_yaml = pkg / "config" / "poses.yaml"
    world = src / "mybot_description" / "worlds" / "offic_room.world"
    if not world.is_file():
        world = yzbot / "src" / "mybot_description" / "worlds" / "offic_room.world"
    return export, vision, world, poses_yaml


def _fmt_pose(x: float, y: float, z: float) -> str:
    return f"({x:.2f}, {y:.2f}, {z:.3f})"


def _gz_model_pose(name: str, timeout: float = 5.0) -> Optional[Tuple[float, float, float]]:
    """Return (x,y,z) from `gz model -m NAME -p`, or None."""
    env = os.environ.copy()
    env.setdefault("GAZEBO_MASTER_URI", "http://127.0.0.1:11345")
    try:
        proc = subprocess.run(
            ["gz", "model", "-m", name, "-p"],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env=env,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    raw = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    if not raw:
        if err:
            # surface once via caller logs
            _gz_model_pose.last_err = err  # type: ignore[attr-defined]
        return None
    nums = []
    for tok in raw.replace(",", " ").split():
        try:
            nums.append(float(tok))
        except ValueError:
            continue
        if len(nums) >= 3:
            break
    if len(nums) < 3:
        _gz_model_pose.last_err = f"bad parse: {raw!r}"  # type: ignore[attr-defined]
        return None
    return float(nums[0]), float(nums[1]), float(nums[2])


_gz_model_pose.last_err = ""  # type: ignore[attr-defined]


def _read_all_via_gz(
    force_z: bool, z_cube: float, z_zone: float, log
) -> Dict[str, Tuple[float, float, float]]:
    out: Dict[str, Tuple[float, float, float]] = {}
    for name in ALL_NAMES:
        xyz = _gz_model_pose(name)
        if xyz is None:
            log.warn(
                f"gz model failed: {name} | {_gz_model_pose.last_err}"  # type: ignore[attr-defined]
            )
            continue
        x, y, z = xyz
        if force_z:
            z = z_zone if name.startswith("zone_") else z_cube
        out[name] = (x, y, z)
        log.info(f"gz OK {name} → {_fmt_pose(x, y, z)}")
    return out


class ExportCubePoses(Node):
    def __init__(self):
        super().__init__("export_cube_poses")
        self.declare_parameter("once", False)
        self.declare_parameter("apply", False)
        self.declare_parameter("period_sec", 2.0)
        self.declare_parameter("z_floor", 0.015)
        self.declare_parameter("z_zone", 0.01)
        self.declare_parameter("force_floor_z", True)
        self.declare_parameter("prefer_gz", True)
        self.declare_parameter("wait_model_states_sec", 3.0)

        self._once = bool(self.get_parameter("once").value)
        self._apply = bool(self.get_parameter("apply").value)
        self._period = float(self.get_parameter("period_sec").value)
        self._z_floor = float(self.get_parameter("z_floor").value)
        self._z_zone = float(self.get_parameter("z_zone").value)
        self._force_z = bool(self.get_parameter("force_floor_z").value)
        self._prefer_gz = bool(self.get_parameter("prefer_gz").value)
        self._wait_ms = float(self.get_parameter("wait_model_states_sec").value)

        self._latest: Optional[ModelStates] = None
        self._done = False
        self._t0 = time.monotonic()
        self.create_subscription(ModelStates, "/gazebo/model_states", self._on_ms, 10)

        self.get_logger().info(
            "export_cube_poses READY | cubes + zone_a/b/c | prefer_gz="
            f"{self._prefer_gz} once={self._once} apply={self._apply}"
        )
        self.get_logger().info(
            "Zones stay <static> (letters won't fall) but ARE draggable "
            "with Gazebo Translation Mode — same as moving a wall plaque."
        )

        # Do NOT wait on sim-time timers — they stall if /clock is odd.
        if self._once:
            self._snapshot("ONCE")
            self._done = True
            # exit from main after spin briefly
            self.create_timer(0.05, self._shutdown)
        else:
            self.create_timer(0.5, self._tick)

    def _on_ms(self, msg: ModelStates):
        self._latest = msg

    def _extract_from_ms(self) -> Optional[Dict[str, Tuple[float, float, float]]]:
        msg = self._latest
        if msg is None:
            return None
        out: Dict[str, Tuple[float, float, float]] = {}
        for name in ALL_NAMES:
            if name not in msg.name:
                continue
            i = msg.name.index(name)
            p = msg.pose[i].position
            if self._force_z:
                z = self._z_zone if name.startswith("zone_") else self._z_floor
            else:
                z = float(p.z)
            out[name] = (float(p.x), float(p.y), float(z))
        return out

    def _collect(self) -> Dict[str, Tuple[float, float, float]]:
        if self._prefer_gz:
            poses = _read_all_via_gz(
                self._force_z, self._z_floor, self._z_zone, self.get_logger()
            )
            if len(poses) >= 1:
                return poses
            self.get_logger().warn("gz returned nothing — trying /gazebo/model_states")
        ms = self._extract_from_ms()
        return ms or {}

    def _log_table(self, poses: Dict[str, Tuple[float, float, float]], tag: str):
        # Zones first — user cares about A/B/C placement
        self.get_logger().info(f"===== ZONE_POSES {tag} =====")
        for name in ZONE_NAMES:
            if name not in poses:
                self.get_logger().warn(f"  MISSING {name}")
                continue
            x, y, z = poses[name]
            letter = name[-1].upper()
            self.get_logger().info(
                f"  zone_{letter}  {_fmt_pose(x, y, z)}   (model={name})"
            )
        self.get_logger().info(
            f"===== CUBE_POSES {tag} "
            f"({sum(1 for n in CUBE_NAMES if n in poses)}/10) ====="
        )
        for name in CUBE_NAMES:
            if name not in poses:
                self.get_logger().warn(f"  MISSING {name}")
                continue
            x, y, z = poses[name]
            self.get_logger().info(f"  {name:12s}  {_fmt_pose(x, y, z)}")
        lines = ["_KNOWN_CUBES = {"]
        for color in ("red", "blue"):
            for i in range(1, 6):
                key = f"{color}_cube_{i}"
                if key not in poses:
                    continue
                x, y, z = poses[key]
                lines.append(f'    ("{color}", {i}): ({x:.2f}, {y:.2f}, {z:.3f}),')
        lines.append("}")
        self.get_logger().info(
            "----- paste into vision_pose.py -----\n" + "\n".join(lines)
        )
        zlines = ["# poses.yaml zones"]
        for name in ZONE_NAMES:
            if name not in poses:
                continue
            x, y, _z = poses[name]
            zlines.append(f"{name}:\n  x: {x:.2f}\n  y: {y:.2f}\n  yaw: 0.0")
        self.get_logger().info("----- paste into poses.yaml -----\n" + "\n".join(zlines))

    def _write_yaml(self, poses: Dict[str, Tuple[float, float, float]], path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Auto-exported from Gazebo (gz model -p)",
            f"# time: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "zones:",
        ]
        for name in ZONE_NAMES:
            if name not in poses:
                continue
            x, y, z = poses[name]
            lines.append(f"  {name}: {{x: {x:.4f}, y: {y:.4f}, z: {z:.4f}}}")
        lines.append("cubes:")
        for name in CUBE_NAMES:
            if name not in poses:
                continue
            x, y, z = poses[name]
            lines.append(f"  {name}: {{x: {x:.4f}, y: {y:.4f}, z: {z:.4f}}}")
        path.write_text("\n".join(lines) + "\n")
        self.get_logger().info(f"WROTE {path}")

    def _apply_files(self, poses: Dict[str, Tuple[float, float, float]]):
        export_path, vision_path, world_path, poses_yaml = _ws_paths()
        n_cubes = sum(1 for n in CUBE_NAMES if n in poses)
        n_zones = sum(1 for n in ZONE_NAMES if n in poses)
        if n_cubes < 10 and n_zones < 3:
            self.get_logger().error(
                f"apply needs 10 cubes and/or 3 zones, got cubes={n_cubes} zones={n_zones}"
            )
            return
        if not world_path.is_file():
            self.get_logger().error(f"world missing: {world_path}")
            return

        old_by_name: Dict[str, Tuple[float, float, float]] = {}
        try:
            from yzbot_task.vision_pose import _KNOWN_CUBES

            for (color, cid), xyz in _KNOWN_CUBES.items():
                old_by_name[f"{color}_cube_{int(cid)}"] = (
                    float(xyz[0]),
                    float(xyz[1]),
                    float(xyz[2]),
                )
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"old cubes skip: {exc}")
        # old zones from poses.yaml
        if poses_yaml.is_file():
            try:
                import yaml

                data = yaml.safe_load(poses_yaml.read_text()) or {}
                for name in ZONE_NAMES:
                    if name in data and "x" in data[name]:
                        old_by_name[name] = (
                            float(data[name]["x"]),
                            float(data[name]["y"]),
                            float(data[name].get("z", 0.01)),
                        )
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warn(f"old zones skip: {exc}")
        # also common world tokens
        for name, (ox, oy, oz) in {
            "zone_a": (-1.00, -8.10, 0.01),
            "zone_b": (1.50, -4.00, 0.01),
            "zone_c": (-0.80, 0.20, 0.01),
        }.items():
            old_by_name.setdefault(name, (ox, oy, oz))

        if n_cubes == 10 and vision_path.is_file():
            block_lines = [
                "# Layout from Gazebo export (export_cube_poses apply:=true).",
                "# Pairwise spacing should stay ≥2m for competition.",
                "_KNOWN_CUBES = {",
            ]
            for color in ("red", "blue"):
                for i in range(1, 6):
                    x, y, z = poses[f"{color}_cube_{i}"]
                    block_lines.append(
                        f'    ("{color}", {i}): ({x:.2f}, {y:.2f}, {z:.3f}),'
                    )
            block_lines.append("}")
            new_block = "\n".join(block_lines)
            vtext = vision_path.read_text()
            pat = re.compile(r"(?:# [^\n]*\n)*_KNOWN_CUBES = \{[^}]+\}", re.M)
            m = pat.search(vtext)
            if m:
                vision_path.write_text(vtext[: m.start()] + new_block + vtext[m.end() :])
                self.get_logger().info(f"UPDATED {vision_path}")
            else:
                self.get_logger().error("could not find _KNOWN_CUBES")

        if n_zones == 3 and poses_yaml.is_file():
            ytext = poses_yaml.read_text()
            for name in ZONE_NAMES:
                x, y, _z = poses[name]
                pat = re.compile(
                    rf"({re.escape(name)}:\s*\n\s*x:\s*)([-0-9.eE+]+)"
                    rf"(\s*\n\s*y:\s*)([-0-9.eE+]+)",
                    re.M,
                )
                ytext2, nsub = pat.subn(rf"\g<1>{x:.2f}\g<3>{y:.2f}", ytext)
                if nsub:
                    ytext = ytext2
                    self.get_logger().info(f"poses.yaml {name} ×{nsub}")
                else:
                    self.get_logger().warn(f"poses.yaml miss {name}")
            poses_yaml.write_text(ytext)
            self.get_logger().info(f"UPDATED {poses_yaml}")

        wtext = world_path.read_text()
        for name, (x, y, z) in poses.items():
            if name not in ALL_NAMES:
                continue
            new_pose = f"{x:.2f} {y:.2f} {z:.3f} 0 -0 0"
            # zones historically use 0.01 without trailing zeros
            if name.startswith("zone_"):
                new_pose = f"{x:.1f} {y:.1f} {z:.2f} 0 -0 0"
                # prefer 1 decimal like 5.4 -11.8 if close
                new_pose = f"{x:.2f} {y:.2f} {z:.2f} 0 -0 0"
            pattern = re.compile(
                rf"(<model name='{re.escape(name)}'>\s*\n?\s*<static>1</static>\s*\n?\s*<pose>)([^<]+)(</pose>)",
                re.M,
            )
            wtext, nsub = pattern.subn(rf"\g<1>{new_pose}\g<3>", wtext)
            if nsub == 0:
                pattern = re.compile(
                    rf"(<model name='{re.escape(name)}'>\s*\n?\s*<pose>)([^<]+)(</pose>)",
                    re.M,
                )
                wtext, nsub = pattern.subn(rf"\g<1>{new_pose}\g<3>", wtext)
            self.get_logger().info(f"world first-pose {name} ×{nsub}")
            old = old_by_name.get(name)
            if old is None:
                continue
            ox, oy, oz = old
            variants = [
                f"{ox:.2f} {oy:.2f} {oz:.3f}",
                f"{ox:.2f} {oy:.2f} {oz:.2f}",
                f"{ox:.1f} {oy:.1f} {oz:.2f}",
                f"{ox} {oy} {oz}",
            ]
            new_tok = f"{x:.2f} {y:.2f} {z:.2f}" if name.startswith("zone_") else f"{x:.2f} {y:.2f} {z:.3f}"
            for old_tok in variants:
                if old_tok in wtext and old_tok != new_tok:
                    c = wtext.count(old_tok)
                    # only replace a few times to avoid nuking letter link offsets wrongly
                    # letter links use relative poses like -0.1 0 0.22 — different
                    wtext = wtext.replace(old_tok, new_tok)
                    self.get_logger().info(f"world token {name} '{old_tok}' ×{c}")

        world_path.write_text(wtext)
        self.get_logger().info(f"UPDATED {world_path}")
        self.get_logger().info(
            "apply done → colcon build --packages-select yzbot_task mybot_description "
            "&& restart Gazebo"
        )

    def _snapshot(self, tag: str) -> bool:
        poses = self._collect()
        if not poses:
            self.get_logger().error(
                "no poses — is Gazebo running? try: gz model -m zone_a -p"
            )
            return False
        self._log_table(poses, tag)
        export_path, _, _, _ = _ws_paths()
        self._write_yaml(poses, export_path)
        alt = Path.home() / "dev_ws" / "yzbot" / "cube_poses_export.yaml"
        try:
            self._write_yaml(poses, alt)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"alt write skip: {exc}")
        n_cubes = sum(1 for n in CUBE_NAMES if n in poses)
        n_zones = sum(1 for n in ZONE_NAMES if n in poses)
        if self._apply:
            if n_cubes == 10 or n_zones == 3:
                self._apply_files(poses)
            else:
                self.get_logger().error(
                    f"apply skipped (need 10 cubes and/or 3 zones; got {n_cubes}/{n_zones})"
                )
        return n_cubes == 10 and n_zones == 3

    def _tick(self):
        if self._done:
            return
        if self._once:
            # Don't wait forever on empty model_states — gz works immediately
            self._snapshot("ONCE")
            self._done = True
            self.create_timer(0.2, self._shutdown)
            return
        now = time.monotonic()
        if not hasattr(self, "_last_print"):
            self._last_print = 0.0
        if now - self._last_print >= max(self._period, 0.5):
            self._last_print = now
            self._snapshot(f"LIVE t={now:.0f}")

    def _shutdown(self):
        self.get_logger().info("export_cube_poses exit")
        raise SystemExit(0)


def main():
    rclpy.init()
    node = ExportCubePoses()
    try:
        rclpy.spin(node)
    except (SystemExit, KeyboardInterrupt):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
