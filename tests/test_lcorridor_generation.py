#!/usr/bin/env python3

import hashlib
import re
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

import numpy as np
import open3d as o3d


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
LAUNCH_DIR = Path(__file__).resolve().parents[2] / "accel_control" / "launch"
sys.path.insert(0, str(SCRIPTS_DIR))


def _run_generate(*extra_args):
    return subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / "generate_lcorridor.py"), *extra_args],
        check=True,
        stdout=subprocess.DEVNULL,
    )


def _run_generate_capture(*extra_args):
    return subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / "generate_lcorridor.py"), *extra_args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _load_boxes(world_path):
    world = ET.parse(world_path).getroot().find("world")
    boxes = []
    for model in world.findall("model"):
        pose = np.fromstring(model.findtext("pose"), sep=" ")
        size = np.fromstring(
            model.findtext("./link/collision/geometry/box/size"), sep=" "
        )
        boxes.append((model.get("name"), pose, size))
    return boxes


def _inside_any_box(points, boxes):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    inside = np.zeros(len(points), dtype=bool)
    for _, pose, size in boxes:
        half = size / 2.0
        inside |= np.all(np.abs(points - pose[:3]) < half - 1e-6, axis=1)
    return inside


def _read_pcd_points(path):
    cloud = o3d.io.read_point_cloud(str(path))
    return np.asarray(cloud.points)


class LCorridorGenerationTest(unittest.TestCase):
    def _generate_to(self, root, *extra_args):
        output_root = Path(root) / "out"
        _run_generate("--output-root", str(output_root), *extra_args)
        return output_root

    def test_world_boxes_match_hand_computed_table(self):
        # Hand-computed from the design spec for leg1=8, leg2=6, W=2, H=2.5,
        # t=0.2, left turn: 4 walls then 2 ceiling slabs as (cx, cy, cz, l, w, h).
        expected = (
            (4.4, -1.1, 1.25, 9.2, 0.2, 2.5),
            (3.4, 1.1, 1.25, 7.2, 0.2, 2.5),
            (9.1, 2.4, 1.25, 0.2, 7.2, 2.5),
            (6.9, 3.5, 1.25, 0.2, 5.0, 2.5),
            (4.5, 0.0, 2.6, 9.4, 2.4, 0.2),
            (8.0, 3.5, 2.6, 2.4, 5.4, 0.2),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = self._generate_to(
                temp_dir,
                "--leg1-length", "8.0",
                "--leg2-length", "6.0",
                "-W", "2.0",
                "-H", "2.5",
                "--wall-thickness", "0.2",
                "--turn-direction", "left",
            )
            boxes = _load_boxes(
                root / "worlds" / "lcorridor" / "cylinder_lcorridor.world"
            )
            self.assertEqual(len(boxes), len(expected))
            for (_, pose, size), (cx, cy, cz, l, w, h) in zip(boxes, expected):
                np.testing.assert_allclose(pose[:3], [cx, cy, cz], atol=5e-4)
                np.testing.assert_allclose(size, [l, w, h], atol=5e-4)

    def test_free_space_and_wall_bands(self):
        leg1, leg2, width, height, t = 8.0, 6.0, 2.0, 2.5, 0.2
        half = width / 2.0
        inner_x, outer_x = leg1 - half, leg1 + half
        with tempfile.TemporaryDirectory() as temp_dir:
            root = self._generate_to(
                temp_dir,
                "--leg1-length", str(leg1),
                "--leg2-length", str(leg2),
                "-W", str(width),
                "-H", str(height),
                "--wall-thickness", str(t),
            )
            boxes = _load_boxes(
                root / "worlds" / "lcorridor" / "cylinder_lcorridor.world"
            )

        zs = np.linspace(0.2, height - 0.2, 5)
        free = []
        for x in np.linspace(0.1, outer_x - 0.1, 20):
            for y in (-half + 0.05, 0.0, half - 0.05):
                for z in zs:
                    free.append((x, y, z))
        for x in (inner_x + 0.05, leg1, outer_x - 0.05):
            for y in np.linspace(half, leg2 - 0.1, 20):
                for z in zs:
                    free.append((x, y, z))
        for y in np.linspace(-half + 0.05, leg2 - 0.1, 25):
            for x in (inner_x + 0.05, outer_x - 0.05):
                for z in zs:
                    free.append((x, y, z))
        self.assertFalse(
            _inside_any_box(free, boxes).any(),
            "free corridor samples must not fall inside any wall/ceiling box",
        )

        # Wall bands must be solid: no gaps in the enclosure.
        bands = []
        for x in np.linspace(0.05, outer_x - 0.05, 25):
            for y in np.linspace(-half - t + 0.01, -half - 0.01, 4):
                bands.append((x, y, height / 2.0))
        for x in np.linspace(0.05, inner_x - 0.05, 20):
            for y in np.linspace(half + 0.01, half + t - 0.01, 4):
                bands.append((x, y, height / 2.0))
        for y in np.linspace(-half - t + 0.01, leg2 - 0.05, 25):
            for x in np.linspace(outer_x + 0.01, outer_x + t - 0.01, 4):
                bands.append((x, y, height / 2.0))
        for y in np.linspace(half + 0.01, leg2 - 0.05, 20):
            for x in np.linspace(inner_x - t + 0.01, inner_x - 0.01, 4):
                bands.append((x, y, height / 2.0))
        self.assertTrue(
            _inside_any_box(bands, boxes).all(),
            "wall band samples must all lie inside some wall box",
        )

        # Ceiling slabs must cover the whole footprint without seams.
        zc = height + t / 2.0
        ceiling = []
        for x in np.linspace(-t + 0.01, outer_x + t - 0.01, 25):
            for y in np.linspace(-half - t + 0.01, half + t - 0.01, 10):
                ceiling.append((x, y, zc))
        for x in np.linspace(inner_x - t + 0.01, outer_x + t - 0.01, 10):
            for y in np.linspace(half, leg2 + t - 0.01, 25):
                ceiling.append((x, y, zc))
        self.assertTrue(
            _inside_any_box(ceiling, boxes).all(),
            "ceiling samples must all lie inside some ceiling box",
        )

        # Cross-sections of the free tube must be exactly W wide. Leg 1 walls
        # are offset in y, leg 2 walls are offset in x.
        probe_free = [
            (1.0, half - 0.02, 1.0), (1.0, -half + 0.02, 1.0),
            (inner_x - 1.0, half - 0.02, 1.0), (inner_x - 1.0, -half + 0.02, 1.0),
            (leg1 - (half - 0.02), leg2 - 1.0, 1.0),
            (leg1 + (half - 0.02), leg2 - 1.0, 1.0),
        ]
        probe_wall = [
            (1.0, half + 0.02, 1.0), (1.0, -half - 0.02, 1.0),
            (inner_x - 1.0, half + 0.02, 1.0), (inner_x - 1.0, -half - 0.02, 1.0),
            (leg1 - (half + 0.02), leg2 - 1.0, 1.0),
            (leg1 + (half + 0.02), leg2 - 1.0, 1.0),
        ]
        self.assertFalse(_inside_any_box(probe_free, boxes).any())
        self.assertTrue(_inside_any_box(probe_wall, boxes).all())

    def test_right_turn_mirrors_y(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            left = self._generate_to(
                Path(temp_dir) / "left", "--turn-direction", "left"
            )
            right = self._generate_to(
                Path(temp_dir) / "right", "--turn-direction", "right"
            )
            left_boxes = _load_boxes(
                left / "worlds" / "lcorridor" / "cylinder_lcorridor.world"
            )
            right_boxes = _load_boxes(
                right / "worlds" / "lcorridor" / "cylinder_lcorridor.world"
            )
            self.assertEqual(len(left_boxes), len(right_boxes))
            for (_, left_pose, left_size), (_, right_pose, right_size) in zip(
                left_boxes, right_boxes
            ):
                np.testing.assert_allclose(
                    right_pose[:3], [left_pose[0], -left_pose[1], left_pose[2]],
                    atol=5e-4,
                )
                np.testing.assert_allclose(right_size, left_size, atol=5e-4)

    def test_reported_home_and_target_follow_geometry(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            left = _run_generate_capture(
                "--output-root", str(Path(temp_dir) / "left"),
                "--leg1-length", "8", "--leg2-length", "6",
            )
            right = _run_generate_capture(
                "--output-root", str(Path(temp_dir) / "right"),
                "--leg1-length", "8", "--leg2-length", "6",
                "--turn-direction", "right",
            )
        self.assertIn("home: (0.000, 0.000)", left)
        self.assertIn("target: (8.000, 6.000)", left)
        self.assertIn("home: (0.000, 0.000)", right)
        self.assertIn("target: (8.000, -6.000)", right)

    def test_launch_defaults_agree_with_generator_defaults(self):
        # yopo_oa_lcorridor.launch and the official-MINCO wrapper both ship
        # target defaults that must match the generator's default dimensions
        # (leg1 = leg2 = 15).
        launch = ET.parse(LAUNCH_DIR / "yopo_oa_lcorridor.launch").getroot()
        defaults = {
            arg.get("name"): arg.get("default")
            for arg in launch.findall("arg")
            if arg.get("name") in ("target_x", "target_y")
        }
        wrapper = (
            Path(__file__).resolve().parents[2]
            / "baseline" / "scripts" / "evaluate_yopo_minco.py"
        ).read_text()
        wrapper_block = wrapper.split('elif args.scene == "lcorridor":', 1)[1]
        with tempfile.TemporaryDirectory() as temp_dir:
            stdout = _run_generate_capture("--output-root", str(Path(temp_dir) / "out"))
        match = re.search(r"target: \((-?\d+\.\d+), (-?\d+\.\d+)\)", stdout)
        self.assertIsNotNone(match, stdout)
        target_x, target_y = float(match.group(1)), float(match.group(2))
        self.assertEqual(float(defaults["target_x"]), target_x)
        self.assertEqual(float(defaults["target_y"]), target_y)
        self.assertEqual(
            float(re.search(r"target_x:=(-?\d+(?:\.\d+)?)", wrapper_block).group(1)),
            target_x,
        )
        self.assertEqual(
            float(re.search(r"target_y:=(-?\d+(?:\.\d+)?)", wrapper_block).group(1)),
            target_y,
        )

    def test_generation_is_deterministic(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            first = self._generate_to(Path(temp_dir) / "a")
            second = self._generate_to(Path(temp_dir) / "b")
            for relative in (
                "worlds/lcorridor/cylinder_lcorridor.world",
                "pcd/cylinder_lcorridor.pcd",
                "worlds/cylinder_positions.txt",
            ):
                self.assertEqual(
                    (first / relative).read_bytes(), (second / relative).read_bytes()
                )

    def test_private_output_root_does_not_modify_package_assets(self):
        package_dir = Path(__file__).resolve().parents[1]
        candidate_assets = (
            package_dir / "worlds" / "cylinder_positions.txt",
            package_dir / "worlds" / "lcorridor" / "cylinder_lcorridor.world",
            package_dir / "pcd" / "cylinder_lcorridor.pcd",
        )

        def _snapshot():
            return {
                path: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in candidate_assets
                if path.exists()
            }

        before = _snapshot()
        with tempfile.TemporaryDirectory() as temp_dir:
            self._generate_to(temp_dir)
        self.assertEqual(before, _snapshot())

    def test_static_pcd_excludes_ceiling(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = self._generate_to(temp_dir, "-H", "2.5")
            points = _read_pcd_points(root / "pcd" / "cylinder_lcorridor.pcd")
            self.assertGreater(len(points), 0)
            # Walls only: no ceiling slab (z > 2.5 m) and no ground-level
            # bottom faces (z == 0); wall top rims end at z == H.
            self.assertLessEqual(points[:, 2].max(), 2.5 + 1e-6)
            self.assertGreater(points[:, 2].min(), 0.0)

    def test_positions_file_is_empty(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = self._generate_to(temp_dir)
            positions = root / "worlds" / "cylinder_positions.txt"
            self.assertTrue(positions.is_file())
            self.assertEqual(positions.read_text(), "")

    def test_rejects_invalid_parameters(self):
        cases = (
            (("--leg1-length", "0"), "leg1 length"),
            (("--leg2-length", "-1"), "leg2 length"),
            (("-W", "nan"), "width"),
            (("-H", "0"), "height"),
            (("--wall-thickness", "0"), "wall thickness"),
            (("--resolution", "0"), "resolution"),
            (("--leg2-length", "1.0", "-W", "3.0"), "half the corridor width"),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            for index, (extra_args, message) in enumerate(cases):
                result = subprocess.run(
                    [
                        sys.executable,
                        str(SCRIPTS_DIR / "generate_lcorridor.py"),
                        "--output-root", str(Path(temp_dir) / f"out{index}"),
                        *extra_args,
                    ],
                    capture_output=True,
                    text=True,
                )
                self.assertNotEqual(result.returncode, 0, extra_args)
                self.assertIn(message, result.stderr, msg=f"{extra_args}: {result.stderr}")


if __name__ == "__main__":
    unittest.main()
