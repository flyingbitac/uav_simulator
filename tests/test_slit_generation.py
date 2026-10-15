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
        [sys.executable, str(SCRIPTS_DIR / "generate_slit.py"), *extra_args],
        check=True,
        stdout=subprocess.DEVNULL,
    )


def _run_generate_capture(*extra_args):
    return subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / "generate_slit.py"), *extra_args],
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


class SlitGenerationTest(unittest.TestCase):
    def _generate_to(self, root, *extra_args):
        output_root = Path(root) / "out"
        _run_generate("--output-root", str(output_root), *extra_args)
        return output_root

    def test_world_boxes_match_hand_computed_table(self):
        # Hand-computed from the design spec for length=8, W=2, H=2.5, t=0.2
        # and the default 1 x 1 m slit centred on the tube axis: two side
        # walls, the four slit-wall pieces framing the opening (pillars, sill,
        # lintel) at the mid-corridor default x = length / 2, then one ceiling
        # slab as (cx, cy, cz, l, w, h).
        expected = (
            (3.9, -1.1, 1.25, 8.2, 0.2, 2.5),
            (3.9, 1.1, 1.25, 8.2, 0.2, 2.5),
            (4.1, -0.85, 1.25, 0.2, 0.7, 2.5),
            (4.1, 0.85, 1.25, 0.2, 0.7, 2.5),
            (4.1, 0.0, 0.375, 0.2, 1.0, 0.75),
            (4.1, 0.0, 2.125, 0.2, 1.0, 0.75),
            (4.0, 0.0, 2.6, 8.4, 2.4, 0.2),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = self._generate_to(
                temp_dir,
                "-L", "8.0",
                "-W", "2.0",
                "-H", "2.5",
                "--wall-thickness", "0.2",
            )
            boxes = _load_boxes(
                root / "worlds" / "slit" / "cylinder_slit.world"
            )
            self.assertEqual(len(boxes), len(expected))
            for (_, pose, size), (cx, cy, cz, l, w, h) in zip(boxes, expected):
                np.testing.assert_allclose(pose[:3], [cx, cy, cz], atol=5e-4)
                np.testing.assert_allclose(size, [l, w, h], atol=5e-4)

    def test_slit_is_the_only_way_through_the_far_wall(self):
        # The whole point of the scene: the tube centre line from the mouth to
        # the slit wall is free, the slit wall blocks everything except the
        # opening, and only the opening lets a body through to the goal side,
        # which stays enclosed past the wall.
        length, width, height, t = 20.0, 2.0, 2.0, 0.15
        half = width / 2.0
        wall_front = length / 2.0
        with tempfile.TemporaryDirectory() as temp_dir:
            root = self._generate_to(temp_dir)
            boxes = _load_boxes(
                root / "worlds" / "slit" / "cylinder_slit.world"
            )

        approach = [(x, 0.0, height / 2.0)
                    for x in np.linspace(0.05, wall_front - 0.05, 40)]
        self.assertFalse(
            _inside_any_box(approach, boxes).any(),
            "the straight approach to the slit wall must be free",
        )

        wall_x = wall_front + t / 2.0
        blocked = [
            (wall_x, half - 0.05, height / 2.0),
            (wall_x, -half + 0.05, height / 2.0),
            (wall_x, half + t - 0.01, height / 2.0),
            (wall_x, -half - t + 0.01, height / 2.0),
            (wall_x, 0.0, 0.02),
            (wall_x, 0.0, height - 0.02),
        ]
        self.assertTrue(
            _inside_any_box(blocked, boxes).all(),
            "the slit wall must block the tube cross-section around the slit",
        )

        # The approach, the slit opening itself and the exit behind the wall
        # are all free for a body smaller than the opening.
        slit = [
            (x, y, z)
            for x in np.linspace(wall_front - 0.1, wall_front + t + 0.1, 5)
            for y in np.linspace(-0.45, 0.45, 5)
            for z in np.linspace(0.55, 1.45, 5)
        ]
        self.assertFalse(
            _inside_any_box(slit, boxes).any(),
            "a 1 x 1 m slit must stay open through the wall thickness",
        )
        exit_line = [(x, 0.0, height / 2.0)
                     for x in np.linspace(wall_front + t + 0.05, length, 20)]
        self.assertFalse(
            _inside_any_box(exit_line, boxes).any(),
            "the goal side of the wall must be free",
        )

    def test_free_space_and_wall_bands(self):
        length, width, height, t = 8.0, 2.0, 2.5, 0.2
        half = width / 2.0
        with tempfile.TemporaryDirectory() as temp_dir:
            root = self._generate_to(
                temp_dir,
                "-L", str(length),
                "-W", str(width),
                "-H", str(height),
                "--wall-thickness", str(t),
            )
            boxes = _load_boxes(
                root / "worlds" / "slit" / "cylinder_slit.world"
            )

        zs = np.linspace(0.2, height - 0.2, 5)
        free = []
        # Skip the slit wall's own x band when probing the open tube.
        free_x = np.concatenate([
            np.linspace(0.1, length / 2.0 - t - 0.1, 10),
            np.linspace(length / 2.0 + t + 0.1, length - 0.1, 10),
        ])
        for x in free_x:
            for y in (-half + 0.05, 0.0, half - 0.05):
                for z in zs:
                    free.append((x, y, z))
        self.assertFalse(
            _inside_any_box(free, boxes).any(),
            "free tube samples must not fall inside any wall/ceiling box",
        )

        # Wall bands must be solid: no gaps in the enclosure.
        bands = []
        for x in np.linspace(0.05, length - 0.05, 25):
            for y in np.linspace(-half - t + 0.01, -half - 0.01, 4):
                bands.append((x, y, height / 2.0))
            for y in np.linspace(half + 0.01, half + t - 0.01, 4):
                bands.append((x, y, height / 2.0))
        # Mid-corridor slit wall: the pillars beside the opening...
        for y in (-0.9, -0.6, 0.6, 0.9):
            for x in np.linspace(length / 2.0 + 0.01, length / 2.0 + t - 0.01, 4):
                bands.append((x, y, height / 2.0))
        # ...and the sill/lintel above and below its opening.
        for y in (-0.4, 0.0, 0.4):
            for x in np.linspace(length / 2.0 + 0.01, length / 2.0 + t - 0.01, 4):
                for z in (0.05, 0.6, 1.9, 2.45):
                    bands.append((x, y, z))
        self.assertTrue(
            _inside_any_box(bands, boxes).all(),
            "wall band samples must all lie inside some wall box",
        )

        # The ceiling slab must cover mouth, tube and far wall without seams.
        zc = height + t / 2.0
        ceiling = [
            (x, y, zc)
            for x in np.linspace(-t + 0.01, length + t - 0.01, 25)
            for y in np.linspace(-half - t + 0.01, half + t - 0.01, 10)
        ]
        self.assertTrue(
            _inside_any_box(ceiling, boxes).all(),
            "ceiling samples must all lie inside the ceiling box",
        )

    def test_reported_home_and_target_follow_geometry(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            stdout = _run_generate_capture(
                "--output-root", str(Path(temp_dir) / "out"),
                "-L", "8", "--wall-thickness", "0.2", "--target-offset", "2.5",
            )
        self.assertIn("home: (0.000, 0.000)", stdout)
        # The goal sits at the far end of the tube (8 m) less the 2.5 m offset.
        self.assertIn("target: (5.500, 0.000)", stdout)

    def test_target_tracks_the_slit_center(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            stdout = _run_generate_capture(
                "--output-root", str(Path(temp_dir) / "out"),
                "--slit-center-y", "0.3",
            )
        self.assertIn("slit centred at (y=0.300", stdout)
        # Default geometry: the goal is the far end of the 20 m tube.
        self.assertIn("target: (20.000, 0.300)", stdout)

    def test_launch_defaults_agree_with_generator_defaults(self):
        # yopo_oa_slit.launch and the official-MINCO wrapper both ship target
        # defaults that must match the generator's default dimensions
        # (length 20, wall thickness 0.15, target offset 3, slit centred on
        # the tube axis).
        launch = ET.parse(LAUNCH_DIR / "yopo_oa_slit.launch").getroot()
        defaults = {
            arg.get("name"): arg.get("default")
            for arg in launch.findall("arg")
            if arg.get("name") in ("target_x", "target_y")
        }
        wrapper = (
            Path(__file__).resolve().parents[2]
            / "baseline" / "scripts" / "evaluate_yopo_minco.py"
        ).read_text()
        wrapper_block = wrapper.split('elif args.scene == "slit":', 1)[1]
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
                "worlds/slit/cylinder_slit.world",
                "pcd/cylinder_slit.pcd",
                "worlds/cylinder_positions.txt",
            ):
                self.assertEqual(
                    (first / relative).read_bytes(), (second / relative).read_bytes()
                )

    def test_private_output_root_does_not_modify_package_assets(self):
        package_dir = Path(__file__).resolve().parents[1]
        candidate_assets = (
            package_dir / "worlds" / "cylinder_positions.txt",
            package_dir / "worlds" / "slit" / "cylinder_slit.world",
            package_dir / "pcd" / "cylinder_slit.pcd",
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
            points = _read_pcd_points(root / "pcd" / "cylinder_slit.pcd")
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
            (("-L", "0"), "length"),
            (("-W", "nan"), "width"),
            (("-H", "0"), "height"),
            (("--wall-thickness", "0"), "wall thickness"),
            (("--target-offset", "-1"), "target offset"),
            (("--slit-width", "0"), "slit width"),
            (("--slit-height", "nan"), "slit height"),
            (("--slit-center-y", "nan"), "slit center y"),
            (("--slit-center-z", "inf"), "slit center z"),
            (("--resolution", "0"), "resolution"),
            (("-L", "1.0", "-W", "3.0"), "half the corridor width"),
            (("--slit-width", "2.6"), "strictly inside the far wall"),
            (("--slit-height", "2.0"), "strictly inside the far wall"),
            (("--slit-center-y", "1.5"), "strictly inside the far wall"),
            (("--slit-height", "1.0", "--slit-center-z", "0.3"),
             "strictly inside the far wall"),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            for index, (extra_args, message) in enumerate(cases):
                result = subprocess.run(
                    [
                        sys.executable,
                        str(SCRIPTS_DIR / "generate_slit.py"),
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
