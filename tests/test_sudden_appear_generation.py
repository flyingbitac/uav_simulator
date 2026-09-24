#!/usr/bin/env python3

import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

import numpy as np
import open3d as o3d


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))


def _run_generate(*extra_args):
    return subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / "generate_sudden_appear.py"), *extra_args],
        check=True,
        stdout=subprocess.DEVNULL,
    )


def _load_plugin(world_path):
    world = ET.parse(world_path).getroot().find("world")
    models = world.findall("model")
    plugin_models = [
        (model, model.find("plugin[@filename='libobstaclePathPlugin.so']"))
        for model in models
    ]
    plugin_models = [(model, plugin) for model, plugin in plugin_models if plugin is not None]
    if len(plugin_models) != 1:
        raise AssertionError(
            f"expected exactly one obstacle plugin, got {len(plugin_models)}"
        )
    return world, *plugin_models[0]


def _read_pcd_points(path):
    cloud = o3d.io.read_point_cloud(str(path))
    return np.asarray(cloud.points)


class SuddenAppearGenerationTest(unittest.TestCase):
    def _generate_to(self, root, *extra_args):
        output_root = Path(root) / "out"
        _run_generate("--output-root", str(output_root), *extra_args)
        return output_root

    def test_world_contains_single_sudden_appear_plugin_with_cli_values(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = self._generate_to(
                temp_dir,
                "--trigger-distance", "4.5",
                "--move-speed", "1.25",
                "--travel-distance", "2.5",
                "--direction", "0", "-1", "0",
                "--activation-model", "quad0",
            )
            world_path = root / "worlds" / "sudden_appear" / "sudden_appear.world"
            _, model, plugin = _load_plugin(world_path)

            self.assertEqual(plugin.findtext("motion_type"), "sudden_appear")
            self.assertEqual(plugin.findtext("activation_model"), "quad0")
            self.assertEqual(float(plugin.findtext("trigger_distance")), 4.5)
            self.assertEqual(float(plugin.findtext("velocity")), 1.25)
            self.assertEqual(float(plugin.findtext("travel_distance")), 2.5)
            direction = np.fromstring(plugin.findtext("direction"), sep=" ")
            np.testing.assert_allclose(direction, [0.0, -1.0, 0.0], atol=1e-9)
            self.assertEqual(plugin.findtext("marker_shape"), "box")

            # Non-static, kinematic body driven by the plugin.
            self.assertEqual(model.findtext("static"), "0")
            self.assertEqual(model.findtext("./link/kinematic"), "1")

            size = np.fromstring(
                model.findtext("./link/collision/geometry/box/size"), sep=" "
            )
            np.testing.assert_allclose(size, [0.2, 0.2, 3.0], atol=1e-9)

    def test_generation_is_deterministic(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            first = self._generate_to(Path(temp_dir) / "a")
            second = self._generate_to(Path(temp_dir) / "b")
            for relative in (
                "worlds/sudden_appear/sudden_appear.world",
                "pcd/sudden_appear_static.pcd",
                "worlds/cylinder_positions.txt",
            ):
                self.assertEqual(
                    (first / relative).read_bytes(), (second / relative).read_bytes()
                )

    def test_private_output_root_does_not_modify_package_assets(self):
        package_dir = Path(__file__).resolve().parents[1]
        candidate_assets = (
            package_dir / "worlds" / "cylinder_positions.txt",
            package_dir / "worlds" / "sudden_appear" / "sudden_appear.world",
            package_dir / "pcd" / "sudden_appear_static.pcd",
        )
        tracked = [path for path in candidate_assets if path.exists()]
        self.assertTrue(tracked)
        before = {
            path: hashlib.sha256(path.read_bytes()).hexdigest() for path in tracked
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            self._generate_to(temp_dir)
        after = {
            path: hashlib.sha256(path.read_bytes()).hexdigest() for path in tracked
        }
        self.assertEqual(before, after)

    def test_static_pcd_excludes_dynamic_object(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = self._generate_to(temp_dir, "--object-x", "15.0")
            points = _read_pcd_points(root / "pcd" / "sudden_appear_static.pcd")
            self.assertGreater(len(points), 0)
            # Walls only: no ceiling slab (z > 3 m) and no ground-level
            # bottom faces (z == 0); wall top rims end at z == 3 m.
            self.assertLessEqual(points[:, 2].max(), 3.0 + 1e-6)
            self.assertGreater(points[:, 2].min(), 0.0)
            # The object sits at (15.0, 2.5) in the alcove; static points must
            # not cover the object volume.
            near_object = points[
                (np.abs(points[:, 0] - 15.0) < 0.3)
                & (np.abs(points[:, 1] - 2.5) < 0.3)
            ]
            self.assertEqual(len(near_object), 0)

    def test_rejects_invalid_parameters(self):
        cases = (
            (("--trigger-distance", "0"), "trigger distance"),
            (("--travel-distance", "-1"), "travel distance"),
            (("--move-speed", "nan"), "move speed"),
            (("--alcove-width", "0.1"), "alcove width"),
            (("--alcove-depth", "0.1"), "alcove depth"),
            (("--direction", "0", "0", "0"), "direction"),
            (("--direction", "0", "1", "0"), "exit the alcove"),
            (("--object-y", "5.0"), "object y"),
            (("--object-y", "0.5"), "object y"),
            (("--object-x", "0.3"), "inside the +y wall"),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            for index, (extra_args, message) in enumerate(cases):
                result = subprocess.run(
                    [
                        sys.executable,
                        str(SCRIPTS_DIR / "generate_sudden_appear.py"),
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
