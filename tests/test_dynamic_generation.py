#!/usr/bin/env python3

import sys
import tempfile
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

import numpy as np


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from xml_utils import (  # noqa: E402
    _dynamic_cylinder_clearance,
    _ray_respects_protected_points,
    _write_binary_pcd,
    generate_dynamic_cylinder_world,
)


def _cylinder_model(index, x, y):
    return f"""
    <model name="unit_cylinder_{index}">
      <pose>{x} {y} 7.7 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry><cylinder><radius>0.3</radius><length>15</length></cylinder></geometry>
        </collision>
        <kinematic>0</kinematic>
      </link>
      <static>1</static>
    </model>
    """


class DynamicGenerationTest(unittest.TestCase):
    def _generate(
        self,
        root,
        seed,
        suffix,
        linear_speed=(0.1, 2.0),
        activation_distance=20.0,
    ):
        source_world = root / "source.world"
        source_pcd = root / "source.pcd"
        if not source_world.exists():
            source_world.write_text(
                "<sdf><world name='default'>"
                + _cylinder_model(0, 10.0, 10.0)
                + _cylinder_model(1, 30.0, 10.0)
                + "</world></sdf>",
                encoding="utf-8",
            )
            header = [
                b"VERSION .7\n",
                b"FIELDS x y z\n",
                b"SIZE 4 4 4\n",
                b"TYPE F F F\n",
                b"COUNT 1 1 1\n",
                b"WIDTH 4\n",
                b"HEIGHT 1\n",
                b"POINTS 4\n",
                b"DATA binary\n",
            ]
            records = np.zeros(
                4,
                dtype=np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4")]),
            )
            _write_binary_pcd(source_pcd, header, records)
        output_world = root / f"dynamic_{suffix}.world"
        output_pcd = root / f"dynamic_{suffix}.pcd"
        result = generate_dynamic_cylinder_world(
            scene_name="test",
            source_world=source_world,
            source_pcd=source_pcd,
            output_world=output_world,
            output_pcd=output_pcd,
            seed=seed,
            dynamic_ratio=1.0,
            linear_speed=linear_speed,
            activation_distance=activation_distance,
            protected_points=((0.0, 0.0), (50.0, 0.0)),
            cylinder_point_counts=(2, 2),
        )
        return output_world, output_pcd, result

    def test_proximity_world_is_seeded_and_contains_only_one_way_motion(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            world_a, pcd_a, result_a = self._generate(root, 7, "a")
            world_b, pcd_b, result_b = self._generate(root, 7, "b")
            world_c, _, _ = self._generate(root, 8, "c")

            self.assertEqual(world_a.read_bytes(), world_b.read_bytes())
            self.assertEqual(pcd_a.read_bytes(), pcd_b.read_bytes())
            self.assertNotEqual(world_a.read_bytes(), world_c.read_bytes())
            self.assertEqual(result_a, {**result_b, "world": str(world_a), "pcd": str(pcd_a)})
            self.assertEqual(result_a["mode_counts"], {"proximity_linear": 2})
            self.assertEqual(result_a["static_pcd_points"], 0)

            tree = ET.parse(world_a)
            plugins = tree.findall("./world/model/plugin")
            self.assertEqual(len(plugins), 2)
            for plugin in plugins:
                self.assertEqual(plugin.findtext("motion_type"), "proximity_linear")
                self.assertEqual(plugin.findtext("activation_model"), "iris")
                self.assertEqual(float(plugin.findtext("activation_distance")), 20.0)
                direction = np.fromstring(plugin.findtext("direction"), sep=" ")
                self.assertAlmostEqual(float(np.linalg.norm(direction)), 1.0, places=8)
                for legacy_tag in ("line_start", "line_end", "circle_center", "phase"):
                    self.assertIsNone(plugin.find(legacy_tag))

    def test_forward_ray_rejects_protected_points_without_blocking_behind(self):
        origin = np.asarray([10.0, 0.0, 0.0])
        forward = np.asarray([1.0, 0.0, 0.0])
        self.assertFalse(
            _ray_respects_protected_points(origin, forward, ((20.0, 0.0),), 1.0)
        )
        self.assertTrue(
            _ray_respects_protected_points(origin, forward, ((0.0, 0.0),), 1.0)
        )

    def test_rejects_non_finite_motion_parameters(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with self.assertRaisesRegex(ValueError, "linear_speed"):
                self._generate(root, 0, "speed", linear_speed=(0.1, float("inf")))
            with self.assertRaisesRegex(ValueError, "activation_distance"):
                self._generate(root, 0, "distance", activation_distance=float("nan"))

    def test_checked_in_dynamic_worlds_use_safe_proximity_rays(self):
        package_dir = Path(__file__).resolve().parents[1]
        scenes = {
            "arena": ((0.0, 0.0), (40.0, 0.0), (40.0, -40.0), (0.0, -40.0)),
            "forest": ((0.0, 0.0), (50.0, 0.0)),
        }
        for scene, protected_points in scenes.items():
            world = ET.parse(
                package_dir / "worlds" / scene / f"{scene}_dynamic.world"
            ).getroot().find("world")
            self.assertIsNotNone(world)
            plugins = []
            for model in world.findall("model"):
                plugin = model.find("plugin[@filename='libobstaclePathPlugin.so']")
                if plugin is None:
                    continue
                plugins.append(plugin)
                pose, _, _, clearance = _dynamic_cylinder_clearance(model)
                direction = np.fromstring(plugin.findtext("direction"), sep=" ")
                self.assertEqual(plugin.findtext("motion_type"), "proximity_linear")
                self.assertTrue(
                    _ray_respects_protected_points(
                        pose, direction, protected_points, clearance
                    ),
                    model.get("name"),
                )
            self.assertTrue(plugins, scene)


if __name__ == "__main__":
    unittest.main()
