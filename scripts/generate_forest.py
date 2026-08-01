#!/usr/bin/env python3

from textwrap import dedent
import os
import argparse
import json
import sys

import numpy as np
import open3d as o3d

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from xml_utils import (
    generate_dynamic_cylinder_world,
    get_cylinder_points,
    get_cylinder_xml,
    resolve_seed,
)

file_path = os.path.dirname(__file__)
world_file_directory = os.path.join(file_path, "../worlds/forest/")
pcd_file_directory = os.path.join(file_path, "../pcd/")
position_file_path = os.path.join(file_path, "../worlds/cylinder_positions.txt")
if not os.path.exists(world_file_directory):
    os.makedirs(world_file_directory)
if not os.path.exists(pcd_file_directory):
    os.makedirs(pcd_file_directory)

def create_forest(cylinders):
    head = dedent("""
        <?xml version="1.0" ?>
        <sdf version="1.5">
            <world name="default">
            <!-- A global light source -->
            <include>
                <uri>model://sun</uri>
            </include>
            <!-- A ground plane -->
            <include>
                <uri>model://ground_plane</uri>
            </include>
            <include>
                <uri>model://asphalt_plane</uri>
                <pose>0 0 -0.000001 0 0 0</pose>
            </include>
    """)
    tail = dedent("""
            <physics name='default_physics' default='0' type='ode'>
              <gravity>0 0 -9.8066</gravity>
              <ode>
                <solver>
                  <type>quick</type>
                  <iters>10</iters>
                  <sor>1.3</sor>
                  <use_dynamic_moi_rescaling>0</use_dynamic_moi_rescaling>
                </solver>
                <constraints>
                  <cfm>0</cfm>
                  <erp>0.2</erp>
                  <contact_max_correcting_vel>100</contact_max_correcting_vel>
                  <contact_surface_layer>0.001</contact_surface_layer>
                </constraints>
              </ode>
              <max_step_size>0.004</max_step_size>
              <real_time_factor>1</real_time_factor>
              <real_time_update_rate>250</real_time_update_rate>
              <magnetic_field>6.0e-6 2.3e-5 -4.2e-5</magnetic_field>
            </physics>
          </world>
        </sdf>
    """)
    world = "".join([head] + cylinders + [tail])
    with open(os.path.join(world_file_directory, "cylinder_forest.world"), "w") as f:
        f.write(world)

parser = argparse.ArgumentParser(description="Generate a cluttered world with cylinders.")
parser.add_argument("-b", type=float, default=5,   help="Start position offset")
parser.add_argument("-l", "--cell-length", type=float, default=4, help="Length of the grid cells")
parser.add_argument("-L", type=float, default=40,  help="Total length of the field")
parser.add_argument("-r", "--min-radius", type=float, default=0.3, help="Minimum radius of the cylinders")
parser.add_argument("-R", "--max-radius", type=float, default=0.3, help="Maximum radius of the cylinders")
parser.add_argument("-H", type=float, default=15,  help="Height of the cylinders")
parser.add_argument("-e", "--euler-range", type=float, default=10., help="Range of euler angles of the cylinders")
parser.add_argument("--seed", type=int, default=None, help="Optional deterministic random seed")
dynamic_group = parser.add_mutually_exclusive_group()
dynamic_group.add_argument(
    "--dynamic", action="store_true", help="Also generate the dynamic forest"
)
dynamic_group.add_argument(
    "--static", dest="dynamic", action="store_false", help="Generate only the static forest"
)
parser.set_defaults(dynamic=True)
parser.add_argument(
    "--dynamic-ratio",
    type=float,
    default=0.5,
    help="Fraction of cylinders converted to dynamic obstacles",
)
args, unknown = parser.parse_known_args()

B = args.b
l = args.cell_length
L = args.L
R_min = args.min_radius
R_max = args.max_radius
assert R_max >= R_min
H = args.H
euler_range_deg = args.euler_range
ratio = 0.6
Nx, Ny = int(L / l), int(ratio * L / l)

if __name__ == "__main__":
    if not 0.0 <= args.dynamic_ratio <= 1.0:
        raise ValueError("--dynamic-ratio must be within [0, 1].")
    seed = resolve_seed(args.seed)
    rng = np.random.default_rng(seed)
    print(f"seed: {seed}")
    print(f"suggested start position: (0, 0)")
    print(f"suggested target position: ({L + 2 * B}, 0)")
    x_base = np.arange(Nx) * l + B
    # y_base = np.arange(Ny) * L + B
    y_base = np.arange(Ny) * l - (Ny / 2) * l
    x_rand = (rng.random((Nx, Ny)) * (1 - 2 * (R_max / l)) + R_max / l) * l
    y_rand = (rng.random((Nx, Ny)) * (1 - 2 * (R_max / l)) + R_max / l) * l
    x_base, y_base = np.meshgrid(x_base, y_base, indexing='ij')
    xs, ys = x_base + x_rand, y_base + y_rand
    rs = rng.random((Nx, Ny)) * (R_max - R_min) + R_min
    rp_range = euler_range_deg * np.pi / 180.0
    eulers = np.concatenate([
        rng.random((Nx, Ny, 2)) * 2 * rp_range - rp_range,
        np.zeros((Nx, Ny, 1)),
    ], axis=-1)

    cylinders, points, colors = [], [], []
    with open(position_file_path, "w") as f:
        for i in range(Nx):
            for j in range(Ny):
                cylinder_xml, (x, y, r) = get_cylinder_xml(xs[i, j], ys[i, j], 0, rs[i, j], H, eulers[i, j])
                point, color = get_cylinder_points(xs[i, j], ys[i, j], 0, rs[i, j], H, eulers[i, j], resolution=0.2)
                points.append(point)
                colors.append(color)
                cylinders.append(cylinder_xml)
                f.write(f"{x:.5f},{y:.5f},{r:.5f}\n")
    create_forest(cylinders)
    print("World file generated at", os.path.abspath(os.path.join(world_file_directory, "cylinder_forest.world")))
    pcd = o3d.geometry.PointCloud()
    point_counts = [len(point) for point in points]
    points = np.concatenate(points, axis=0)
    colors = np.concatenate(colors, axis=0)
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    assert pcd.has_colors()
    o3d.io.write_point_cloud(os.path.join(pcd_file_directory, "cylinder_forest.pcd"), pcd)
    print("Point cloud file generated at", os.path.abspath(os.path.join(pcd_file_directory, "cylinder_forest.pcd")))
    if args.dynamic:
        result = generate_dynamic_cylinder_world(
            scene_name="forest",
            source_world=os.path.join(world_file_directory, "cylinder_forest.world"),
            source_pcd=os.path.join(pcd_file_directory, "cylinder_forest.pcd"),
            output_world=os.path.join(world_file_directory, "forest_dynamic.world"),
            output_pcd=os.path.join(pcd_file_directory, "forest_dynamic_static.pcd"),
            seed=seed,
            dynamic_ratio=args.dynamic_ratio,
            linear_speed=(0.1, 2.0),
            circular_angular_speed=(0.1, 1.0),
            protected_points=((0.0, 0.0), (L + 2 * B, 0.0)),
            cylinder_point_counts=point_counts,
            seed_offset=1_000_003,
        )
        print(json.dumps([result], indent=2, sort_keys=True))
