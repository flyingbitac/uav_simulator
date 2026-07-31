#!/usr/bin/env python3

from textwrap import dedent
import os
import argparse
import sys

import numpy as np
import open3d as o3d

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from xml_utils import get_cylinder_xml, get_cylinder_points

file_path = os.path.dirname(__file__)
world_file_directory = os.path.join(file_path, "../worlds/arena/")
pcd_file_directory = os.path.join(file_path, "../pcd/")
position_file_path = os.path.join(file_path, "../worlds/cylinder_positions.txt")
if not os.path.exists(world_file_directory):
    os.makedirs(world_file_directory)
if not os.path.exists(pcd_file_directory):
    os.makedirs(pcd_file_directory)

def create_arena(cylinders):
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
    with open(os.path.join(world_file_directory, "cylinder_arena.world"), "w") as f:
        f.write(world)

#  #####################
#  #                   #
#  #   1           2   #
#  #                   #
#  #                   #
#  #                   #
#  #                   #
#  #   4           3-d-#
#  #                   #
#  #####################
#  ----------L----------
# `1` is the start point
# order: `1` -> `2` -> `3` -> `4` -> `1`

parser = argparse.ArgumentParser(description="Generate a box arena.")
parser.add_argument("-s", type=float, default=4,   help="Safety range of obstacle generation")
parser.add_argument("-d", type=float, default=10,  help="Distance from the waypoint to the arena border")
parser.add_argument("-l", type=float, default=3.5, help="Length of the grid cells, each cell contains one cylinder")
parser.add_argument("-L", type=float, default=60,  help="Total length of the arena")
parser.add_argument("-r", type=float, default=0.4, help="Minimum radius of the cylinders")
parser.add_argument("-R", type=float, default=0.4, help="Maximum radius of the cylinders")
parser.add_argument("-H", type=float, default=15,  help="Height of the arena")
parser.add_argument("-e", type=float, default=0.,  help="Range of euler angles of the cylinders")
parser.add_argument("--seed", type=int, default=None, help="Optional deterministic random seed")
args, unknown = parser.parse_known_args()

S = args.s
d = args.d
l = args.l
L = args.L
R_min = args.r
R_max = args.R
assert R_max >= R_min
H = args.H
euler_range_deg = args.e
ratio = 1.0
Nx, Ny = int(L / l), int(L / l)

if __name__ == "__main__":
    rng = np.random.default_rng(args.seed)
    W = ratio * L
    if L <= 2 * d or W <= 2 * d:
        raise ValueError("Arena size must be larger than 2*d in both dimensions.")

    # Waypoints are inset by d from the arena border, with waypoint 1 at (0, 0).
    waypoints = np.array([
        [0.0,                0.0],
        [L - 2 * d,          0.0],
        [L - 2 * d,         -(W - 2 * d)],
        [0.0,               -(W - 2 * d)],
    ])
    print("suggested waypoints (1->2->3->4->1):")
    for idx, (wx, wy) in enumerate(waypoints, start=1):
        print(f"{idx}: ({wx:.2f}, {wy:.2f})")

    # Shift the arena so waypoint 1 sits at (0, 0).
    x_base = np.arange(Nx) * l - d
    y_base = np.arange(Ny) * l - (W - d)
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

    # Exclude obstacles within radius S of any waypoint.
    dx = xs[..., None] - waypoints[:, 0]
    dy = ys[..., None] - waypoints[:, 1]
    inside_exclusion = (dx ** 2 + dy ** 2) <= (S ** 2)
    inside_exclusion = np.any(inside_exclusion, axis=-1)

    cylinders, points, colors = [], [], []
    with open(position_file_path, "w") as f:
        for i in range(Nx):
            for j in range(Ny):
                if inside_exclusion[i, j]:
                    continue
                cylinder_xml, (x, y, r) = get_cylinder_xml(xs[i, j], ys[i, j], 0, rs[i, j], H, eulers[i, j])
                point, color = get_cylinder_points(xs[i, j], ys[i, j], 0, rs[i, j], H, eulers[i, j], resolution=0.2)
                points.append(point)
                colors.append(color)
                cylinders.append(cylinder_xml)
                f.write(f"{x:.5f},{y:.5f},{r:.5f}\n")
    create_arena(cylinders)
    print("World file generated at", os.path.abspath(os.path.join(world_file_directory, "cylinder_arena.world")))
    pcd = o3d.geometry.PointCloud()
    points = np.concatenate(points, axis=0)
    colors = np.concatenate(colors, axis=0)
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    assert pcd.has_colors()
    o3d.io.write_point_cloud(os.path.join(pcd_file_directory, "cylinder_arena.pcd"), pcd)
    print("Point cloud file generated at", os.path.abspath(os.path.join(pcd_file_directory, "cylinder_arena.pcd")))
