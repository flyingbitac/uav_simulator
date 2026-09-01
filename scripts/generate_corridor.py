#!/usr/bin/env python3

import argparse
from pathlib import Path
from textwrap import dedent
import sys

import numpy as np
import open3d as o3d

sys.path.insert(0, str(Path(__file__).resolve().parent))
from xml_utils import (  # noqa: E402
    get_cube_points,
    get_cube_xml,
    resolve_private_output_root,
    resolve_seed,
)


def _world_xml(models):
    head = dedent("""\
        <?xml version="1.0" ?>
        <sdf version="1.5">
          <world name="default">
            <include>
              <uri>model://sun</uri>
            </include>
            <include>
              <uri>model://ground_plane</uri>
            </include>
            <include>
              <uri>model://asphalt_plane</uri>
              <pose>0 0 -0.000001 0 0 0</pose>
            </include>
    """)
    tail = dedent("""\
            <physics name="default_physics" default="0" type="ode">
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
    return "".join([head, *models, tail])


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a seeded box-obstacle corridor with walls and a ceiling."
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("-L", "--length", type=float, default=40.0)
    parser.add_argument("-W", "--width", type=float, default=5.0)
    parser.add_argument("-H", "--height", type=float, default=3.0)
    parser.add_argument("--wall-thickness", type=float, default=0.15)
    parser.add_argument("-l", "--cell-length", type=float, default=4.0)
    parser.add_argument("--start-clearance", type=float, default=5.0)
    parser.add_argument("--end-clearance", type=float, default=5.0)
    parser.add_argument("--box-min-size", type=float, default=0.6)
    parser.add_argument("--box-max-size", type=float, default=1.2)
    parser.add_argument("--box-min-height", type=float, default=0.8)
    parser.add_argument("--box-max-height", type=float, default=2.2)
    parser.add_argument("--resolution", type=float, default=0.2)
    parser.add_argument("--static", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main():
    args = _parse_args()
    positive = {
        "length": args.length,
        "width": args.width,
        "height": args.height,
        "wall thickness": args.wall_thickness,
        "cell length": args.cell_length,
        "resolution": args.resolution,
    }
    for name, value in positive.items():
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive.")
    if args.start_clearance < 0.0 or args.end_clearance < 0.0:
        raise ValueError("End clearances must be non-negative.")
    usable_length = args.length - args.start_clearance - args.end_clearance
    if usable_length < args.cell_length:
        raise ValueError("Corridor clearances leave no full obstacle cell.")
    if not 0.0 < args.box_min_size <= args.box_max_size < args.width:
        raise ValueError("Box sizes must satisfy 0 < min <= max < corridor width.")
    if not 0.0 < args.box_min_height <= args.box_max_height < args.height:
        raise ValueError("Box heights must satisfy 0 < min <= max < corridor height.")

    package_root = Path(__file__).resolve().parents[1]
    if args.output_root:
        output_root = resolve_private_output_root(args.output_root, package_root)
    else:
        output_root = package_root
    world_dir = output_root / "worlds" / "corridor"
    pcd_dir = output_root / "pcd"
    positions_path = output_root / "worlds" / "cylinder_positions.txt"
    world_dir.mkdir(parents=True, exist_ok=True)
    pcd_dir.mkdir(parents=True, exist_ok=True)

    seed = resolve_seed(args.seed)
    rng = np.random.default_rng(seed)
    boundary_specs = [
        (args.length / 2.0, args.width / 2.0, args.height / 2.0,
         args.length, args.wall_thickness, args.height),
        (args.length / 2.0, -args.width / 2.0, args.height / 2.0,
         args.length, args.wall_thickness, args.height),
        (args.length / 2.0, 0.0, args.height + args.wall_thickness / 2.0,
         args.length, args.width + 2.0 * args.wall_thickness, args.wall_thickness),
    ]

    obstacle_specs = []
    cell_starts = np.arange(
        args.start_clearance,
        args.length - args.end_clearance - args.cell_length + 1e-9,
        args.cell_length,
    )
    for cell_start in cell_starts:
        box_length = rng.uniform(args.box_min_size, args.box_max_size)
        box_width = rng.uniform(args.box_min_size, args.box_max_size)
        box_height = rng.uniform(args.box_min_height, args.box_max_height)
        x_margin = box_length / 2.0
        x = rng.uniform(cell_start + x_margin, cell_start + args.cell_length - x_margin)
        y_limit = args.width / 2.0 - args.wall_thickness - box_width / 2.0
        y = rng.uniform(-y_limit, y_limit)
        obstacle_specs.append((x, y, box_height / 2.0, box_length, box_width, box_height))

    all_specs = boundary_specs + obstacle_specs
    models = [get_cube_xml(*spec) for spec in all_specs]
    point_clouds = [get_cube_points(*spec, resolution=args.resolution) for spec in all_specs]
    points = np.concatenate([item[0] for item in point_clouds], axis=0)
    colors = np.concatenate([item[1] for item in point_clouds], axis=0)

    world_path = world_dir / "cylinder_corridor.world"
    pcd_path = pcd_dir / "cylinder_corridor.pcd"
    world_path.write_text(_world_xml(models), encoding="utf-8")
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(colors)
    if not o3d.io.write_point_cloud(str(pcd_path), cloud):
        raise RuntimeError(f"Failed to write {pcd_path}")
    with positions_path.open("w", encoding="utf-8") as positions:
        for x, y, _, box_length, box_width, _ in obstacle_specs:
            radius = 0.5 * np.hypot(box_length, box_width)
            positions.write(f"{x:.5f},{y:.5f},{radius:.5f}\n")

    print(f"seed: {seed}")
    print(f"boxes: {len(obstacle_specs)}")
    print(f"clear corridor ends: [0, {args.start_clearance}] and "
          f"[{args.length - args.end_clearance}, {args.length}]")
    print(f"World file generated at {world_path.resolve()}")
    print(f"Point cloud file generated at {pcd_path.resolve()}")


if __name__ == "__main__":
    main()
