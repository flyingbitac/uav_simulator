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
        description="Generate a straight dead-end corridor with walls and a "
        "ceiling. The free space is one rectangular tube of width W and height "
        "H running along +x from the open mouth at x=0, closed at x=length by "
        "a far wall. The goal sits target-offset metres behind that wall, so "
        "the only way to reach it is through the wall."
    )
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("-L", "--length", type=float, default=20.0)
    parser.add_argument("-W", "--width", type=float, default=2.0)
    parser.add_argument("-H", "--height", type=float, default=2.0)
    parser.add_argument("--wall-thickness", type=float, default=0.15)
    parser.add_argument("--target-offset", type=float, default=3.0)
    parser.add_argument("--resolution", type=float, default=0.2)
    return parser.parse_args()


def _build_specs(length, width, height, thickness):
    """Box specs (cx, cy, cz, l, w, h); the open mouth sits at x = 0."""
    half_width = width / 2.0
    # Walls span z in [0, height]; the ceiling sits above at z in [height,
    # height+t]. Side walls run from t behind the mouth up to the far wall,
    # which closes the whole cross-section at x in [length, length+t].
    walls = [
        # Left wall.
        ((length - thickness) / 2.0, -half_width - thickness / 2.0,
         height / 2.0, length + thickness, thickness, height),
        # Right wall.
        ((length - thickness) / 2.0, half_width + thickness / 2.0,
         height / 2.0, length + thickness, thickness, height),
        # Far wall: the dead end.
        (length + thickness / 2.0, 0.0,
         height / 2.0, thickness, width + 2.0 * thickness, height),
    ]
    ceilings = [
        # One slab over mouth, tube and far wall so no seam opens.
        (length / 2.0, 0.0,
         height + thickness / 2.0,
         length + 2.0 * thickness, width + 2.0 * thickness, thickness),
    ]
    return walls, ceilings


def main():
    args = _parse_args()
    positive = {
        "length": args.length,
        "width": args.width,
        "height": args.height,
        "wall thickness": args.wall_thickness,
        "target offset": args.target_offset,
        "resolution": args.resolution,
    }
    for name, value in positive.items():
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive.")
    if args.length <= args.width / 2.0:
        raise ValueError(
            "length must exceed half the corridor width, otherwise the tube "
            "has no straight section."
        )

    package_root = Path(__file__).resolve().parents[1]
    if args.output_root:
        output_root = resolve_private_output_root(args.output_root, package_root)
    else:
        output_root = package_root
    world_dir = output_root / "worlds" / "deadend"
    pcd_dir = output_root / "pcd"
    positions_path = output_root / "worlds" / "cylinder_positions.txt"
    world_dir.mkdir(parents=True, exist_ok=True)
    pcd_dir.mkdir(parents=True, exist_ok=True)

    wall_specs, ceiling_specs = _build_specs(
        args.length, args.width, args.height, args.wall_thickness,
    )

    models = [get_cube_xml(*spec) for spec in wall_specs + ceiling_specs]
    # The static pcd is display-only: wall boxes only (vertical faces plus
    # their top rims), with the ceiling slab and ground-level bottom faces
    # dropped so rviz shows the dead-end outline alone.
    point_clouds = [
        get_cube_points(*spec, resolution=args.resolution) for spec in wall_specs
    ]
    points = np.concatenate([item[0] for item in point_clouds], axis=0)
    colors = np.concatenate([item[1] for item in point_clouds], axis=0)
    keep = points[:, 2] > 0.0
    points = points[keep]
    colors = colors[keep]

    world_path = world_dir / "cylinder_deadend.world"
    pcd_path = pcd_dir / "cylinder_deadend.pcd"
    world_path.write_text(_world_xml(models), encoding="utf-8")
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(colors)
    if not o3d.io.write_point_cloud(str(pcd_path), cloud):
        raise RuntimeError(f"Failed to write {pcd_path}")
    with positions_path.open("w", encoding="utf-8"):
        pass

    target_x = args.length + args.wall_thickness + args.target_offset
    print(f"tube: {args.length} m along +x from the mouth at x=0, "
          f"closed at x={args.length:.3f}")
    print(f"free section: {args.width} m wide x {args.height} m high, "
          f"wall thickness {args.wall_thickness} m")
    print(f"home: (0.000, 0.000), target: ({target_x:.3f}, {0.000:.3f})")
    print(f"target sits {args.target_offset} m behind the far wall "
          f"(outer face x={target_x - args.target_offset:.3f})")
    print("cylinder_positions.txt: empty (no obstacles)")
    print(f"static pcd: walls only ({len(points)} points, no ceiling or "
          "ground faces)")
    print(f"World file generated at {world_path.resolve()}")
    print(f"Point cloud file generated at {pcd_path.resolve()}")


if __name__ == "__main__":
    main()
