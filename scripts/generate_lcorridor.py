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
        description="Generate an L-shaped right-angle corridor with walls and a "
        "ceiling. The free space is the union of two rectangular tubes with a "
        "square elbow: leg 1 runs along +x from the start, leg 2 turns "
        "(left: +y, right: -y) at x=leg1-length and ends at the goal."
    )
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--leg1-length", type=float, default=15.0)
    parser.add_argument("--leg2-length", type=float, default=15.0)
    parser.add_argument("-W", "--width", type=float, default=2.0)
    parser.add_argument("-H", "--height", type=float, default=2.0)
    parser.add_argument("--wall-thickness", type=float, default=0.15)
    parser.add_argument("--turn-direction", choices=("left", "right"), default="left")
    parser.add_argument("--resolution", type=float, default=0.2)
    return parser.parse_args()


def _build_specs(leg1, leg2, width, height, thickness, sign):
    """Box specs (cx, cy, cz, l, w, h); sign = +1 for a left turn, -1 right."""
    half_width = width / 2.0
    inner_x = leg1 - half_width
    outer_x = leg1 + half_width
    # Walls span z in [0, height]; ceilings sit above at z in [height, height+t].
    walls = [
        # South wall: outer side of leg 1, running up to the outer corner.
        ((outer_x - thickness) / 2.0, sign * (-half_width - thickness / 2.0),
         height / 2.0, outer_x + thickness, thickness, height),
        # North wall: inner side of leg 1, stopping at the inner corner.
        ((inner_x - thickness) / 2.0, sign * (half_width + thickness / 2.0),
         height / 2.0, inner_x + thickness, thickness, height),
        # East wall: outer side of leg 2, from the outer corner to the goal end.
        (outer_x + thickness / 2.0, sign * (leg2 - half_width - thickness) / 2.0,
         height / 2.0, thickness, leg2 + half_width + thickness, height),
        # West wall: inner side of leg 2, from the inner corner to the goal end.
        (inner_x - thickness / 2.0, sign * (half_width + leg2) / 2.0,
         height / 2.0, thickness, leg2 - half_width, height),
    ]
    ceilings = [
        # Over leg 1 (including the elbow).
        (outer_x / 2.0, 0.0,
         height + thickness / 2.0,
         outer_x + 2.0 * thickness, width + 2.0 * thickness, thickness),
        # Over leg 2, overlapping the elbow slab so no seam opens.
        (leg1, sign * (half_width + leg2) / 2.0,
         height + thickness / 2.0,
         width + 2.0 * thickness, leg2 - half_width + 2.0 * thickness, thickness),
    ]
    return walls, ceilings


def main():
    args = _parse_args()
    positive = {
        "leg1 length": args.leg1_length,
        "leg2 length": args.leg2_length,
        "width": args.width,
        "height": args.height,
        "wall thickness": args.wall_thickness,
        "resolution": args.resolution,
    }
    for name, value in positive.items():
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive.")
    half_width = args.width / 2.0
    if args.leg1_length <= half_width or args.leg2_length <= half_width:
        raise ValueError(
            "leg lengths must each exceed half the corridor width, otherwise "
            "the elbow has no straight section."
        )

    package_root = Path(__file__).resolve().parents[1]
    if args.output_root:
        output_root = resolve_private_output_root(args.output_root, package_root)
    else:
        output_root = package_root
    world_dir = output_root / "worlds" / "lcorridor"
    pcd_dir = output_root / "pcd"
    positions_path = output_root / "worlds" / "cylinder_positions.txt"
    world_dir.mkdir(parents=True, exist_ok=True)
    pcd_dir.mkdir(parents=True, exist_ok=True)

    sign = 1.0 if args.turn_direction == "left" else -1.0
    wall_specs, ceiling_specs = _build_specs(
        args.leg1_length, args.leg2_length, args.width, args.height,
        args.wall_thickness, sign,
    )

    models = [get_cube_xml(*spec) for spec in wall_specs + ceiling_specs]
    # The static pcd is display-only: wall boxes only (vertical faces plus
    # their top rims), with ceiling slabs and ground-level bottom faces dropped
    # so rviz shows the corridor outline alone.
    point_clouds = [
        get_cube_points(*spec, resolution=args.resolution) for spec in wall_specs
    ]
    points = np.concatenate([item[0] for item in point_clouds], axis=0)
    colors = np.concatenate([item[1] for item in point_clouds], axis=0)
    keep = points[:, 2] > 0.0
    points = points[keep]
    colors = colors[keep]

    world_path = world_dir / "cylinder_lcorridor.world"
    pcd_path = pcd_dir / "cylinder_lcorridor.pcd"
    world_path.write_text(_world_xml(models), encoding="utf-8")
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(colors)
    if not o3d.io.write_point_cloud(str(pcd_path), cloud):
        raise RuntimeError(f"Failed to write {pcd_path}")
    with positions_path.open("w", encoding="utf-8"):
        pass

    goal_y = sign * args.leg2_length
    print(f"turn: {args.turn_direction}")
    print(f"leg1: {args.leg1_length} m along +x, leg2: {args.leg2_length} m "
          f"along {'+' if sign > 0 else '-'}y")
    print(f"free section: {args.width} m wide x {args.height} m high, "
          f"wall thickness {args.wall_thickness} m")
    print(f"home: (0.000, 0.000), target: ({args.leg1_length:.3f}, {goal_y:.3f})")
    print("cylinder_positions.txt: empty (no obstacles)")
    print(f"static pcd: walls only ({len(points)} points, no ceiling or "
          "ground faces)")
    print(f"World file generated at {world_path.resolve()}")
    print(f"Point cloud file generated at {pcd_path.resolve()}")


if __name__ == "__main__":
    main()
