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
        "ceiling, whose far wall carries a rectangular opening (slit) the "
        "drone can fly through. The free space is one rectangular tube of "
        "width W and height H running along +x from the open mouth at x=0, "
        "closed at x=length by a wall with a slit-width x slit-height hole. "
        "The goal sits target-offset metres behind that wall, so the only way "
        "to reach it is through the slit."
    )
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("-L", "--length", type=float, default=20.0)
    parser.add_argument("-W", "--width", type=float, default=2.0)
    parser.add_argument("-H", "--height", type=float, default=2.0)
    parser.add_argument("--wall-thickness", type=float, default=0.15)
    parser.add_argument(
        "--target-offset", type=float, default=0.0,
        help="how far short of the tube's far end the goal sits (default: "
             "0, i.e. the goal is at the end of the corridor).",
    )
    parser.add_argument("--slit-width", type=float, default=1.0)
    parser.add_argument("--slit-height", type=float, default=1.0)
    parser.add_argument("--slit-center-y", type=float, default=0.0)
    parser.add_argument("--slit-center-z", type=float, default=None,
                        help="default: half the tube height")
    parser.add_argument(
        "--wall-x", type=float, default=None,
        help="x of the slit wall along the tube (default: halfway down the "
             "tube, --length/2). Pass --length to put it back at the far end, "
             "or any smaller value to move it. The tube still runs to the far "
             "end either way, so the drone threads the slit and keeps flying "
             "to the goal, which stays --target-offset metres behind the wall.",
    )
    parser.add_argument("--resolution", type=float, default=0.2)
    args = parser.parse_args()
    if args.wall_x is None:
        args.wall_x = args.length / 2.0
    return args


def _span_spec(x0, x1, y0, y1, z0, z1):
    """Box spec (cx, cy, cz, l, w, h) for the span x0..x1 x y0..y1 x z0..z1."""
    return (
        (x0 + x1) / 2.0, (y0 + y1) / 2.0, (z0 + z1) / 2.0,
        x1 - x0, y1 - y0, z1 - z0,
    )


def _slit_span(width, height, thickness, slit_width, slit_height, center_y,
               center_z):
    """Return (y0, y1, z0, z1) of the slit inside the far-wall cross-section."""
    half_width = width / 2.0
    outer_y = half_width + thickness
    y0 = center_y - slit_width / 2.0
    y1 = center_y + slit_width / 2.0
    z0 = center_z - slit_height / 2.0
    z1 = center_z + slit_height / 2.0
    if not (y0 > -outer_y and y1 < outer_y and z0 > 0.0 and z1 < height):
        raise ValueError(
            "the slit must lie strictly inside the far wall cross-section, "
            "otherwise the wall has no frame around it."
        )
    return y0, y1, z0, z1


def _build_specs(length, width, height, thickness, wall_x, slit_y0, slit_y1,
                 slit_z0, slit_z1):
    """Box specs (cx, cy, cz, l, w, h); the open mouth sits at x = 0."""
    half_width = width / 2.0
    outer_y = half_width + thickness
    x0 = wall_x
    x1 = wall_x + thickness
    # Walls span z in [0, height]; the ceiling sits above at z in [height,
    # height+t]. The side walls and the ceiling run the whole tube, so the
    # corridor continues past the slit wall; only the slit wall blocks a
    # cross-section, and only over the frame around the slit.
    walls = [
        # Left wall.
        ((length - thickness) / 2.0, -half_width - thickness / 2.0,
         height / 2.0, length + thickness, thickness, height),
        # Right wall.
        ((length - thickness) / 2.0, half_width + thickness / 2.0,
         height / 2.0, length + thickness, thickness, height),
        # Far wall: two pillars beside the slit, a sill below and a lintel
        # above it.
        _span_spec(x0, x1, -outer_y, slit_y0, 0.0, height),
        _span_spec(x0, x1, slit_y1, outer_y, 0.0, height),
        _span_spec(x0, x1, slit_y0, slit_y1, 0.0, slit_z0),
        _span_spec(x0, x1, slit_y0, slit_y1, slit_z1, height),
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
        "slit width": args.slit_width,
        "slit height": args.slit_height,
        "resolution": args.resolution,
    }
    for name, value in positive.items():
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive.")
    if not np.isfinite(args.target_offset) or args.target_offset < 0.0:
        raise ValueError("target offset must be finite and non-negative.")
    center_z = args.height / 2.0 if args.slit_center_z is None else args.slit_center_z
    for name, value in (("slit center y", args.slit_center_y),
                        ("slit center z", center_z)):
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite.")
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
    world_dir = output_root / "worlds" / "slit"
    pcd_dir = output_root / "pcd"
    positions_path = output_root / "worlds" / "cylinder_positions.txt"
    world_dir.mkdir(parents=True, exist_ok=True)
    pcd_dir.mkdir(parents=True, exist_ok=True)

    slit_y0, slit_y1, slit_z0, slit_z1 = _slit_span(
        args.width, args.height, args.wall_thickness, args.slit_width,
        args.slit_height, args.slit_center_y, center_z,
    )
    if not (args.wall_thickness < args.wall_x <= args.length):
        raise ValueError(
            "wall-x must sit behind the mouth and no further than the far "
            f"end of the tube ({args.length:.3f} m)."
        )
    wall_specs, ceiling_specs = _build_specs(
        args.length, args.width, args.height, args.wall_thickness,
        args.wall_x, slit_y0, slit_y1, slit_z0, slit_z1,
    )

    models = [get_cube_xml(*spec) for spec in wall_specs + ceiling_specs]
    # The static pcd is display-only: wall boxes only (vertical faces plus
    # their top rims), with the ceiling slab and ground-level bottom faces
    # dropped so rviz shows the tube outline and the slit alone.
    point_clouds = [
        get_cube_points(*spec, resolution=args.resolution) for spec in wall_specs
    ]
    points = np.concatenate([item[0] for item in point_clouds], axis=0)
    colors = np.concatenate([item[1] for item in point_clouds], axis=0)
    keep = points[:, 2] > 0.0
    points = points[keep]
    colors = colors[keep]

    world_path = world_dir / "cylinder_slit.world"
    pcd_path = pcd_dir / "cylinder_slit.pcd"
    world_path.write_text(_world_xml(models), encoding="utf-8")
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(colors)
    if not o3d.io.write_point_cloud(str(pcd_path), cloud):
        raise RuntimeError(f"Failed to write {pcd_path}")
    with positions_path.open("w", encoding="utf-8"):
        pass

    target_x = args.length - args.target_offset
    target_y = args.slit_center_y
    print(f"tube: {args.length} m along +x from the mouth at x=0, blocked at "
          f"x={args.wall_x:.3f} by a wall with a "
          f"{args.slit_width} x {args.slit_height} m slit")
    print(f"free section: {args.width} m wide x {args.height} m high, "
          f"wall thickness {args.wall_thickness} m")
    print(f"slit centred at (y={args.slit_center_y:.3f}, z={center_z:.3f}), "
          f"open y in [{slit_y0:.3f}, {slit_y1:.3f}], "
          f"z in [{slit_z0:.3f}, {slit_z1:.3f}]")
    print(f"home: (0.000, 0.000), target: ({target_x:.3f}, {target_y:.3f})")
    print(f"target sits {args.target_offset} m short of the tube's far end "
          f"at x={args.length:.3f}")
    print("cylinder_positions.txt: empty (no obstacles)")
    print(f"static pcd: walls only ({len(points)} points, no ceiling or "
          "ground faces)")
    print(f"World file generated at {world_path.resolve()}")
    print(f"Point cloud file generated at {pcd_path.resolve()}")


if __name__ == "__main__":
    main()
