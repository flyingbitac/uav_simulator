#!/usr/bin/env python3

import argparse
from pathlib import Path
from textwrap import dedent
import sys

import numpy as np
import open3d as o3d

sys.path.insert(0, str(Path(__file__).resolve().parent))
from xml_utils import (  # noqa: E402
    _format_vector,
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


def get_sudden_object_xml(
    x, y, z, l, w, h, direction, velocity, trigger_distance,
    travel_distance, activation_model,
):
    model = dedent(f"""
        <model name='sudden_appear_object'>
          <pose>{x:.5f} {y:.5f} {z:.5f} 0 0 0</pose>
          <link name='link'>
            <gravity>0</gravity>
            <inertial>
              <mass>10000</mass>
              <inertia>
                <ixx>2000</ixx>
                <ixy>0</ixy>
                <ixz>0</ixz>
                <iyy>2000</iyy>
                <iyz>0</iyz>
                <izz>2000</izz>
              </inertia>
              <pose>0 0 0 0 0 0</pose>
            </inertial>
            <collision name='collision'>
              <geometry>
                <box>
                  <size>{l:.3f} {w:.3f} {h:.3f}</size>
                </box>
              </geometry>
              <max_contacts>10</max_contacts>
              <surface>
                <contact>
                  <ode/>
                </contact>
                <bounce/>
                <friction>
                  <torsional>
                    <ode/>
                  </torsional>
                  <ode/>
                </friction>
              </surface>
            </collision>
            <visual name='visual'>
              <geometry>
                <box>
                  <size>{l:.3f} {w:.3f} {h:.3f}</size>
                </box>
              </geometry>
              <material>
                <script>
                  <name>Gazebo/Blue</name>
                  <uri>file://media/materials/scripts/gazebo.material</uri>
                </script>
              </material>
            </visual>
            <kinematic>1</kinematic>
          </link>
          <static>0</static>
          <plugin name='sudden_appear_motion' filename='libobstaclePathPlugin.so'>
            <motion_type>sudden_appear</motion_type>
            <activation_model>{activation_model}</activation_model>
            <trigger_distance>{trigger_distance:.9f}</trigger_distance>
            <direction>{_format_vector(direction)}</direction>
            <velocity>{velocity:.9f}</velocity>
            <travel_distance>{travel_distance:.9f}</travel_distance>
            <marker_enabled>true</marker_enabled>
            <marker_id>0</marker_id>
            <marker_shape>box</marker_shape>
            <marker_size>{l:.9f} {w:.9f} {h:.9f}</marker_size>
            <marker_rate>20.0</marker_rate>
            <marker_topic>/uav_simulator/dynamic_obstacles</marker_topic>
            <marker_frame>map</marker_frame>
          </plugin>
        </model>
    """).replace("\n", "    \n    ")
    return model


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate the sudden_appear scene: a corridor whose side wall hides "
            "an object that pops out onto the flight path once the x-axis "
            "distance to the vehicle drops below the triggering distance."
        )
    )
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("-L", "--length", type=float, default=30.0)
    parser.add_argument("-W", "--width", type=float, default=3.0)
    parser.add_argument("-H", "--height", type=float, default=3.0)
    parser.add_argument("--wall-thickness", type=float, default=0.15)
    parser.add_argument("--object-x", type=float, default=15.0)
    parser.add_argument("--object-length", type=float, default=0.5)
    parser.add_argument("--object-width", type=float, default=0.5)
    parser.add_argument("--object-height", type=float, default=3.0)
    parser.add_argument("--object-y", type=float, default=None)
    parser.add_argument("--alcove-width", type=float, default=1.0)
    parser.add_argument("--alcove-depth", type=float, default=1.0)
    parser.add_argument("--trigger-distance", type=float, default=4.0)
    parser.add_argument("--move-speed", type=float, default=5.0)
    parser.add_argument(
        "--travel-distance", type=float, default=None,
        help=(
            "distance the object travels after triggering; default: the "
            "object y, so the default direction stops on the y=0 flight line"
        ),
    )
    parser.add_argument(
        "--direction", type=float, nargs=3, default=(0.0, -1.0, 0.0),
        metavar=("DX", "DY", "DZ"),
        help=(
            "pop-out direction; must exit the alcove toward the corridor "
            "(negative y component), default '0 -1 0'"
        ),
    )
    parser.add_argument("--activation-model", type=str, default="quad0")
    parser.add_argument("--resolution", type=float, default=0.2)
    return parser.parse_args()


def main():
    args = _parse_args()
    positive = {
        "length": args.length,
        "width": args.width,
        "height": args.height,
        "wall thickness": args.wall_thickness,
        "object length": args.object_length,
        "object width": args.object_width,
        "object height": args.object_height,
        "alcove width": args.alcove_width,
        "alcove depth": args.alcove_depth,
        "trigger distance": args.trigger_distance,
        "move speed": args.move_speed,
        "resolution": args.resolution,
    }
    for name, value in positive.items():
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive.")
    if args.object_height > args.height:
        raise ValueError("object height must not exceed the corridor height.")
    if args.alcove_width <= args.object_length:
        raise ValueError("alcove width must exceed the object length.")
    if args.alcove_depth <= args.object_width:
        raise ValueError("alcove depth must exceed the object width.")

    direction = np.asarray(args.direction, dtype=np.float64)
    if not np.all(np.isfinite(direction)):
        raise ValueError("direction must be finite.")
    direction[2] = 0.0
    direction_length = float(np.linalg.norm(direction))
    if direction_length <= 0.0:
        raise ValueError("direction must be non-zero.")
    direction /= direction_length
    if direction[1] >= 0.0:
        raise ValueError(
            "direction must exit the alcove toward the corridor "
            "(negative y component)."
        )

    half_gap = 0.5 * args.alcove_width
    if args.object_x - half_gap - args.wall_thickness <= 0.0 or (
        args.object_x + half_gap + args.wall_thickness >= args.length
    ):
        raise ValueError("alcove must lie strictly inside the +y wall.")

    wall_y = 0.5 * args.width
    alcove_back_y = wall_y + args.alcove_depth
    object_y = (
        wall_y + 0.5 * args.alcove_depth
        if args.object_y is None
        else args.object_y
    )
    if not np.isfinite(object_y):
        raise ValueError("object y must be finite.")
    object_min_y = wall_y + 0.5 * (args.wall_thickness + args.object_width)
    object_max_y = alcove_back_y - 0.5 * args.object_width
    if not object_min_y <= object_y <= object_max_y:
        raise ValueError(
            f"object y must lie within [{object_min_y:.3f}, {object_max_y:.3f}] "
            "so the object fits inside the alcove pocket."
        )
    travel_distance = (
        object_y if args.travel_distance is None else args.travel_distance
    )
    if not np.isfinite(travel_distance) or travel_distance <= 0.0:
        raise ValueError("travel distance must be finite and positive.")
    if args.activation_model == "":
        raise ValueError("activation model must be non-empty.")

    package_root = Path(__file__).resolve().parents[1]
    if args.output_root:
        output_root = resolve_private_output_root(args.output_root, package_root)
    else:
        output_root = package_root
    world_dir = output_root / "worlds" / "sudden_appear"
    pcd_dir = output_root / "pcd"
    world_dir.mkdir(parents=True, exist_ok=True)
    pcd_dir.mkdir(parents=True, exist_ok=True)

    gap_start = args.object_x - half_gap
    gap_end = args.object_x + half_gap
    half_t = 0.5 * args.wall_thickness
    # Wall footprints as (x_min, x_max, y_min, y_max): corridor walls, the +y
    # wall split around the alcove mouth, and the alcove pocket walls.
    wall_rects = [
        (0.0, args.length, -wall_y - half_t, -wall_y + half_t),
        (0.0, gap_start, wall_y - half_t, wall_y + half_t),
        (gap_end, args.length, wall_y - half_t, wall_y + half_t),
        (gap_start - args.wall_thickness, gap_start, wall_y, alcove_back_y),
        (gap_end, gap_end + args.wall_thickness, wall_y, alcove_back_y),
        (gap_start - args.wall_thickness, gap_end + args.wall_thickness,
         alcove_back_y, alcove_back_y + args.wall_thickness),
    ]

    half_length = 0.5 * args.object_length
    half_width = 0.5 * args.object_width
    sweep_step = 0.5 * min(half_length, half_width)
    sweep = np.linspace(
        0.0, travel_distance,
        max(2, int(np.ceil(travel_distance / sweep_step)) + 1),
    )
    for traveled in sweep:
        sample_x = args.object_x + traveled * direction[0]
        sample_y = object_y + traveled * direction[1]
        for x_min, x_max, y_min, y_max in wall_rects:
            if (
                sample_x + half_length > x_min
                and sample_x - half_length < x_max
                and sample_y + half_width > y_min
                and sample_y - half_width < y_max
            ):
                raise ValueError(
                    "pop-out path must clear the walls; "
                    "adjust direction or travel distance."
                )

    boundary_specs = [
        (0.5 * (x_min + x_max), 0.5 * (y_min + y_max), args.height / 2.0,
         x_max - x_min, y_max - y_min, args.height)
        for x_min, x_max, y_min, y_max in wall_rects
    ]
    # Ceiling over both the corridor and the alcove so the object cannot be
    # flown over. It stays in the gazebo world but is excluded from the static
    # pcd below, which only visualises the side walls.
    ceiling_y_min = -wall_y - half_t
    ceiling_y_max = alcove_back_y + args.wall_thickness
    ceiling_spec = (
        args.length / 2.0, 0.5 * (ceiling_y_min + ceiling_y_max),
        args.height + half_t, args.length,
        ceiling_y_max - ceiling_y_min, args.wall_thickness)
    wall_specs = boundary_specs
    boundary_specs = wall_specs + [ceiling_spec]

    models = [get_cube_xml(*spec) for spec in boundary_specs]
    models.append(get_sudden_object_xml(
        args.object_x,
        object_y,
        args.object_height / 2.0,
        args.object_length,
        args.object_width,
        args.object_height,
        direction,
        args.move_speed,
        args.trigger_distance,
        travel_distance,
        args.activation_model,
    ))
    # The static pcd is display-only: side walls (vertical faces plus their
    # top rims), with the ceiling slab and the ground-level bottom faces
    # dropped so rviz shows the walls alone.
    point_clouds = [
        get_cube_points(*spec, resolution=args.resolution)
        for spec in wall_specs
    ]
    points = np.concatenate([item[0] for item in point_clouds], axis=0)
    colors = np.concatenate([item[1] for item in point_clouds], axis=0)
    keep = points[:, 2] > 0.0
    points = points[keep]
    colors = colors[keep]

    world_path = world_dir / "sudden_appear.world"
    pcd_path = pcd_dir / "sudden_appear_static.pcd"
    positions_path = output_root / "worlds" / "cylinder_positions.txt"
    world_path.write_text(_world_xml(models), encoding="utf-8")
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(colors)
    if not o3d.io.write_point_cloud(str(pcd_path), cloud):
        raise RuntimeError(f"Failed to write {pcd_path}")
    radius = 0.5 * np.hypot(args.object_length, args.object_width)
    with positions_path.open("w", encoding="utf-8") as positions:
        positions.write(f"{args.object_x:.5f},{object_y:.5f},{radius:.5f}\n")

    print(f"object: {args.object_length} x {args.object_width} x "
          f"{args.object_height} at ({args.object_x:.3f}, {object_y:.3f})")
    print(f"trigger_distance: {args.trigger_distance} "
          f"(x-axis, activation model '{args.activation_model}')")
    print(f"motion: direction ({_format_vector(direction)}), "
          f"speed {args.move_speed} m/s, travel {travel_distance:.3f} m")
    print(f"static pcd: side walls only ({len(points)} points, "
          "no ceiling or ground faces)")
    print(f"World file generated at {world_path.resolve()}")
    print(f"Point cloud file generated at {pcd_path.resolve()}")


if __name__ == "__main__":
    main()
