#!/usr/bin/env python3

from textwrap import dedent
import math
from pathlib import Path
import random
import xml.etree.ElementTree as ET

import numpy as np
import open3d as o3d


LINE_SEGMENT_RANGE = (8.0, 16.0)
CIRCLE_RADIUS_RANGE = (0.5, 1.5)
MOTION_MODE_LINEAR_PROBABILITY = 0.5
PROTECTED_POINT_RADIUS = 4.0
MAX_MOTION_SAMPLE_ATTEMPTS = 1_000

cylinder_idx = 0
def get_cylinder_xml(x, y, z, r, h, euler):
    global cylinder_idx
    cylinder = dedent(f"""
        <model name='unit_cylinder_{cylinder_idx}'>
          <pose>{x:.5f} {y:.5f} {z+h/2+0.2:.5f} {euler[0]:.5f} {euler[1]:.5f} {euler[2]:.5f}</pose>
          <gravity>0 0 0</gravity>
          <link name='link'>
            <gravity>1</gravity>
            <inertial>
              <mass>100</mass>
              <inertia>
                <ixx>20</ixx>
                <ixy>0</ixy>
                <ixz>0</ixz>
                <iyy>20</iyy>
                <iyz>0</iyz>
                <izz>20</izz>
              </inertia>
              <pose>0 0 0 0 -0 0</pose>
            </inertial>
            <collision name='collision'>
              <geometry>
                <cylinder>
                  <radius>{r:.3f}</radius>
                  <length>{h:.3f}</length>
                </cylinder>
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
                <cylinder>
                  <radius>{r:.3f}</radius>
                  <length>{h:.3f}</length>
                </cylinder>
              </geometry>
              <material>
                <script>
                  <name>Gazebo/Bricks</name>
                  <uri>file://media/materials/scripts/gazebo.material</uri>
                </script>
              </material>
            </visual>
            <self_collide>0</self_collide>
            <enable_wind>0</enable_wind>
            <kinematic>0</kinematic>
          </link>
          <static>1</static>
        </model>
    """).replace("\n", "    \n    ")
    cylinder_idx += 1
    return cylinder, (x, y, r)

def euler2rotmat(euler: np.ndarray) -> np.ndarray:
    """Convert euler angles to rotation matrix.
    Args:
        euler: (3,) array of euler angles (roll, pitch, yaw)
    Returns:
        rotmat: (3, 3) rotation matrix
    """
    roll, pitch, yaw = euler
    sr, cr, sp, cp, sy, cy = np.sin(roll), np.cos(roll), np.sin(pitch), np.cos(pitch), np.sin(yaw), np.cos(yaw)
    Rx = np.array([[1,   0,  0],
                   [0,  cr, sr],
                   [0, -sr, cr]])
    Ry = np.array([[cp, 0, -sp],
                   [ 0, 1,   0],
                   [sp, 0,  cp]])
    Rz = np.array([[ cy, sy, 0],
                   [-sy, cy, 0],
                   [  0,  0, 1]])
    rotmat = Rx @ Ry @ Rz
    return rotmat

def get_cylinder_points(x, y, z, r, h, euler, resolution=0.05):
    """Generate point cloud of a cylinder.
    Args:
        x, y, z: center of the cylinder base
        r: radius
        h: height
        euler: (3,) euler angles (roll, pitch, yaw)
    Returns:
        points: (N, 3) point cloud of the cylinder
    """
    r_res = int(2 * r * np.pi / resolution)
    h_res = int(h / resolution)
    cylinder = o3d.geometry.TriangleMesh.create_cylinder(
      radius=r, height=h, resolution=r_res, split=h_res, create_uv_map=False).vertices
    points = np.concatenate([cylinder], axis=0)
    R = euler2rotmat(euler)
    max_color = np.array([16, 36, 116]) / 255.0
    min_color = np.array([196, 229, 235]) / 255.0
    colors = (points[:, 2:] / h) * (max_color - min_color) + min_color
    points = points @ R
    points += np.array([x, y, z + h / 2])
    return points, colors

cube_idx = 0
def get_cube_xml(x, y, z, l, w, h):
    global cube_idx
    z = max(z, 0.5 * h)
    cube = dedent(f"""
        <model name='unit_box_{cube_idx}'>
          <pose>{x:.3f} {y:.3f} {z:.3f} 0 0 0</pose>
          <link name='link'>
            <gravity>1</gravity>
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
                  <name>Gazebo/Grey</name>
                  <uri>file://media/materials/scripts/gazebo.material</uri>
                </script>
              </material>
            </visual>
          </link>
          <static>1</static>
        </model>
    """).replace("\n", "    \n    ")
    cube_idx += 1
    return cube

def get_cube_points(x, y, z, l, w, h, resolution=0.05):
    """Generate point cloud of a cube.
    Args:
        x, y, z: center of the cube
        l, w, h: length, width, height
    Returns:
        points: (N, 3) point cloud of the cube
    """
    nx = max(2, int(np.ceil(l / resolution)) + 1)
    ny = max(2, int(np.ceil(w / resolution)) + 1)
    nz = max(2, int(np.ceil(h / resolution)) + 1)

    xs = np.linspace(-l / 2, l / 2, nx)
    ys = np.linspace(-w / 2, w / 2, ny)
    zs = np.linspace(-h / 2, h / 2, nz)

    yy, zz = np.meshgrid(ys, zs, indexing="xy")
    yy = yy.ravel()
    zz = zz.ravel()
    xx_neg = np.full_like(yy, -l / 2)
    xx_pos = np.full_like(yy, l / 2)

    xx, zz2 = np.meshgrid(xs, zs, indexing="xy")
    xx = xx.ravel()
    zz2 = zz2.ravel()
    yy_neg = np.full_like(xx, -w / 2)
    yy_pos = np.full_like(xx, w / 2)

    xx2, yy2 = np.meshgrid(xs, ys, indexing="xy")
    xx2 = xx2.ravel()
    yy2 = yy2.ravel()
    zz_neg = np.full_like(xx2, -h / 2)
    zz_pos = np.full_like(xx2, h / 2)

    points = np.concatenate(
        [
            np.stack((xx_neg, yy, zz), axis=1),
            np.stack((xx_pos, yy, zz), axis=1),
            np.stack((xx, yy_neg, zz2), axis=1),
            np.stack((xx, yy_pos, zz2), axis=1),
            np.stack((xx2, yy2, zz_neg), axis=1),
            np.stack((xx2, yy2, zz_pos), axis=1),
        ],
        axis=0,
    )
    max_color = np.array([16, 36, 116]) / 255.0
    min_color = np.array([196, 229, 235]) / 255.0
    colors = ((points[:, 2:3] + h / 2) / h) * (max_color - min_color) + min_color
    points += np.array([x, y, z])
    return points, colors


def resolve_seed(seed):
    if seed is not None:
        return seed
    return random.SystemRandom().randrange(0, 2**63)


def _parse_pose(model):
    pose = model.find("pose")
    if pose is None or not pose.text:
        raise ValueError(f"Model {model.get('name')} has no pose.")
    values = [float(value) for value in pose.text.split()]
    if len(values) != 6:
        raise ValueError(f"Model {model.get('name')} pose must contain six values.")
    return np.asarray(values, dtype=np.float64)


def _parse_cylinder(model):
    radius_element = model.find("./link/collision/geometry/cylinder/radius")
    height_element = model.find("./link/collision/geometry/cylinder/length")
    if radius_element is None or height_element is None:
        return None
    return float(radius_element.text), float(height_element.text)


def _format_vector(vector):
    return " ".join(f"{float(value):.9f}" for value in vector)


def _add_plugin_value(plugin, name, value):
    element = ET.SubElement(plugin, name)
    element.text = str(value)


def _indent_xml(element, level=0):
    indentation = "\n" + level * "  "
    child_indentation = "\n" + (level + 1) * "  "
    if len(element):
        if not element.text or not element.text.strip():
            element.text = child_indentation
        for child in element:
            _indent_xml(child, level + 1)
            if not child.tail or not child.tail.strip():
                child.tail = child_indentation
        element[-1].tail = indentation
    elif level and (not element.tail or not element.tail.strip()):
        element.tail = indentation


def _point_to_segment_distance_2d(point, start, end):
    segment = end[:2] - start[:2]
    length_squared = float(np.dot(segment, segment))
    if length_squared == 0.0:
        return float(np.linalg.norm(point - start[:2]))
    progress = float(np.dot(point - start[:2], segment) / length_squared)
    closest = start[:2] + np.clip(progress, 0.0, 1.0) * segment
    return float(np.linalg.norm(point - closest))


def _motion_respects_protected_points(
    mode,
    protected_points,
    required_clearance,
    line_start=None,
    line_end=None,
    circle_center=None,
    circle_radius=None,
):
    for point in protected_points:
        point = np.asarray(point, dtype=np.float64)
        if mode == "linear":
            distance = _point_to_segment_distance_2d(point, line_start, line_end)
        else:
            center_distance = float(np.linalg.norm(point - circle_center[:2]))
            distance = abs(center_distance - circle_radius)
        if distance < required_clearance:
            return False
    return True


def _dynamic_cylinder_clearance(model):
    pose = _parse_pose(model)
    cylinder = _parse_cylinder(model)
    if cylinder is None:
        raise ValueError(f"Selected model {model.get('name')} is not a cylinder.")
    body_radius, body_height = cylinder
    rotation = euler2rotmat(pose[3:])
    cylinder_axis = rotation[2]
    horizontal_half_height = 0.5 * body_height * np.linalg.norm(cylinder_axis[:2])
    return pose, body_radius, body_height, (
        PROTECTED_POINT_RADIUS + body_radius + horizontal_half_height
    )


def _configure_dynamic_model(model, marker_id, rng, config):
    pose, body_radius, body_height, required_clearance = (
        _dynamic_cylinder_clearance(model)
    )

    static_element = model.find("static")
    if static_element is None:
        static_element = ET.SubElement(model, "static")
    static_element.text = "0"

    link = model.find("link")
    if link is None:
        raise ValueError(f"Selected model {model.get('name')} has no link.")
    kinematic_element = link.find("kinematic")
    if kinematic_element is None:
        kinematic_element = ET.SubElement(link, "kinematic")
    kinematic_element.text = "1"

    center = pose[:3]
    for _ in range(MAX_MOTION_SAMPLE_ATTEMPTS):
        if rng.random() < MOTION_MODE_LINEAR_PROBABILITY:
            mode = "linear"
            direction_angle = rng.uniform(-math.pi, math.pi)
            direction = np.asarray(
                [math.cos(direction_angle), math.sin(direction_angle), 0.0],
                dtype=np.float64,
            )
            segment_length = rng.uniform(*LINE_SEGMENT_RANGE)
            line_start = center - 0.5 * segment_length * direction
            line_end = center + 0.5 * segment_length * direction
            speed = rng.uniform(*config["linear_speed"])
            phase = rng.choice((0.5, 1.5))
            valid = _motion_respects_protected_points(
                mode,
                config["protected_points"],
                required_clearance,
                line_start=line_start,
                line_end=line_end,
            )
        else:
            mode = "circular"
            trajectory_radius = rng.uniform(*CIRCLE_RADIUS_RANGE)
            angular_speed = rng.uniform(*config["circular_angular_speed"])
            phase = rng.uniform(-math.pi, math.pi)
            circle_center = center.copy()
            circle_center[0] -= trajectory_radius * math.cos(phase)
            circle_center[1] -= trajectory_radius * math.sin(phase)
            valid = _motion_respects_protected_points(
                mode,
                config["protected_points"],
                required_clearance,
                circle_center=circle_center,
                circle_radius=trajectory_radius,
            )
        if valid:
            break
    else:
        raise RuntimeError(
            f"Could not sample a safe motion for {model.get('name')} after "
            f"{MAX_MOTION_SAMPLE_ATTEMPTS} attempts."
        )

    plugin = ET.SubElement(
        model,
        "plugin",
        {
            "name": f"dynamic_obstacle_motion_{marker_id}",
            "filename": "libobstaclePathPlugin.so",
        },
    )
    _add_plugin_value(plugin, "motion_type", mode)
    if mode == "linear":
        _add_plugin_value(plugin, "line_start", _format_vector(line_start))
        _add_plugin_value(plugin, "line_end", _format_vector(line_end))
        _add_plugin_value(plugin, "velocity", f"{speed:.9f}")
        _add_plugin_value(plugin, "phase", f"{phase:.1f}")
    else:
        _add_plugin_value(plugin, "circle_center", _format_vector(circle_center))
        _add_plugin_value(plugin, "circle_radius", f"{trajectory_radius:.9f}")
        _add_plugin_value(plugin, "angular_velocity", f"{angular_speed:.9f}")
        _add_plugin_value(plugin, "phase", f"{phase:.9f}")

    _add_plugin_value(plugin, "orientation", "false")
    _add_plugin_value(plugin, "marker_enabled", "true")
    _add_plugin_value(plugin, "marker_id", marker_id)
    _add_plugin_value(plugin, "marker_radius", f"{body_radius:.9f}")
    _add_plugin_value(plugin, "marker_height", f"{body_height:.9f}")
    _add_plugin_value(plugin, "marker_rate", "20.0")
    _add_plugin_value(plugin, "marker_topic", "/uav_simulator/dynamic_obstacles")
    _add_plugin_value(plugin, "marker_frame", "map")

    return {
        "id": marker_id,
        "name": model.get("name"),
        "mode": mode,
        "pose": pose,
        "radius": body_radius,
        "height": body_height,
    }


def _pcd_dtype(header):
    fields = header["FIELDS"]
    sizes = [int(value) for value in header["SIZE"]]
    types = header["TYPE"]
    counts = [int(value) for value in header.get("COUNT", ["1"] * len(fields))]
    if not (len(fields) == len(sizes) == len(types) == len(counts)):
        raise ValueError("Malformed PCD field metadata.")

    type_map = {
        ("F", 4): "<f4",
        ("F", 8): "<f8",
        ("I", 1): "<i1",
        ("I", 2): "<i2",
        ("I", 4): "<i4",
        ("I", 8): "<i8",
        ("U", 1): "<u1",
        ("U", 2): "<u2",
        ("U", 4): "<u4",
        ("U", 8): "<u8",
    }
    descriptors = []
    for field, size, field_type, count in zip(fields, sizes, types, counts):
        dtype = type_map.get((field_type, size))
        if dtype is None:
            raise ValueError(f"Unsupported PCD field type: {field_type}{size}.")
        descriptors.append(
            (field, dtype) if count == 1 else (field, dtype, (count,))
        )
    return np.dtype(descriptors)


def _read_binary_pcd(path):
    header_lines = []
    header = {}
    with path.open("rb") as source:
        while True:
            line = source.readline()
            if not line:
                raise ValueError(f"{path} has no DATA header.")
            header_lines.append(line)
            decoded = line.decode("ascii").strip()
            if decoded and not decoded.startswith("#"):
                key, *values = decoded.split()
                header[key.upper()] = values
            if decoded.upper().startswith("DATA "):
                break
        payload = source.read()

    if header.get("DATA", [""])[0].lower() != "binary":
        raise ValueError(f"{path} must use binary PCD data.")
    dtype = _pcd_dtype(header)
    records = np.frombuffer(payload, dtype=dtype).copy()
    expected_points = int(header["POINTS"][0])
    if len(records) != expected_points:
        raise ValueError(
            f"{path} contains {len(records)} records but declares {expected_points}."
        )
    return header_lines, records


def _write_binary_pcd(path, header_lines, records):
    point_count = len(records)
    rewritten_header = []
    for line in header_lines:
        decoded = line.decode("ascii")
        key = decoded.strip().split(maxsplit=1)[0].upper() if decoded.strip() else ""
        if key == "WIDTH":
            rewritten_header.append(f"WIDTH {point_count}\n".encode("ascii"))
        elif key == "HEIGHT":
            rewritten_header.append(b"HEIGHT 1\n")
        elif key == "POINTS":
            rewritten_header.append(f"POINTS {point_count}\n".encode("ascii"))
        else:
            rewritten_header.append(line)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as destination:
        destination.writelines(rewritten_header)
        destination.write(records.tobytes(order="C"))


def _select_static_cylinder_points(
    source_pcd, output_pcd, cylinder_point_counts, dynamic_indices
):
    header_lines, records = _read_binary_pcd(source_pcd)
    point_counts = [int(count) for count in cylinder_point_counts]
    if any(count < 0 for count in point_counts):
        raise ValueError("Cylinder point counts must be non-negative.")
    if sum(point_counts) != len(records):
        raise ValueError(
            f"Cylinder point counts sum to {sum(point_counts)}, but {source_pcd} "
            f"contains {len(records)} records."
        )

    dynamic_indices = set(dynamic_indices)
    static_chunks = []
    offset = 0
    for index, point_count in enumerate(point_counts):
        next_offset = offset + point_count
        if index not in dynamic_indices:
            static_chunks.append(records[offset:next_offset])
        offset = next_offset
    filtered_records = (
        np.concatenate(static_chunks) if static_chunks else records[:0]
    )
    _write_binary_pcd(output_pcd, header_lines, filtered_records)
    return len(records), len(filtered_records)


def generate_dynamic_cylinder_world(
    *,
    scene_name,
    source_world,
    source_pcd,
    output_world,
    output_pcd,
    seed,
    dynamic_ratio,
    linear_speed,
    circular_angular_speed,
    protected_points,
    cylinder_point_counts,
    seed_offset=0,
):
    if not 0.0 <= dynamic_ratio <= 1.0:
        raise ValueError("dynamic_ratio must be within [0, 1].")

    source_world = Path(source_world).resolve()
    source_pcd = Path(source_pcd).resolve()
    output_world = Path(output_world).resolve()
    output_pcd = Path(output_pcd).resolve()
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    root = ET.fromstring(
        source_world.read_text(encoding="utf-8").lstrip(), parser=parser
    )
    tree = ET.ElementTree(root)
    world = root.find("world")
    if world is None:
        raise ValueError(f"{source_world} has no world element.")
    cylinders = [
        model
        for model in world.findall("model")
        if model.get("name", "").startswith("unit_cylinder_")
        and _parse_cylinder(model) is not None
    ]
    dynamic_count = int(len(cylinders) * dynamic_ratio)
    rng = random.Random(seed + seed_offset)
    eligible_indices = []
    for index, model in enumerate(cylinders):
        pose, _, _, required_clearance = _dynamic_cylinder_clearance(model)
        if all(
            np.linalg.norm(pose[:2] - np.asarray(point, dtype=np.float64))
            >= required_clearance
            for point in protected_points
        ):
            eligible_indices.append(index)
    if dynamic_count > len(eligible_indices):
        raise ValueError(
            f"Requested {dynamic_count} dynamic cylinders, but only "
            f"{len(eligible_indices)} avoid the protected points at their initial pose."
        )
    selected_indices = sorted(rng.sample(eligible_indices, dynamic_count))
    config = {
        "linear_speed": linear_speed,
        "circular_angular_speed": circular_angular_speed,
        "protected_points": protected_points,
    }
    dynamic_models = [
        _configure_dynamic_model(cylinders[index], index, rng, config)
        for index in selected_indices
    ]

    output_world.parent.mkdir(parents=True, exist_ok=True)
    _indent_xml(root)
    tree.write(output_world, encoding="utf-8", xml_declaration=True)
    if len(cylinder_point_counts) != len(cylinders):
        raise ValueError(
            "cylinder_point_counts must contain one entry per cylinder model."
        )
    source_points, static_points = _select_static_cylinder_points(
        source_pcd, output_pcd, cylinder_point_counts, selected_indices
    )
    mode_counts = {
        mode: sum(model["mode"] == mode for model in dynamic_models)
        for mode in ("linear", "circular")
    }
    return {
        "scene": scene_name,
        "seed": seed,
        "dynamic_ratio": dynamic_ratio,
        "total_cylinders": len(cylinders),
        "dynamic_cylinders": len(dynamic_models),
        "mode_counts": mode_counts,
        "source_pcd_points": source_points,
        "static_pcd_points": static_points,
        "world": str(output_world),
        "pcd": str(output_pcd),
    }
