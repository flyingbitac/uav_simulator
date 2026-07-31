#!/usr/bin/env python3
"""Generate deterministic dynamic arena/forest worlds from the static assets."""

import argparse
import json
import math
from pathlib import Path
import random
import xml.etree.ElementTree as ET

import numpy as np


SCENES = {
    "arena": {
        "source_world": "cylinder_arena.world",
        "source_pcd": "cylinder_arena.pcd",
        "output_world": "worlds/arena/arena_dynamic.world",
        "output_pcd": "pcd/arena_dynamic_static.pcd",
        "seed_offset": 0,
        "linear_speed": (0.5, 2.0),
        "circular_angular_speed": (0.5, 1.0),
        "protected_points": (
            (0.0, 0.0),
            (40.0, 0.0),
            (40.0, -40.0),
            (0.0, -40.0),
        ),
    },
    "forest": {
        "source_world": "cylinder_forest.world",
        "source_pcd": "cylinder_forest.pcd",
        "output_world": "worlds/forest/forest_dynamic.world",
        "output_pcd": "pcd/forest_dynamic_static.pcd",
        "seed_offset": 1_000_003,
        "linear_speed": (0.1, 2.0),
        "circular_angular_speed": (0.1, 1.0),
        "protected_points": ((0.0, 0.0), (50.0, 0.0)),
    },
}

LINE_SEGMENT_RANGE = (8.0, 16.0)
CIRCLE_RADIUS_RANGE = (0.5, 1.5)
MOTION_MODE_LINEAR_PROBABILITY = 0.5
PROTECTED_POINT_RADIUS = 4.0
MAX_MOTION_SAMPLE_ATTEMPTS = 1_000
PCD_WORLD_Z_OFFSET = 0.2
PCD_FILTER_EPSILON = 0.03


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Derive deterministic dynamic-cylinder Gazebo worlds and static-only RViz PCDs."
    )
    parser.add_argument(
        "--scene",
        choices=("arena", "forest", "all"),
        default="all",
        help="Scene to generate (default: all).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional deterministic random seed (default: random).",
    )
    parser.add_argument(
        "--dynamic-ratio",
        type=float,
        default=0.5,
        help="Fraction of cylinders converted to dynamic obstacles (default: 0.5).",
    )
    return parser.parse_args()


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


def _configure_dynamic_model(model, marker_id, rng, scene_config):
    pose = _parse_pose(model)
    cylinder = _parse_cylinder(model)
    if cylinder is None:
        raise ValueError(f"Selected model {model.get('name')} is not a cylinder.")
    body_radius, body_height = cylinder

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
    required_clearance = PROTECTED_POINT_RADIUS + body_radius
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
            speed = rng.uniform(*scene_config["linear_speed"])
            phase = rng.choice((0.5, 1.5))
            valid = _motion_respects_protected_points(
                mode,
                scene_config["protected_points"],
                required_clearance,
                line_start=line_start,
                line_end=line_end,
            )
        else:
            mode = "circular"
            trajectory_radius = rng.uniform(*CIRCLE_RADIUS_RANGE)
            angular_speed = rng.uniform(*scene_config["circular_angular_speed"])
            phase = rng.uniform(-math.pi, math.pi)
            circle_center = center.copy()
            circle_center[0] -= trajectory_radius * math.cos(phase)
            circle_center[1] -= trajectory_radius * math.sin(phase)
            valid = _motion_respects_protected_points(
                mode,
                scene_config["protected_points"],
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
        descriptors.append((field, dtype) if count == 1 else (field, dtype, (count,)))
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


def _euler_rotation(euler):
    roll, pitch, yaw = euler
    sr, cr = math.sin(roll), math.cos(roll)
    sp, cp = math.sin(pitch), math.cos(pitch)
    sy, cy = math.sin(yaw), math.cos(yaw)
    rx = np.asarray([[1, 0, 0], [0, cr, sr], [0, -sr, cr]])
    ry = np.asarray([[cp, 0, -sp], [0, 1, 0], [sp, 0, cp]])
    rz = np.asarray([[cy, sy, 0], [-sy, cy, 0], [0, 0, 1]])
    return rx @ ry @ rz


def _filter_dynamic_cylinders(source_pcd, output_pcd, dynamic_models):
    header_lines, records = _read_binary_pcd(source_pcd)
    for coordinate in ("x", "y", "z"):
        if coordinate not in records.dtype.names:
            raise ValueError(f"{source_pcd} is missing coordinate field {coordinate}.")
    points = np.column_stack((records["x"], records["y"], records["z"])).astype(
        np.float64
    )
    keep = np.ones(len(records), dtype=bool)

    for obstacle in dynamic_models:
        pose = obstacle["pose"]
        pcd_center = pose[:3].copy()
        pcd_center[2] -= PCD_WORLD_Z_OFFSET
        rotation = _euler_rotation(pose[3:])
        local_points = (points - pcd_center) @ rotation.T
        radial_distance = np.linalg.norm(local_points[:, :2], axis=1)
        inside = (
            (radial_distance <= obstacle["radius"] + PCD_FILTER_EPSILON)
            & (
                np.abs(local_points[:, 2])
                <= 0.5 * obstacle["height"] + PCD_FILTER_EPSILON
            )
        )
        keep &= ~inside

    filtered_records = records[keep]
    _write_binary_pcd(output_pcd, header_lines, filtered_records)
    return len(records), len(filtered_records)


def _generate_scene(package_dir, scene_name, seed, dynamic_ratio):
    scene_config = SCENES[scene_name]
    source_world = package_dir / "worlds" / scene_name / scene_config["source_world"]
    source_pcd = package_dir / "pcd" / scene_config["source_pcd"]
    output_world = package_dir / scene_config["output_world"]
    output_pcd = package_dir / scene_config["output_pcd"]

    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    source_xml = source_world.read_text(encoding="utf-8").lstrip()
    root = ET.fromstring(source_xml, parser=parser)
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
    rng = random.Random(seed + scene_config["seed_offset"])
    selected_indices = sorted(rng.sample(range(len(cylinders)), dynamic_count))
    dynamic_models = [
        _configure_dynamic_model(cylinders[index], index, rng, scene_config)
        for index in selected_indices
    ]

    output_world.parent.mkdir(parents=True, exist_ok=True)
    _indent_xml(root)
    tree.write(output_world, encoding="utf-8", xml_declaration=True)
    source_points, static_points = _filter_dynamic_cylinders(
        source_pcd, output_pcd, dynamic_models
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


def main():
    args = _parse_args()
    if not 0.0 <= args.dynamic_ratio <= 1.0:
        raise ValueError("--dynamic-ratio must be within [0, 1].")

    seed = (
        args.seed
        if args.seed is not None
        else random.SystemRandom().randrange(0, 2**63)
    )
    package_dir = Path(__file__).resolve().parents[1]
    scene_names = SCENES if args.scene == "all" else (args.scene,)
    results = [
        _generate_scene(package_dir, scene, seed, args.dynamic_ratio)
        for scene in scene_names
    ]
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
