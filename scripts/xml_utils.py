from textwrap import dedent
import os
import argparse

#!/usr/bin/env python3

import numpy as np
import open3d as o3d

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
