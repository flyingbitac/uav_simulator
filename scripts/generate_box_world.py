#!/usr/bin/env python3

from textwrap import dedent
import os

import numpy as np

world_dir = os.path.join(os.path.dirname(__file__), "../worlds/test/")
if not os.path.exists(world_dir):
    os.makedirs(world_dir)

def create_cluttered_world(cylinders, cubes, path):
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
    world = "".join([head] + cylinders + cubes + [tail])
    world_file_path = os.path.join(world_dir, path)
    with open(world_file_path, "w") as f:
        f.write(world)
    print("World file generated at", os.path.abspath(world_file_path))

cylinder_idx = 0
def create_cylinder(x, y, r, h):
    global cylinder_idx
    cylinder = dedent(f"""
        <model name='unit_cylinder_{cylinder_idx}'>
          <pose>{x:.5f} {y:.5f} {h/2+0.02:.5f} 0 -0 0</pose>
          <gravity>0 0 -0.1</gravity>
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
        </model>
    """).replace("\n", "    \n    ")
    cylinder_idx += 1
    return cylinder


cube_idx = 0
def create_cube(x, y, z, l, w, h):
    global cube_idx
    z = max(z, 0.5 * h)
    cube = dedent(f"""
        <model name='unit_box_{cube_idx}'>
          <pose>{x:.3f} {y:.3f} {z:.3f} 0 0 0</pose>
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
                  <name>Gazebo/Grey</name>
                  <uri>file://media/materials/scripts/gazebo.material</uri>
                </script>
              </material>
            </visual>
          </link>
        </model>
    """).replace("\n", "    \n    ")
    cube_idx += 1
    return cube

def create_room(L, W, X, Y, D=0.1, H=5, eps=0.03):
    cubes = [
        create_cube(x=X+(D/2+eps),   y=Y-W/2,         z=H/2+0.1, l=D, w=W+2*(D+eps), h=H),
        create_cube(x=X-L-(D/2+eps), y=Y-W/2,         z=H/2+0.1, l=D, w=W+2*(D+eps), h=H),
        create_cube(x=X-L/2,         y=Y+(D/2+eps),   z=H/2+0.1, l=L, w=D,           h=H),
        create_cube(x=X-L/2,         y=Y-W-(D/2+eps), z=H/2+0.1, l=L, w=D,           h=H),
    ]
    return cubes

def big_room():
    L, W, X, Y = 10, 10, 8, 5
    D, H, eps = 1, 10, 0.03
    cylinders = [
        create_cylinder(x=3, y=0, r=0.15, h=H),
        create_cylinder(x=2.5, y=0.5, r=0.15, h=H),
        create_cylinder(x=3.5, y=-0.5, r=0.15, h=H),
        create_cylinder(x=4, y=0, r=0.15, h=H),
        create_cylinder(x=3.5, y=0.5, r=0.15, h=H),
        create_cylinder(x=4.5, y=-0.5, r=0.15, h=H),
    ]
    cubes = create_room(L, W, X, Y, D=D, H=H, eps=eps)
    create_cluttered_world(cylinders, cubes, "big_room.world")

def small_room():
    L, W, X, Y = 5, 4, 4, 2
    D, H, eps = 0.1, 10, 0.03
    cylinders = [
        create_cylinder(x=2, y=0, r=0.15, h=H),
    ]
    cubes = create_room(L, W, X, Y, D=D, H=H, eps=eps)
    create_cluttered_world(cylinders, cubes, "small_room.world")

def mid_room():
    L, W, X, Y = 8, 7, 6, 3
    D, H, eps = 0.1, 10, 0.03
    cylinders = [
        create_cylinder(x=3, y=0, r=0.3, h=H),
    ]
    cubes = create_room(L, W, X, Y, D=D, H=H, eps=eps)
    create_cluttered_world(cylinders, cubes, "mid_room.world")

if __name__ == "__main__":
    big_room()
    mid_room()
    small_room()
