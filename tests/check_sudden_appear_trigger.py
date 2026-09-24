#!/usr/bin/env python3
"""Verify the sudden_appear obstacle triggers correctly inside a live Gazebo.

Usage: start a headless gazebo with the sudden_appear world (no drone spawned),
then run this script against it, e.g.

    roslaunch gazebo_ros empty_world.launch world_name:=<world> gui:=false &
    python3 check_sudden_appear_trigger.py --world <world>

The script spawns a minimal box named after the plugin's activation model,
walks it toward the obstacle along x, and asserts the obstacle stays hidden,
pops out at the triggering distance, moves at the configured speed, and stops
after the configured travel distance.
"""

import argparse
import math
import sys
import time
import xml.etree.ElementTree as ET

import rospy
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.srv import SpawnModel, SetModelState
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Pose

ACTIVATION_SDF = """
<sdf version="1.5">
  <model name="{name}">
    <static>true</static>
    <link name="link">
      <collision name="collision">
        <geometry><box><size>0.2 0.2 0.2</size></box></geometry>
      </collision>
    </link>
  </model>
</sdf>
"""

# Probe half-width of the triggering band: Phase 1 parks just outside
# trigger_distance and Phase 2 just inside it, so the checks pin the effective
# threshold to within this margin.
PROBE_MARGIN = 0.1


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world", required=True, help="sudden_appear.world path")
    parser.add_argument(
        "--object-model", default="sudden_appear_object",
        help="Gazebo model name of the dynamic obstacle",
    )
    parser.add_argument(
        "--position-tolerance", type=float, default=0.02,
        help="Allowed pose jitter (m) while the obstacle must stay still",
    )
    parser.add_argument(
        "--speed-tolerance", type=float, default=0.35,
        help="Relative speed tolerance while the obstacle is moving",
    )
    return parser.parse_args()


def read_plugin_config(world_path, object_model):
    world = ET.parse(world_path).getroot().find("world")
    for model in world.findall("model"):
        if model.get("name") != object_model:
            continue
        plugin = model.find("plugin[@filename='libobstaclePathPlugin.so']")
        if plugin is None:
            raise RuntimeError(f"{object_model} has no obstaclePathPlugin")
        pose = [float(v) for v in model.findtext("pose").split()]
        config = {
            "activation_model": plugin.findtext("activation_model"),
            "trigger_distance": float(plugin.findtext("trigger_distance")),
            "travel_distance": float(plugin.findtext("travel_distance")),
            "velocity": float(plugin.findtext("velocity")),
            "direction": [
                float(v) for v in plugin.findtext("direction").split()
            ],
            "start": pose[:3],
        }
        return config
    raise RuntimeError(f"model {object_model} not found in {world_path}")


class GazeboProbe:
    def __init__(self):
        self.states = None
        self.state_sub = rospy.Subscriber(
            "/gazebo/model_states", ModelStates, self._on_states, queue_size=1
        )
        rospy.wait_for_service("/gazebo/spawn_sdf_model", timeout=30.0)
        rospy.wait_for_service("/gazebo/set_model_state", timeout=30.0)
        self.spawn_srv = rospy.ServiceProxy("/gazebo/spawn_sdf_model", SpawnModel)
        self.set_state_srv = rospy.ServiceProxy(
            "/gazebo/set_model_state", SetModelState
        )

    def _on_states(self, msg):
        self.states = msg

    def wait_for_states(self, timeout=30.0):
        deadline = time.time() + timeout
        while self.states is None and time.time() < deadline and not rospy.is_shutdown():
            time.sleep(0.05)
        if self.states is None:
            raise RuntimeError("no /gazebo/model_states received")

    def spawn_box(self, name, x, y, z):
        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = z
        pose.orientation.w = 1.0
        response = self.spawn_srv(
            name, ACTIVATION_SDF.format(name=name), "", pose, "world"
        )
        if not response.success:
            raise RuntimeError(f"spawn failed: {response.status_message}")

    def set_pose(self, name, x, y, z):
        state = ModelState()
        state.model_name = name
        state.pose.position.x = x
        state.pose.position.y = y
        state.pose.position.z = z
        state.pose.orientation.w = 1.0
        state.reference_frame = "world"
        response = self.set_state_srv(state)
        if not response.success:
            raise RuntimeError(f"set_model_state failed: {response.status_message}")

    def get_position(self, name):
        self.wait_for_states()
        if name not in self.states.name:
            raise RuntimeError(f"model {name} missing from /gazebo/model_states")
        index = self.states.name.index(name)
        pose = self.states.pose[index]
        return (pose.position.x, pose.position.y, pose.position.z)


def wait_until(predicate, timeout, poll=0.02):
    deadline = time.time() + timeout
    while time.time() < deadline and not rospy.is_shutdown():
        if predicate():
            return True
        time.sleep(poll)
    return predicate()


def main():
    args = parse_args()
    config = read_plugin_config(args.world, args.object_model)
    rospy.init_node("check_sudden_appear_trigger", anonymous=True)
    probe = GazeboProbe()
    probe.wait_for_states()

    start = config["start"]
    direction = config["direction"]
    trigger = config["trigger_distance"]
    travel = config["travel_distance"]
    speed = config["velocity"]
    activation_model = config["activation_model"]

    rospy.loginfo(
        "config: trigger=%.2f m (x-axis), speed=%.2f m/s, travel=%.2f m, "
        "direction=(%.2f, %.2f, %.2f)",
        trigger, speed, travel, *direction,
    )

    if activation_model in probe.states.name:
        raise RuntimeError(
            f"model '{activation_model}' already exists in the world; this "
            "check expects a bare world without the drone"
        )
    probe.spawn_box(activation_model, start[0] - trigger - 3.0, 0.0, 1.0)
    initial = probe.get_position(args.object_model)
    rospy.loginfo("object initial position: (%.3f, %.3f, %.3f)", *initial)

    failures = []

    def check(condition, message):
        if condition:
            rospy.loginfo("PASS: %s", message)
        else:
            rospy.logerr("FAIL: %s", message)
            failures.append(message)

    # Phase 1: just outside the triggering band -> the object must stay
    # hidden. Phase 2 probes just inside the band, so together they pin the
    # effective threshold to trigger_distance within PROBE_MARGIN.
    margin = PROBE_MARGIN
    probe.set_pose(activation_model, start[0] - trigger - margin, 0.0, 1.0)
    time.sleep(1.5)
    before_trigger = probe.get_position(args.object_model)
    still = all(
        abs(a - b) <= args.position_tolerance for a, b in zip(initial, before_trigger)
    )
    check(still, "object stays hidden while |x_uav - x_obj| > trigger_distance")

    # Phase 2: enter the triggering band -> the object must pop out.
    probe.set_pose(activation_model, start[0] - trigger + margin, 0.0, 1.0)
    moved = wait_until(
        lambda: math.dist(probe.get_position(args.object_model), before_trigger)
        > 0.1,
        timeout=10.0,
    )
    check(moved, "object starts moving within the triggering band")

    # Phase 3: measure the popping speed along the configured direction
    # against simulation time so the real-time factor does not matter.
    samples = []
    sample_deadline = time.time() + 0.5
    while time.time() < sample_deadline and not rospy.is_shutdown():
        samples.append((rospy.get_time(), probe.get_position(args.object_model)))
        time.sleep(0.05)
    if len(samples) >= 2:
        (t0, p0), (t1, p1) = samples[0], samples[-1]
        dt = t1 - t0
        displacement = math.dist(p1, p0) / dt if dt > 0 else 0.0
        speed_ok = abs(displacement - speed) <= args.speed_tolerance * speed
        check(
            speed_ok,
            f"pop-out speed {displacement:.2f} m/s within tolerance of {speed:.2f} m/s",
        )
    else:
        check(False, "could not sample pop-out motion")

    # Phase 4: the object must stop at the travel end and hold still. The
    # plugin clamps the final step exactly onto the stop pose, so a tight
    # arrival tolerance only passes once the motion is complete.
    stop_pos = (
        start[0] + travel * direction[0],
        start[1] + travel * direction[1],
        start[2] + travel * direction[2],
    )
    arrived = 1e-3
    reached = wait_until(
        lambda: math.dist(probe.get_position(args.object_model), stop_pos) <= arrived,
        timeout=2.0 * travel / max(speed, 1e-3) + 10.0,
    )
    final_pos = probe.get_position(args.object_model)
    check(
        reached,
        f"object reaches the stop pose {tuple(round(v, 2) for v in stop_pos)} "
        f"(got {tuple(round(v, 2) for v in final_pos)})",
    )

    time.sleep(2.0)
    settled_pos = probe.get_position(args.object_model)
    drift = math.dist(final_pos, settled_pos)
    off_target = math.dist(settled_pos, stop_pos)
    check(
        drift <= args.position_tolerance and off_target <= arrived,
        f"object stays stopped at the travel end (drift {drift:.4f} m, "
        f"offset {off_target:.4f} m <= {arrived} m)",
    )

    if failures:
        rospy.logerr("RESULT: FAIL (%d checks failed)", len(failures))
        return 1
    rospy.loginfo("RESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
