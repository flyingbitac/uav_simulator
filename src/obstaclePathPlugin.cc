#include <algorithm>
#include <cmath>
#include <functional>
#include <map>
#include <memory>
#include <mutex>

#include <uav_simulator/obstaclePathPlugin.hh>

#include <ros/ros.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

namespace gazebo
{
namespace
{
constexpr double kEpsilon = 1e-6;

struct SharedMarkerPublisher
{
  std::mutex mutex;
  std::unique_ptr<ros::NodeHandle> node;
  ros::Publisher publisher;
  std::string topic;
  std::map<int, visualization_msgs::Marker> markers;
  double lastPublishSimTime = -1.0;
};

SharedMarkerPublisher &MarkerPublisher()
{
  static SharedMarkerPublisher publisher;
  return publisher;
}
}

DynamicObstacle::~DynamicObstacle()
{
  if (!this->markerEnabled)
  {
    return;
  }

  SharedMarkerPublisher &shared = MarkerPublisher();
  std::lock_guard<std::mutex> lock(shared.mutex);
  shared.markers.erase(this->markerId);
  if (shared.markers.empty())
  {
    shared.publisher.shutdown();
    shared.node.reset();
    shared.topic.clear();
    shared.lastPublishSimTime = -1.0;
  }
}

double DynamicObstacle::ReadDouble(
    const sdf::ElementPtr &_sdf, const std::string &name, double fallback)
{
  return _sdf->HasElement(name) ? _sdf->Get<double>(name) : fallback;
}

int DynamicObstacle::ReadInt(
    const sdf::ElementPtr &_sdf, const std::string &name, int fallback)
{
  return _sdf->HasElement(name) ? _sdf->Get<int>(name) : fallback;
}

bool DynamicObstacle::ReadBool(
    const sdf::ElementPtr &_sdf, const std::string &name, bool fallback)
{
  return _sdf->HasElement(name) ? _sdf->Get<bool>(name) : fallback;
}

std::string DynamicObstacle::ReadString(
    const sdf::ElementPtr &_sdf,
    const std::string &name,
    const std::string &fallback)
{
  return _sdf->HasElement(name) ? _sdf->Get<std::string>(name) : fallback;
}

ignition::math::Vector3d DynamicObstacle::ReadVector(
    const sdf::ElementPtr &_sdf,
    const std::string &name,
    const ignition::math::Vector3d &fallback)
{
  return _sdf->HasElement(name)
      ? _sdf->Get<ignition::math::Vector3d>(name)
      : fallback;
}

void DynamicObstacle::Load(physics::ModelPtr _parent, sdf::ElementPtr _sdf)
{
  this->model = _parent;
  this->sdf = _sdf;
  this->initialPose = this->model->WorldPose();

  this->velocity = std::max(
      kEpsilon, ReadDouble(this->sdf, "velocity", this->velocity));
  this->orientation = ReadBool(this->sdf, "orientation", this->orientation);
  this->loop = ReadBool(this->sdf, "loop", this->loop);
  this->angularVelocity = ReadDouble(
      this->sdf, "angular_velocity", this->angularVelocity);

  const std::string motionType =
      ReadString(this->sdf, "motion_type", "path");
  if (motionType == "linear")
  {
    this->motionType = MotionType::LINEAR;
    this->lineStart = ReadVector(
        this->sdf, "line_start", this->initialPose.Pos());
    this->lineEnd = ReadVector(
        this->sdf, "line_end", this->initialPose.Pos());
    this->lineLength = this->lineStart.Distance(this->lineEnd);
    this->linePhase = ReadDouble(this->sdf, "phase", this->linePhase);
    if (this->lineLength <= kEpsilon)
    {
      gzerr << "DynamicObstacle '" << this->model->GetName()
            << "' has a zero-length linear path." << std::endl;
      return;
    }
  }
  else if (motionType == "circular")
  {
    this->motionType = MotionType::CIRCULAR;
    this->circleCenter = ReadVector(
        this->sdf, "circle_center", this->initialPose.Pos());
    this->circleRadius = std::max(
        kEpsilon, ReadDouble(this->sdf, "circle_radius", this->circleRadius));
    this->circlePhase = ReadDouble(this->sdf, "phase", this->circlePhase);
  }
  else if (motionType == "proximity_linear")
  {
    this->motionType = MotionType::PROXIMITY_LINEAR;
    this->proximityDirection = ReadVector(
        this->sdf, "direction", this->proximityDirection);
    this->proximityDirection.Z(0.0);
    const double directionLength = this->proximityDirection.Length();
    if (!std::isfinite(directionLength) || directionLength <= kEpsilon)
    {
      gzerr << "DynamicObstacle '" << this->model->GetName()
            << "' has a zero-length proximity direction." << std::endl;
      return;
    }
    this->proximityDirection /= directionLength;
    this->proximityPosition = this->initialPose.Pos();
    this->activationDistance = ReadDouble(
        this->sdf, "activation_distance", this->activationDistance);
    if (!std::isfinite(this->activationDistance) ||
        this->activationDistance <= 0.0)
    {
      gzerr << "DynamicObstacle '" << this->model->GetName()
            << "' requires a positive activation_distance." << std::endl;
      return;
    }
    if (!std::isfinite(this->velocity))
    {
      gzerr << "DynamicObstacle '" << this->model->GetName()
            << "' requires a finite velocity." << std::endl;
      return;
    }
    this->activationModelName = ReadString(
        this->sdf, "activation_model", this->activationModelName);
    if (this->activationModelName.empty())
    {
      gzerr << "DynamicObstacle '" << this->model->GetName()
            << "' requires a non-empty activation_model." << std::endl;
      return;
    }
  }
  else
  {
    this->motionType = MotionType::LEGACY_PATH;
    this->ConfigureLegacyPath();
  }

  this->markerEnabled = ReadBool(this->sdf, "marker_enabled", false);
  if (this->markerEnabled)
  {
    this->markerId = ReadInt(this->sdf, "marker_id", this->markerId);
    this->markerRadius = std::max(
        kEpsilon, ReadDouble(this->sdf, "marker_radius", this->markerRadius));
    this->markerHeight = std::max(
        kEpsilon, ReadDouble(this->sdf, "marker_height", this->markerHeight));
    this->markerRate = std::max(
        kEpsilon, ReadDouble(this->sdf, "marker_rate", this->markerRate));
    this->markerTopic = ReadString(
        this->sdf, "marker_topic", this->markerTopic);
    this->markerFrame = ReadString(
        this->sdf, "marker_frame", this->markerFrame);

    if (!ros::isInitialized())
    {
      gzerr << "DynamicObstacle marker publishing requires gazebo_ros_init; "
            << "motion will continue without RViz markers." << std::endl;
      this->markerEnabled = false;
    }
    else
    {
      SharedMarkerPublisher &shared = MarkerPublisher();
      std::lock_guard<std::mutex> lock(shared.mutex);
      if (!shared.node)
      {
        shared.node.reset(new ros::NodeHandle(""));
        shared.topic = this->markerTopic;
        shared.publisher =
            shared.node->advertise<visualization_msgs::MarkerArray>(
                shared.topic, 10, false);
        shared.lastPublishSimTime = this->model->GetWorld()->SimTime().Double();
      }
      else if (shared.topic != this->markerTopic)
      {
        gzerr << "DynamicObstacle MarkerArray publisher already uses topic '"
              << shared.topic << "'; cannot also publish to '"
              << this->markerTopic << "'." << std::endl;
        this->markerEnabled = false;
      }
    }
  }

  if (this->motionType != MotionType::LEGACY_PATH || this->markerEnabled)
  {
    this->updateConnection = event::Events::ConnectWorldUpdateBegin(
        std::bind(&DynamicObstacle::OnUpdate, this, std::placeholders::_1));
  }
}

void DynamicObstacle::ConfigureLegacyPath()
{
  this->path.clear();
  if (this->sdf->HasElement("path"))
  {
    sdf::ElementPtr waypointElem =
        this->sdf->GetElement("path")->GetElement("waypoint");
    while (waypointElem)
    {
      this->path.push_back(
          waypointElem->Get<ignition::math::Vector3d>());
      waypointElem = waypointElem->GetNextElement("waypoint");
    }
  }

  if (this->path.size() < 2)
  {
    gzerr << "DynamicObstacle '" << this->model->GetName()
          << "' requires at least two path waypoints." << std::endl;
    return;
  }

  if (this->loop)
  {
    this->path.push_back(this->path.front());
  }
  else
  {
    const std::vector<ignition::math::Vector3d> forwardPath = this->path;
    for (int i = static_cast<int>(forwardPath.size()) - 2; i >= 0; --i)
    {
      this->path.push_back(forwardPath[static_cast<std::size_t>(i)]);
    }
  }

  this->pathWithAngle.clear();
  if (this->orientation)
  {
    double yawLast = 0.0;
    for (std::size_t i = 0; i < this->path.size(); ++i)
    {
      const auto &current = this->path[i];
      const auto &next =
          i + 1 < this->path.size() ? this->path[i + 1] : this->path[1];
      const double yawCurrent =
          std::atan2(next.Y() - current.Y(), next.X() - current.X());

      if (i == 0)
      {
        this->pathWithAngle.push_back(
            {current.X(), current.Y(), current.Z(), yawCurrent});
      }
      else
      {
        this->pathWithAngle.push_back(
            {current.X(), current.Y(), current.Z(), yawLast});
        this->pathWithAngle.push_back(
            {current.X(), current.Y(), current.Z(), yawCurrent});
      }
      yawLast = yawCurrent;
    }
  }
  else
  {
    for (const auto &waypoint : this->path)
    {
      this->pathWithAngle.push_back(
          {waypoint.X(), waypoint.Y(), waypoint.Z(), 0.0});
    }
  }

  this->timeKnot.clear();
  double totalTime = 0.0;
  this->timeKnot.push_back(totalTime);
  for (std::size_t i = 0; i + 1 < this->pathWithAngle.size(); ++i)
  {
    const auto &current = this->pathWithAngle[i];
    const auto &next = this->pathWithAngle[i + 1];
    const bool rotation =
        current[0] == next[0] && current[1] == next[1] &&
        current[2] == next[2];

    double duration = 0.0;
    if (rotation)
    {
      const double angleDifference = std::abs(std::atan2(
          std::sin(next[3] - current[3]),
          std::cos(next[3] - current[3])));
      duration = angleDifference /
          std::max(kEpsilon, std::abs(this->angularVelocity));
    }
    else
    {
      const double distance = std::sqrt(
          std::pow(next[0] - current[0], 2) +
          std::pow(next[1] - current[1], 2) +
          std::pow(next[2] - current[2], 2));
      // Preserve the legacy plugin's integer-second path timing.
      duration = static_cast<double>(
          static_cast<int>(distance / this->velocity));
    }
    totalTime += std::max(kEpsilon, duration);
    this->timeKnot.push_back(totalTime);
  }

  gazebo::common::PoseAnimationPtr animation(
      new gazebo::common::PoseAnimation(
          "obstaclePathLoop", totalTime, true));
  for (std::size_t i = 0; i < this->pathWithAngle.size(); ++i)
  {
    const auto &pose = this->pathWithAngle[i];
    gazebo::common::PoseKeyFrame *key =
        animation->CreateKeyFrame(this->timeKnot[i]);
    key->Translation(
        ignition::math::Vector3d(pose[0], pose[1], pose[2]));
    key->Rotation(ignition::math::Quaterniond(0.0, 0.0, pose[3]));
  }
  this->model->SetAnimation(animation);
}

void DynamicObstacle::OnUpdate(const common::UpdateInfo &_info)
{
  if (!this->model)
  {
    return;
  }

  const double simTime = _info.simTime.Double();
  if (this->motionType == MotionType::PROXIMITY_LINEAR)
  {
    double dt = 0.0;
    bool timeReset = false;
    if (this->lastUpdateSimTime >= 0.0 && simTime >= this->lastUpdateSimTime)
    {
      dt = simTime - this->lastUpdateSimTime;
    }
    else if (this->lastUpdateSimTime >= 0.0)
    {
      timeReset = true;
      this->proximityPosition = this->model->WorldPose().Pos();
    }
    this->lastUpdateSimTime = simTime;

    if (!this->activationModel)
    {
      this->activationModel =
          this->model->GetWorld()->ModelByName(this->activationModelName);
    }

    bool active = false;
    if (this->activationModel && !timeReset)
    {
      const ignition::math::Vector3d offset =
          this->activationModel->WorldPose().Pos() - this->proximityPosition;
      active = std::hypot(offset.X(), offset.Y()) <= this->activationDistance;
    }

    ignition::math::Vector3d linearVelocity =
        ignition::math::Vector3d::Zero;
    if (active)
    {
      linearVelocity = this->velocity * this->proximityDirection;
      const double requestedTravel = dt * this->velocity;
      const ignition::math::Vector3d relative =
          this->proximityPosition - this->activationModel->WorldPose().Pos();
      const double projection =
          relative.X() * this->proximityDirection.X() +
          relative.Y() * this->proximityDirection.Y();
      const double radiusResidual =
          relative.X() * relative.X() + relative.Y() * relative.Y() -
          this->activationDistance * this->activationDistance;
      const double exitTravel = -projection + std::sqrt(std::max(
          0.0, projection * projection - radiusResidual));
      const double actualTravel = std::min(requestedTravel, exitTravel);
      this->proximityPosition += actualTravel * this->proximityDirection;
      if (actualTravel + kEpsilon < requestedTravel)
      {
        linearVelocity = ignition::math::Vector3d::Zero;
      }
    }
    this->model->SetWorldPose(
        ignition::math::Pose3d(
            this->proximityPosition, this->initialPose.Rot()));
    this->model->SetLinearVel(linearVelocity);
    this->model->SetAngularVel(ignition::math::Vector3d::Zero);

    if (this->markerEnabled)
    {
      this->UpdateMarkerArray(simTime);
    }
    return;
  }

  if (!this->motionStarted)
  {
    this->motionStarted = true;
    this->motionStartSimTime = simTime;
  }
  const double elapsed = std::max(0.0, simTime - this->motionStartSimTime);

  if (this->motionType == MotionType::LINEAR)
  {
    double pathPhase = std::fmod(
        this->linePhase + this->velocity * elapsed / this->lineLength, 2.0);
    if (pathPhase < 0.0)
    {
      pathPhase += 2.0;
    }
    const bool forward = pathPhase <= 1.0;
    const double progress = forward ? pathPhase : 2.0 - pathPhase;
    const ignition::math::Vector3d direction =
        (this->lineEnd - this->lineStart) / this->lineLength;
    const ignition::math::Vector3d position =
        this->lineStart + progress * (this->lineEnd - this->lineStart);
    const ignition::math::Vector3d linearVelocity =
        (forward ? 1.0 : -1.0) * this->velocity * direction;

    this->model->SetWorldPose(
        ignition::math::Pose3d(position, this->initialPose.Rot()));
    this->model->SetLinearVel(linearVelocity);
    this->model->SetAngularVel(ignition::math::Vector3d::Zero);
  }
  else if (this->motionType == MotionType::CIRCULAR)
  {
    const double angle =
        this->circlePhase + this->angularVelocity * elapsed;
    const double cosine = std::cos(angle);
    const double sine = std::sin(angle);
    const ignition::math::Vector3d position(
        this->circleCenter.X() + this->circleRadius * cosine,
        this->circleCenter.Y() + this->circleRadius * sine,
        this->circleCenter.Z());
    const ignition::math::Vector3d linearVelocity(
        -this->circleRadius * this->angularVelocity * sine,
        this->circleRadius * this->angularVelocity * cosine,
        0.0);

    this->model->SetWorldPose(
        ignition::math::Pose3d(position, this->initialPose.Rot()));
    this->model->SetLinearVel(linearVelocity);
    this->model->SetAngularVel(ignition::math::Vector3d::Zero);
  }

  if (this->markerEnabled)
  {
    this->UpdateMarkerArray(simTime);
  }
}

void DynamicObstacle::UpdateMarkerArray(double simTime)
{
  const double markerPeriod = 1.0 / this->markerRate;
  const ignition::math::Pose3d pose = this->model->WorldPose();
  visualization_msgs::Marker marker;
  marker.header.frame_id = this->markerFrame;
  marker.header.stamp = ros::Time::now();
  marker.ns = "dynamic_obstacles";
  marker.id = this->markerId;
  marker.type = visualization_msgs::Marker::CYLINDER;
  marker.action = visualization_msgs::Marker::ADD;
  marker.pose.position.x = pose.Pos().X();
  marker.pose.position.y = pose.Pos().Y();
  marker.pose.position.z = pose.Pos().Z();
  marker.pose.orientation.x = pose.Rot().X();
  marker.pose.orientation.y = pose.Rot().Y();
  marker.pose.orientation.z = pose.Rot().Z();
  marker.pose.orientation.w = pose.Rot().W();
  marker.scale.x = 2.0 * this->markerRadius;
  marker.scale.y = 2.0 * this->markerRadius;
  marker.scale.z = this->markerHeight;
  marker.color.r = 1.0;
  marker.color.g = 0.35;
  marker.color.b = 0.0;
  marker.color.a = 0.65;
  marker.lifetime = ros::Duration(2.5 * markerPeriod);

  SharedMarkerPublisher &shared = MarkerPublisher();
  std::lock_guard<std::mutex> lock(shared.mutex);
  shared.markers[this->markerId] = marker;
  if (shared.lastPublishSimTime >= 0.0 &&
      simTime < shared.lastPublishSimTime)
  {
    // Gazebo resets simulation time when the world is reset.
    shared.lastPublishSimTime = -1.0;
  }
  if (shared.lastPublishSimTime >= 0.0 &&
      simTime - shared.lastPublishSimTime < markerPeriod)
  {
    return;
  }
  shared.lastPublishSimTime = simTime;

  visualization_msgs::MarkerArray markerArray;
  markerArray.markers.reserve(shared.markers.size());
  for (const auto &entry : shared.markers)
  {
    markerArray.markers.push_back(entry.second);
  }
  shared.publisher.publish(markerArray);
}
}
