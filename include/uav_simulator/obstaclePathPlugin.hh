#ifndef OBSTACLEPATHPLUGIN_HH
#define OBSTACLEPATHPLUGIN_HH

#include <memory>
#include <string>
#include <vector>

#include <gazebo/common/common.hh>
#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <ignition/math.hh>
#include <ros/ros.h>
#include <sdf/sdf.hh>
#include <visualization_msgs/Marker.h>

namespace gazebo
{
  class DynamicObstacle : public ModelPlugin
  {
  public:
    void Load(physics::ModelPtr _parent, sdf::ElementPtr _sdf) override;

  private:
    enum class MotionType
    {
      LEGACY_PATH,
      LINEAR,
      CIRCULAR,
    };

    void ConfigureLegacyPath();
    void OnUpdate(const common::UpdateInfo &_info);
    void PublishMarker(double simTime);

    static double ReadDouble(
        const sdf::ElementPtr &_sdf, const std::string &name, double fallback);
    static int ReadInt(
        const sdf::ElementPtr &_sdf, const std::string &name, int fallback);
    static bool ReadBool(
        const sdf::ElementPtr &_sdf, const std::string &name, bool fallback);
    static std::string ReadString(
        const sdf::ElementPtr &_sdf, const std::string &name, const std::string &fallback);
    static ignition::math::Vector3d ReadVector(
        const sdf::ElementPtr &_sdf,
        const std::string &name,
        const ignition::math::Vector3d &fallback);

    physics::ModelPtr model;
    event::ConnectionPtr updateConnection;
    sdf::ElementPtr sdf;

    MotionType motionType = MotionType::LEGACY_PATH;
    ignition::math::Pose3d initialPose;
    bool motionStarted = false;
    double motionStartSimTime = 0.0;

    double velocity = 1.0;
    bool orientation = true;
    bool loop = false;
    double angularVelocity = 0.8;

    ignition::math::Vector3d lineStart;
    ignition::math::Vector3d lineEnd;
    double lineLength = 0.0;
    double linePhase = 0.5;

    ignition::math::Vector3d circleCenter;
    double circleRadius = 1.0;
    double circlePhase = 0.0;

    std::vector<ignition::math::Vector3d> path;
    std::vector<std::vector<double>> pathWithAngle;
    std::vector<double> timeKnot;

    bool markerEnabled = false;
    int markerId = 0;
    double markerRadius = 0.4;
    double markerHeight = 1.0;
    double markerRate = 20.0;
    double lastMarkerPublishSimTime = -1.0;
    std::string markerTopic = "/uav_simulator/dynamic_obstacles";
    std::string markerFrame = "map";
    std::unique_ptr<ros::NodeHandle> rosNode;
    ros::Publisher markerPublisher;
  };

  GZ_REGISTER_MODEL_PLUGIN(DynamicObstacle)
}

#endif
