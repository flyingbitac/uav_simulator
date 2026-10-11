// Optical-flow camera sensor plugin: dense per-pixel flow between consecutive
// frames of the attached camera, published next to the gazebo_ros camera
// stream as <model>/optical_flow/flow (32FC2) and <model>/optical_flow/flow_vis
// (bgr8 color wheel). Flow follows the policy optical-flow contract
// (cfg/sensor/optical_flow.yaml): per-frame pixel displacement of the image
// content, channel 0 horizontal right+, channel 1 vertical down+.

#include <gazebo/common/Plugin.hh>
#include <gazebo/gazebo.hh>
#include <gazebo/rendering/Camera.hh>
#include <gazebo/sensors/CameraSensor.hh>
#include <gazebo/sensors/Sensor.hh>

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>

#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

namespace gazebo
{
class OpticalFlowCameraPlugin : public SensorPlugin
{
public:
  void Load(sensors::SensorPtr sensor, sdf::ElementPtr sdf) override
  {
    if (!ros::isInitialized())
    {
      ROS_FATAL_STREAM("OpticalFlowCameraPlugin requires gazebo_ros_init.");
      return;
    }

    camera_sensor_ = std::dynamic_pointer_cast<sensors::CameraSensor>(sensor);
    if (!camera_sensor_)
    {
      gzerr << "OpticalFlowCameraPlugin must be attached to a camera sensor.\n";
      return;
    }

    rendering::CameraPtr camera = camera_sensor_->Camera();
    if (!camera)
    {
      gzerr << "OpticalFlowCameraPlugin: sensor has no rendering camera.\n";
      return;
    }

    frame_id_ = "optical_flow_link";
    if (sdf->HasElement("frameName"))
    {
      frame_id_ = sdf->Get<std::string>("frameName");
    }

    output_rate_ = 30.0;
    if (sdf->HasElement("outputRate"))
    {
      output_rate_ = sdf->Get<double>("outputRate");
    }

    // Flow is computed on frames downscaled by this factor and published in
    // that pixel grid (displacements in flow-grid pixels). The camera stream
    // can be much larger than the flow consumers need.
    flow_scale_ = 0.5;
    if (sdf->HasElement("flowScale"))
    {
      flow_scale_ = sdf->Get<double>("flowScale");
    }

    std::string topic_base;
    if (sdf->HasElement("topicBase"))
    {
      topic_base = sdf->Get<std::string>("topicBase");
    }
    else
    {
      // Same namespace rule as gazebo_ros GetRobotNamespace: the root model
      // name, so the flow topics land next to gazebo_ros_camera's image topics
      // (e.g. quad0 -> /quad0/optical_flow/flow).
      std::string scoped = sensor->ParentName();
      const std::string::size_type cut = scoped.find("::");
      topic_base =
          "/" + (cut == std::string::npos ? scoped : scoped.substr(0, cut)) +
          "/optical_flow";
    }

    flow_pub_ = node_handle_.advertise<sensor_msgs::Image>(topic_base + "/flow", 1);
    flow_vis_pub_ = node_handle_.advertise<sensor_msgs::Image>(topic_base + "/flow_vis", 1);

    update_connection_ = camera_sensor_->ConnectUpdated(
        [this]() { this->OnSensorUpdate(); });

    camera_sensor_->SetActive(true);

    ROS_INFO_STREAM("OpticalFlowCameraPlugin publishing dense flow from "
                    << camera_sensor_->Name() << " to " << topic_base);
  }

private:
  // Triggered on sensor updates rather than on the camera object's frame
  // signal: the render camera can be re-created across sensor de/reactivation,
  // which silently orphaned a frame-signal connection.
  void OnSensorUpdate()
  {
    rendering::CameraPtr camera = camera_sensor_->Camera();
    if (!camera)
    {
      return;
    }

    const common::Time frame_time = camera->LastRenderWallTime();
    if (has_last_frame_time_ && frame_time <= last_frame_time_)
    {
      return;  // no new frame since the last look
    }

    if (has_last_flow_time_ &&
        (frame_time - last_flow_time_).Double() < 1.0 / output_rate_)
    {
      last_frame_time_ = frame_time;
      has_last_frame_time_ = true;
      return;
    }

    const unsigned char *image = camera->ImageData(0);
    const unsigned int width = camera->ImageWidth();
    const unsigned int height = camera->ImageHeight();
    if (image == nullptr || width == 0 || height == 0)
    {
      ROS_WARN_THROTTLE(5.0, "OpticalFlowCameraPlugin: empty camera frame");
      return;
    }

    last_frame_time_ = frame_time;
    has_last_frame_time_ = true;

    cv::Mat rgb(static_cast<int>(height), static_cast<int>(width), CV_8UC3,
                const_cast<unsigned char *>(image));
    cv::Mat gray_full, gray;
    cv::cvtColor(rgb, gray_full, cv::COLOR_RGB2GRAY);
    if (flow_scale_ != 1.0)
    {
      cv::resize(gray_full, gray, cv::Size(), flow_scale_, flow_scale_,
                 cv::INTER_AREA);
    }
    else
    {
      gray = gray_full;
    }

    try
    {
      if (!prev_gray_.empty() && prev_gray_.size() == gray.size())
      {
        cv::Mat flow;
        cv::calcOpticalFlowFarneback(prev_gray_, gray, flow, 0.5, 3, 15, 3, 5,
                                     1.2, 0);
        Publish(flow);
      }
    }
    catch (const cv::Exception &e)
    {
      ROS_WARN_THROTTLE(5.0, "OpticalFlowCameraPlugin: OpenCV error: %s",
                        e.what());
      prev_gray_.release();
      return;
    }
    prev_gray_ = gray;
    last_flow_time_ = frame_time;
    has_last_flow_time_ = true;
  }

  void Publish(const cv::Mat &flow)
  {
    const ros::Time stamp = ros::Time::now();

    sensor_msgs::Image flow_msg;
    flow_msg.header.stamp = stamp;
    flow_msg.header.frame_id = frame_id_;
    flow_msg.height = static_cast<uint32_t>(flow.rows);
    flow_msg.width = static_cast<uint32_t>(flow.cols);
    flow_msg.encoding = sensor_msgs::image_encodings::TYPE_32FC2;
    flow_msg.is_bigendian = 0;
    flow_msg.step = static_cast<uint32_t>(flow.cols * 2 * sizeof(float));
    const cv::Mat contiguous = flow.isContinuous() ? flow : flow.clone();
    flow_msg.data.resize(static_cast<size_t>(contiguous.rows) * flow_msg.step);
    std::memcpy(flow_msg.data.data(), contiguous.data, flow_msg.data.size());
    flow_pub_.publish(flow_msg);

    flow_vis_pub_.publish(ToVis(flow, stamp));
  }

  sensor_msgs::Image ToVis(const cv::Mat &flow, const ros::Time &stamp) const
  {
    std::vector<cv::Mat> xy(2);
    cv::split(flow, xy);
    cv::Mat magnitude, angle;
    cv::cartToPolar(xy[0], xy[1], angle, magnitude);

    // Hue encodes direction on the color wheel (half the angle fits OpenCV's
    // 0-180 hue range); value encodes magnitude, saturating at
    // kVisMaxFlowPx pixels per frame.
    cv::Mat hue, value;
    const cv::Mat hue_angle = angle * (90.0f / static_cast<float>(CV_PI));
    hue_angle.convertTo(hue, CV_8U);
    magnitude.convertTo(value, CV_8U, 255.0 / kVisMaxFlowPx);
    cv::Mat saturation(flow.size(), CV_8U, cv::Scalar::all(255));

    std::vector<cv::Mat> hsv_channels = {hue, saturation, value};
    cv::Mat hsv, bgr;
    cv::merge(hsv_channels, hsv);
    cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);

    sensor_msgs::Image vis_msg;
    vis_msg.header.stamp = stamp;
    vis_msg.header.frame_id = frame_id_;
    vis_msg.height = static_cast<uint32_t>(bgr.rows);
    vis_msg.width = static_cast<uint32_t>(bgr.cols);
    vis_msg.encoding = sensor_msgs::image_encodings::BGR8;
    vis_msg.is_bigendian = 0;
    const cv::Mat contiguous = bgr.isContinuous() ? bgr : bgr.clone();
    vis_msg.step = static_cast<uint32_t>(contiguous.cols * 3);
    vis_msg.data.resize(static_cast<size_t>(contiguous.rows) * vis_msg.step);
    std::memcpy(vis_msg.data.data(), contiguous.data, vis_msg.data.size());
    return vis_msg;
  }

  sensors::CameraSensorPtr camera_sensor_;
  event::ConnectionPtr update_connection_;
  ros::NodeHandle node_handle_;
  ros::Publisher flow_pub_;
  ros::Publisher flow_vis_pub_;
  cv::Mat prev_gray_;
  std::string frame_id_;
  double output_rate_ = 30.0;
  double flow_scale_ = 0.5;
  common::Time last_flow_time_;
  bool has_last_flow_time_ = false;
  common::Time last_frame_time_;
  bool has_last_frame_time_ = false;

  // Flow color wheel saturates at this many pixels of displacement per frame.
  static constexpr double kVisMaxFlowPx = 20.0;
};

GZ_REGISTER_SENSOR_PLUGIN(OpticalFlowCameraPlugin)
}  // namespace gazebo
