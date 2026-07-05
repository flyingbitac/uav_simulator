#include <gazebo/common/Plugin.hh>
#include <gazebo/gazebo.hh>
#include <gazebo/msgs/contacts.pb.h>
#include <gazebo/sensors/ContactSensor.hh>
#include <gazebo/sensors/Sensor.hh>

#include <ros/ros.h>
#include <std_msgs/Bool.h>

#include <boost/algorithm/string/predicate.hpp>
#include <boost/bind.hpp>
#include <boost/pointer_cast.hpp>
#include <boost/shared_ptr.hpp>

#include <string>

namespace gazebo
{
class CollisionSignalSensorPlugin : public SensorPlugin
{
public:
  void Load(sensors::SensorPtr sensor, sdf::ElementPtr sdf) override
  {
    if (!ros::isInitialized())
    {
      ROS_FATAL_STREAM("CollisionSignalSensorPlugin requires gazebo_ros_init.");
      return;
    }

    contact_sensor_ = boost::dynamic_pointer_cast<sensors::ContactSensor>(sensor);
    if (!contact_sensor_)
    {
      ROS_FATAL_STREAM("CollisionSignalSensorPlugin must be attached to a contact sensor.");
      return;
    }

    if (sdf->HasElement("topicName"))
    {
      topic_name_ = sdf->Get<std::string>("topicName");
    }
    if (sdf->HasElement("selfModelName"))
    {
      self_model_name_ = sdf->Get<std::string>("selfModelName");
    }

    collision_pub_ = node_handle_.advertise<std_msgs::Bool>(topic_name_, 10, false);
    update_connection_ = contact_sensor_->ConnectUpdated(
        boost::bind(&CollisionSignalSensorPlugin::OnUpdate, this));
    contact_sensor_->SetActive(true);

    ROS_INFO_STREAM(
        "CollisionSignalSensorPlugin publishing external contacts from "
        << contact_sensor_->Name() << " to " << topic_name_);
  }

private:
  void OnUpdate()
  {
    const gazebo::msgs::Contacts contacts = contact_sensor_->Contacts();
    bool external_contact = false;
    for (int i = 0; i < contacts.contact_size(); ++i)
    {
      const gazebo::msgs::Contact &contact = contacts.contact(i);
      if (!IsSelfContact(contact.collision1(), contact.collision2()))
      {
        external_contact = true;
        break;
      }
    }

    if (external_contact && !in_contact_)
    {
      std_msgs::Bool msg;
      msg.data = true;
      collision_pub_.publish(msg);
      ROS_WARN_STREAM(
          "Gazebo collision detected by " << contact_sensor_->Name()
                                          << " on " << topic_name_);
    }
    in_contact_ = external_contact;
  }

  bool IsSelfContact(const std::string &collision1, const std::string &collision2) const
  {
    return HasSelfModelScope(collision1) && HasSelfModelScope(collision2);
  }

  bool HasSelfModelScope(const std::string &collision_name) const
  {
    return collision_name == self_model_name_ ||
           boost::starts_with(collision_name, self_model_name_ + "::") ||
           collision_name.find("::" + self_model_name_ + "::") != std::string::npos;
  }

  sensors::ContactSensorPtr contact_sensor_;
  event::ConnectionPtr update_connection_;
  ros::NodeHandle node_handle_;
  ros::Publisher collision_pub_;
  std::string topic_name_ = "/drone_0/collision";
  std::string self_model_name_ = "iris";
  bool in_contact_ = false;
};

GZ_REGISTER_SENSOR_PLUGIN(CollisionSignalSensorPlugin)
}  // namespace gazebo
