// ROS
#include <angles/angles.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/empty.hpp>
#include <std_srvs/srv/empty.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <visualization_msgs/msg/marker.hpp>

// STL
#include <cmath>
#include <memory>
#include <random>

class World : public rclcpp::Node
{
 public:
	World() : Node("world")
	{
		tf_buffer_   = std::make_unique<tf2_ros::Buffer>(this->get_clock());
		tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

		vis_pub_ = this->create_publisher<visualization_msgs::msg::Marker>("wall_marker", 0);
		reset_pub_ = this->create_publisher<std_msgs::msg::Empty>("reset_distance", 0);
		reset_srv_ = this->create_service<std_srvs::srv::Empty>(
		    "reset_world",
		    std::bind(&World::reset, this, std::placeholders::_1, std::placeholders::_2));

		wall_marker_.header.frame_id = "odom";
		// wall_marker_.header.stamp = this->now();
		wall_marker_.ns      = "world";
		wall_marker_.id      = 0;
		wall_marker_.type    = visualization_msgs::msg::Marker::CUBE;
		wall_marker_.action  = visualization_msgs::msg::Marker::ADD;
		wall_marker_.scale.x = 500.0;
		wall_marker_.scale.y = 0.01;
		wall_marker_.scale.z = 0.2;
		wall_marker_.color.a = 1.0;
		wall_marker_.color.r = (255.0 / 255.0);
		wall_marker_.color.g = (0.0 / 255.0);
		wall_marker_.color.b = (0.0 / 255.0);
		setRandom();

		using namespace std::chrono_literals;
		timer_ = this->create_wall_timer(100ms, std::bind(&World::publish, this));
	}

	void publish() { vis_pub_->publish(wall_marker_); }

 private:
	double randomAngle() const
	{
		// Create random number generators
		static std::random_device               rd;
		static std::mt19937                     gen(rd());
		static std::uniform_real_distribution<> dis(angles::from_degrees(-20.0),
		                                            angles::from_degrees(20.0));
		return dis(gen);
	}

	void setRandom(tf2::Transform t = tf2::Transform())
	{
		auto angle_z = randomAngle();

		tf2::Quaternion q = t.getRotation();
		tf2::Vector3    p = t.inverse().getOrigin();

		RCLCPP_INFO(this->get_logger(), "Robot angle: %f", q.getAngle());
		RCLCPP_INFO(this->get_logger(), "Robot axis: %f %f %f", q.getAxis().getX(),
		            q.getAxis().getY(), q.getAxis().getZ());
		RCLCPP_INFO(this->get_logger(), "Robot position: %f %f %f", t.getOrigin().getX(),
		            t.getOrigin().getY(), t.getOrigin().getZ());

		if (angles::from_degrees(5.0) >= angle_z && angles::from_degrees(0.0) <= angle_z) {
			angle_z += angles::from_degrees(5.0);
		}

		if (angles::from_degrees(-5.0) <= angle_z && angles::from_degrees(0.0) >= angle_z) {
			angle_z -= angles::from_degrees(5.0);
		}

		angle_z += q.getAngle() * (-q.getAxis().getZ());

		p.setX(p.getX() - 0.4 * std::sin(angle_z));
		p.setY(p.getY() + 0.4 * std::cos(angle_z));
		p.setZ(std::isnan(p.getZ()) ? 0.1 : p.getZ() + 0.1);
		q.setRotation(tf2::Vector3(0, 0, 1), angle_z);

		tf2::toMsg(p, wall_marker_.pose.position);
		wall_marker_.pose.orientation = tf2::toMsg(q);
	}

	void reset(std::shared_ptr<std_srvs::srv::Empty::Request> const /* request */,
	           std::shared_ptr<std_srvs::srv::Empty::Response> /* response */)
	{
		reset_pub_->publish(std_msgs::msg::Empty());

		tf2::Transform transform{};
		try {
			tf2::fromMsg(
			    tf_buffer_->lookupTransform("base_link", "odom", rclcpp::Time(0)).transform,
			    transform);

		} catch (tf2::TransformException &ex) {
			RCLCPP_WARN(this->get_logger(), "%s", ex.what());
		}

		setRandom(transform);
	}

 private:
	rclcpp::TimerBase::SharedPtr timer_;

	visualization_msgs::msg::Marker wall_marker_;

	std::shared_ptr<tf2_ros::TransformListener> tf_listener_{nullptr};
	std::unique_ptr<tf2_ros::Buffer>            tf_buffer_;

	rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr vis_pub_;
	rclcpp::Publisher<std_msgs::msg::Empty>::SharedPtr            reset_pub_;

	rclcpp::Service<std_srvs::srv::Empty>::SharedPtr reset_srv_;
};

int main(int argc, char *argv[])
{
	rclcpp::init(argc, argv);
	rclcpp::spin(std::make_shared<World>());
	rclcpp::shutdown();
	return 0;
}