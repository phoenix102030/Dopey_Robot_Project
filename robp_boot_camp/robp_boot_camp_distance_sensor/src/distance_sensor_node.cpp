// Boot camp
#include <robp_boot_camp_distance_sensor/distance_sensor.h>

#include <robp_boot_camp_interfaces/msg/ad_converter.hpp>

// ROS
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/empty.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

// STL
#include <cmath>
#include <memory>
#include <optional>

class DistanceSensorNode : public rclcpp::Node
{
 public:
	DistanceSensorNode() : Node("distance_sensor")
	{
		distance_sensor_ = std::make_unique<DistanceSensor>();

		tf_buffer_   = std::make_unique<tf2_ros::Buffer>(this->get_clock());
		tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

		distance_sensor_vis_pub_ =
		    this->create_publisher<visualization_msgs::msg::MarkerArray>(
		        "dist_sensor_markers", 0);
		adc_pub_ = this->create_publisher<robp_boot_camp_interfaces::msg::ADConverter>(
		    "kobuki/adc", 1);

		world_sub_ = this->create_subscription<visualization_msgs::msg::Marker>(
		    "/wall_marker", 1,
		    std::bind(&DistanceSensorNode::worldCallback, this, std::placeholders::_1));
		reset_sub_ = this->create_subscription<std_msgs::msg::Empty>(
		    "/reset_distance", 1,
		    std::bind(&DistanceSensorNode::resetCallback, this, std::placeholders::_1));

		using namespace std::chrono_literals;
		timer_ = this->create_wall_timer(1s / 30.0, std::bind(&DistanceSensorNode::publish, this));
	}

	void publish()
	{
		geometry_msgs::msg::TransformStamped        transform_back_sensor;
		geometry_msgs::msg::TransformStamped        transform_front_sensor;
		visualization_msgs::msg::MarkerArray        markers;
		robp_boot_camp_interfaces::msg::ADConverter adc_msg;

		using namespace std::chrono_literals;

		try {
			transform_back_sensor = tf_buffer_->lookupTransform(
			    "odom", "distance_sensor_back_link", this->now(), 5s);
			transform_front_sensor = tf_buffer_->lookupTransform(
			    "odom", "distance_sensor_front_link", this->now(), 5s);

			// Back distance sensor
			// convert range to odom frame of ref and check for intersection
			geometry_msgs::msg::PointStamped start_point, start_point_out, end_point,
			    end_point_out;
			start_point.point.y         = 0;
			end_point.point.y           = -max_sensor_range_;
			start_point.header.stamp    = this->now();
			start_point.header.frame_id = "/distance_sensor_back_link";
			end_point.header            = start_point.header;
			tf2::doTransform(start_point, start_point_out, transform_back_sensor);
			tf2::doTransform(end_point, end_point_out, transform_back_sensor);

			if (auto intersection_point = segmentIntersection(
			        wall_start_, wall_end_, start_point_out.point, end_point_out.point)) {
				intersection_point->z = sensor_height_;
				// publish distance
				double distance = pointDistance(start_point_out.point, *intersection_point);
				if (distance > min_sensor_range_ && distance < max_sensor_range_) {
					adc_msg.ch2 =
					    static_cast<unsigned>(distance_sensor_->sample(distance) * (1023 / 5.0));

					// marker
					visualization_msgs::msg::Marker marker;
					marker.header             = start_point_out.header;
					marker.lifetime           = rclcpp::Duration(1s / 10.0);
					marker.ns                 = "distance_back";
					marker.id                 = 0;
					marker.type               = visualization_msgs::msg::Marker::LINE_STRIP;
					marker.action             = visualization_msgs::msg::Marker::ADD;
					marker.scale.x            = 0.01;
					marker.scale.y            = 0.1;
					marker.scale.z            = 0.1;
					marker.color.a            = 1.0;
					marker.color.r            = 0.0;
					marker.color.g            = 1.0;
					marker.color.b            = 1.0;
					marker.pose.orientation.w = 1.0;
					marker.points.push_back(start_point_out.point);
					marker.points.push_back(*intersection_point);
					markers.markers.push_back(marker);
				}
			}

			// Front distance sensor
			// convert range to odom frame of ref and check for intersection
			tf2::doTransform(start_point, start_point_out, transform_front_sensor);
			tf2::doTransform(end_point, end_point_out, transform_front_sensor);

			if (auto intersection_point = segmentIntersection(
			        wall_start_, wall_end_, start_point_out.point, end_point_out.point)) {
				intersection_point->z = sensor_height_;
				// publish distance
				double distance = pointDistance(start_point_out.point, *intersection_point);
				if (distance > min_sensor_range_ && distance < max_sensor_range_) {
					adc_msg.ch1 =
					    static_cast<unsigned>(distance_sensor_->sample(distance) * (1023 / 5.0));

					// marker
					visualization_msgs::msg::Marker marker;
					marker.header             = start_point_out.header;
					marker.lifetime           = rclcpp::Duration(1s / 10.0);
					marker.ns                 = "distance_front";
					marker.id                 = 1;
					marker.type               = visualization_msgs::msg::Marker::LINE_STRIP;
					marker.action             = visualization_msgs::msg::Marker::ADD;
					marker.scale.x            = 0.01;
					marker.scale.y            = 0.1;
					marker.scale.z            = 0.1;
					marker.color.a            = 1.0;
					marker.color.r            = 0.0;
					marker.color.g            = 1.0;
					marker.color.b            = 1.0;
					marker.pose.orientation.w = 1.0;
					marker.points.push_back(start_point_out.point);
					marker.points.push_back(*intersection_point);
					markers.markers.push_back(marker);
				}
			}

			adc_pub_->publish(adc_msg);

			if (!markers.markers.empty()) {
				distance_sensor_vis_pub_->publish(std::move(markers));
			}
		} catch (tf2::TransformException const &ex) {
			RCLCPP_WARN(this->get_logger(), "%s", ex.what());
		}
	}

 private:
	void worldCallback(visualization_msgs::msg::Marker const &msg)
	{
		if (world_initialized_) {
			return;
		}

		world_initialized_ = true;
		tf2::Transform wall_start(tf2::Quaternion::getIdentity(),
		                          tf2::Vector3(-200.0, 0.0, 0.0));
		tf2::Transform wall_end(tf2::Quaternion::getIdentity(),
		                        tf2::Vector3(200.0, 0.0, 0.0));

		tf2::Transform wall_pose;
		tf2::fromMsg(msg.pose, wall_pose);

		tf2::toMsg((wall_pose * wall_start).getOrigin(), wall_start_);
		tf2::toMsg((wall_pose * wall_end).getOrigin(), wall_end_);

		RCLCPP_INFO(this->get_logger(), "World initialized");
	}

	void resetCallback(std_msgs::msg::Empty const &) { world_initialized_ = false; }

	static std::optional<geometry_msgs::msg::Point> segmentIntersection(
	    geometry_msgs::msg::Point p1, geometry_msgs::msg::Point p2,
	    geometry_msgs::msg::Point p3, geometry_msgs::msg::Point p4)
	{
		float x1 = p1.x, x2 = p2.x, x3 = p3.x, x4 = p4.x;
		float y1 = p1.y, y2 = p2.y, y3 = p3.y, y4 = p4.y;

		float d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);
		// If d is zero, there is no intersection
		if (d == 0) {
			return std::nullopt;
		}

		// Get the x and y
		float pre = (x1 * y2 - y1 * x2), post = (x3 * y4 - y3 * x4);
		float x = (pre * (x3 - x4) - (x1 - x2) * post) / d;
		float y = (pre * (y3 - y4) - (y1 - y2) * post) / d;

		// Check if the x and y coordinates are within both lines
		if (x < std::min(x1, x2) || x > std::max(x1, x2) || x < std::min(x3, x4) ||
		    x > std::max(x3, x4)) {
			return std::nullopt;
		}
		if (y < std::min(y1, y2) || y > std::max(y1, y2) || y < std::min(y3, y4) ||
		    y > std::max(y3, y4)) {
			return std::nullopt;
		}

		// Return the point of intersection
		geometry_msgs::msg::Point intersection;
		intersection.x = x;
		intersection.y = y;
		return std::optional<geometry_msgs::msg::Point>{std::move(intersection)};
	}

	static double pointDistance(geometry_msgs::msg::Point p1, geometry_msgs::msg::Point p2)
	{
		return std::sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y));
	}

 private:
	rclcpp::TimerBase::SharedPtr timer_;

	std::unique_ptr<DistanceSensor> distance_sensor_;

	std::shared_ptr<tf2_ros::TransformListener> tf_listener_{nullptr};
	std::unique_ptr<tf2_ros::Buffer>            tf_buffer_;

	rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
	    distance_sensor_vis_pub_;
	rclcpp::Publisher<robp_boot_camp_interfaces::msg::ADConverter>::SharedPtr adc_pub_;

	rclcpp::Subscription<visualization_msgs::msg::Marker>::SharedPtr world_sub_;
	rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr            reset_sub_;

	double max_sensor_range_{0.8};  // meters
	double min_sensor_range_{0.1};
	double sensor_height_{0.1};
	bool   world_initialized_{false};

	geometry_msgs::msg::Point wall_start_, wall_end_;
};

int main(int argc, char *argv[])
{
	rclcpp::init(argc, argv);
	rclcpp::spin(std::make_shared<DistanceSensorNode>());
	rclcpp::shutdown();
	return 0;
}