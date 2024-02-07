// ROBP
#include <robp_boot_camp_motors/kobuki_motors.h>

#include <robp_interfaces/msg/duty_cycles.hpp>
#include <robp_interfaces/msg/encoders.hpp>

// ROS
#include <geometry_msgs/msg/twist.hpp>
#include <rclcpp/rclcpp.hpp>

// STL
#include <memory>
#include <vector>

class Motors : public rclcpp::Node
{
 public:
	Motors() : Node("motors")
	{
		kobuki_motors_ = std::make_unique<KobukiMotors>();

		encoders_pub_ =
		    this->create_publisher<robp_interfaces::msg::Encoders>("/motor/encoders", 1);
		twist_pub_ = this->create_publisher<geometry_msgs::msg::Twist>(
		    "/mobile_base/commands/velocity", 1);

		duty_cycles_sub_ = this->create_subscription<robp_interfaces::msg::DutyCycles>(
		    "/motor/duty_cycles", 1,
		    std::bind(&Motors::dutyCyclesCallback, this, std::placeholders::_1));

		duty_cyles_time_ = this->now();

		using namespace std::chrono_literals;
		timer_ = this->create_wall_timer(1s / 10.0, std::bind(&Motors::updateMotors, this));
	}

	// [0] corresponds to left wheel, [1] corresponds to right wheel
	void dutyCyclesCallback(robp_interfaces::msg::DutyCycles const &msg)
	{
		duty_cycles_[0]  = msg.duty_cycle_left;
		duty_cycles_[1]  = msg.duty_cycle_right;
		duty_cyles_time_ = this->now();
	}

	void updateMotors()
	{
		// if more than 0.5 seconds have passed and no messages have been received,
		// shutdown the motors
		if ((this->now() - duty_cyles_time_).seconds() > 0.5) {
			duty_cycles_[0] = 0.0;
			duty_cycles_[1] = 0.0;
		}

		if (1.0 < std::abs(duty_cycles_[0]) || 1.0 < std::abs(duty_cycles_[1])) {
			RCLCPP_FATAL(this->get_logger(), "Duty cycles should be between [-1, 1]");
			exit(1);
		}

		// [0] corresponds to left wheel, [1] corresponds to right wheel
		std::vector<double> wheel_angular_velocities(2, 0.0);
		std::vector<int>    abs_encoders(2, 0);
		std::vector<int>    diff_encoders(2, 0);

		std::vector<int> pwm{static_cast<int>(255 * duty_cycles_[0]),
		                     static_cast<int>(255 * duty_cycles_[1])};

		kobuki_motors_->update(pwm, wheel_angular_velocities, abs_encoders, diff_encoders);

		// publish encoders
		encoders_msg_.header.stamp        = this->now();
		encoders_msg_.encoder_left        = abs_encoders[0];
		encoders_msg_.encoder_right       = abs_encoders[1];
		encoders_msg_.delta_encoder_left  = diff_encoders[0];
		encoders_msg_.delta_encoder_right = diff_encoders[1];

		encoders_pub_->publish(encoders_msg_);

		// calculate kinematics and send twist to robot simulation node
		geometry_msgs::msg::Twist twist_msg;

		twist_msg.linear.x =
		    (wheel_angular_velocities[1] + wheel_angular_velocities[0]) * wheel_radius_ / 2.0;
		twist_msg.angular.z = (wheel_angular_velocities[1] - wheel_angular_velocities[0]) *
		                      wheel_radius_ / base_;

		twist_pub_->publish(twist_msg);
	}

 private:
	rclcpp::TimerBase::SharedPtr timer_;

	std::unique_ptr<KobukiMotors> kobuki_motors_;

	rclcpp::Publisher<robp_interfaces::msg::Encoders>::SharedPtr encoders_pub_;
	rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr      twist_pub_;

	rclcpp::Subscription<robp_interfaces::msg::DutyCycles>::SharedPtr duty_cycles_sub_;

	// [0] corresponds to left wheel, [1] corresponds to right wheel
	std::array<double, 2> duty_cycles_{};
	rclcpp::Time          duty_cyles_time_;

	robp_interfaces::msg::Encoders encoders_msg_;

	double wheel_radius_{0.0352};
	double base_{0.23};
};

int main(int argc, char *argv[])
{
	rclcpp::init(argc, argv);
	rclcpp::spin(std::make_shared<Motors>());
	rclcpp::shutdown();
	return 0;
}