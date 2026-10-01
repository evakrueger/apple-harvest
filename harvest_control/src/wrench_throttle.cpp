// Republishes a WrenchStamped topic at a lower rate, for the harvest GUI's force plot.
//
// The UR driver publishes the force/torque wrench at ~500 Hz. Handling that in Python costs about half a CPU core
// per subscriber; this does the thinning in C++ so the GUI only gets `rate` Hz. Bags still record the full-rate topic.
//
// Parameters: input_topic, output_topic, rate (Hz). Keeps the newest message and publishes it at most `rate` times a
// second (messages arriving sooner after the last one sent are dropped).

#include <algorithm>
#include <chrono>
#include <memory>
#include <string>

#include <geometry_msgs/msg/wrench_stamped.hpp>
#include <rclcpp/rclcpp.hpp>

class WrenchThrottle : public rclcpp::Node
{
public:
    WrenchThrottle() : Node("wrench_throttle")
    {
        const auto input = this->declare_parameter<std::string>("input_topic", "/force_torque_sensor_broadcaster/wrench");
        const auto output = this->declare_parameter<std::string>("output_topic",
                                                                 "/force_torque_sensor_broadcaster/wrench_throttled");
        const double rate = this->declare_parameter<double>("rate", 50.0);
        min_period_ = rclcpp::Duration::from_seconds(1.0 / std::max(rate, 0.1));

        publisher_ = this->create_publisher<geometry_msgs::msg::WrenchStamped>(output, rclcpp::SensorDataQoS());
        subscription_ = this->create_subscription<geometry_msgs::msg::WrenchStamped>(
            input, rclcpp::SensorDataQoS(),
            [this](geometry_msgs::msg::WrenchStamped::ConstSharedPtr msg)
            {
                const auto now = this->now();
                if (last_sent_.nanoseconds() == 0 || now - last_sent_ >= min_period_)
                {
                    publisher_->publish(*msg);
                    last_sent_ = now;
                }
            });
        RCLCPP_INFO(this->get_logger(), "Throttling %s to %.0f Hz on %s", input.c_str(), rate, output.c_str());
    }

private:
    rclcpp::Publisher<geometry_msgs::msg::WrenchStamped>::SharedPtr publisher_;
    rclcpp::Subscription<geometry_msgs::msg::WrenchStamped>::SharedPtr subscription_;
    rclcpp::Duration min_period_{0, 0};
    rclcpp::Time last_sent_{0, 0, RCL_ROS_TIME};
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<WrenchThrottle>());
    rclcpp::shutdown();
    return 0;
}
