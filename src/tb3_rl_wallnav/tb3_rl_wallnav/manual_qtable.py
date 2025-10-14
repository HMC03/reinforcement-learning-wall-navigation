#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import numpy as np

class ManualQTableNode(Node):
    def __init__(self):
        super().__init__('manual_qtable_node')

        # Publisher for robot velocity
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Subscriber for LIDAR data
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        # Timer for control loop (10 Hz)
        self.timer = self.create_timer(0.1, self.control_loop)

        # Latest LIDAR data
        self.lidar_ranges = None
        self.lidar_angles = None

        self.get_logger().info('Manual Q-table controller started.')

    def scan_callback(self, msg: LaserScan):
        """Callback that receives LIDAR data and stores usable arrays."""
        # Convert to numpy array
        ranges = np.array(msg.ranges)

        # Replace NaN or inf with max range for safety
        self.lidar_ranges = np.nan_to_num(ranges, nan=msg.range_max, posinf=msg.range_max)

        # Compute the corresponding angle for each beam
        self.lidar_angles = np.rad2deg(msg.angle_min + np.arange(len(ranges)) * msg.angle_increment)

    def get_lidar_segments(self):
        """Compute 10th-percentile LIDAR distances for key angular segments."""
        if self.lidar_ranges is None:
            return None

        deg = self.lidar_angles
        ranges = self.lidar_ranges

        def percentile_in_range(low, high):
            """Return the 10th percentile distance in the given angular window."""
            # Handle wrap-around for rear segment (e.g., 160°–180° and -180°–-160°)
            if low < high:
                mask = (deg >= low) & (deg <= high)
            else:
                mask = (deg >= low) | (deg <= high)

            if not np.any(mask):
                return np.nan

            segment = ranges[mask]
            return np.percentile(segment, 10)  # “close but robust” distance

        # Define key directions (in degrees)
        lidar_segments = {
            "rear_left":   percentile_in_range(120, 160),
            "left":        percentile_in_range(60, 120),
            "front_left":  percentile_in_range(20, 60),
            "front":       percentile_in_range(-20, 20),
            "front_right": percentile_in_range(-60, -20),
            "right":       percentile_in_range(-120, -60),
            "rear_right":  percentile_in_range(-160, -120),
            "rear":        percentile_in_range(160, -160),  # handles wrap-around
        }

        return lidar_segments

    def control_loop(self):
        """Called periodically to decide motion."""
        if self.lidar_ranges is None:
            return  # Wait until we have data

        # --- Placeholder logic for now ---
        # We’ll later use a Q-table here to decide an action.
        # For now, let's just stop the robot.
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        self.cmd_vel_pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = ManualQTableNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
