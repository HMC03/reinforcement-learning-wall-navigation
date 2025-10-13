#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import numpy as np
import math

class LidarDebug(Node):
    def __init__(self):
        super().__init__('lidar_debug')
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )
        self.latest_scan = None
        self.get_logger().info('Lidar debug node started. Listening to /scan topic...')

        # Print every 0.5 seconds
        self.timer = self.create_timer(0.5, self.print_latest_scan)

    def scan_callback(self, msg):
        self.latest_scan = msg

    def get_avg_range(self, msg, angle_deg, window_deg=5):
        """Return average distance for a small angular window around a given direction."""
        angle_min = msg.angle_min
        angle_increment = msg.angle_increment
        ranges = np.array(msg.ranges)
        num_readings = len(ranges)

        # Convert degrees to radians
        angle_rad = math.radians(angle_deg)
        window_rad = math.radians(window_deg)

        # Convert angle range to index range
        start_idx = int((angle_rad - window_rad - angle_min) / angle_increment)
        end_idx = int((angle_rad + window_rad - angle_min) / angle_increment)

        start_idx = max(0, start_idx)
        end_idx = min(num_readings - 1, end_idx)

        segment = ranges[start_idx:end_idx]
        finite_values = segment[np.isfinite(segment)]

        if len(finite_values) == 0:
            return None
        return float(np.mean(finite_values))

    def format_distance(self, dist):
        """Color-code distances for quick interpretation."""
        if dist is None:
            return "  ---  "
        dist = round(dist, 2)
        if dist < 0.3:
            return f"\033[91m{dist:5.2f}\033[0m"   # Red (too close)
        elif dist < 0.6:
            return f"\033[93m{dist:5.2f}\033[0m"   # Yellow (medium)
        else:
            return f"\033[92m{dist:5.2f}\033[0m"   # Green (clear)

    def print_latest_scan(self):
        """Print lidar distance averages in 45° increments."""
        if self.latest_scan is None:
            return

        msg = self.latest_scan

        # Full-circle coverage in 45° increments
        # 0 = front, positive = right side, negative = left side
        angles = [-135, -90, -45, 0, 45, 90, 135, 180]
        readings = {a: self.get_avg_range(msg, a) for a in angles}

        print("\n--- LIDAR AVERAGE DISTANCES (m) ---")
        print(
            f"Rear Left: {self.format_distance(readings[-135])}  "
            f"Left: {self.format_distance(readings[-90])}  "
            f"Front Left: {self.format_distance(readings[-45])}  "
            f"Front: {self.format_distance(readings[0])}  "
            f"Front Right: {self.format_distance(readings[45])}  "
            f"Right: {self.format_distance(readings[90])}  "
            f"Rear Right: {self.format_distance(readings[135])}  "
            f"Rear: {self.format_distance(readings[180])}"
        )

def main(args=None):
    rclpy.init(args=args)
    node = LidarDebug()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
