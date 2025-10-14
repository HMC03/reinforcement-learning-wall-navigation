#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import numpy as np
import time
from colorama import Fore, Style, init

init(autoreset=True)

class LidarDebug(Node):
    def __init__(self):
        super().__init__('lidar_debug_node')
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10)
        self.last_print_time = 0.0
        self.get_logger().info("LIDAR Debug Node Started. Listening to /scan topic...")

    def scan_callback(self, msg: LaserScan):
        now = time.time()
        # limit updates to 0.5 Hz
        if now - self.last_print_time < 2.0:
            return
        self.last_print_time = now

        ranges = np.array(msg.ranges)
        ranges = np.where(np.isinf(ranges), np.nan, ranges)
        angle_increment_deg = np.degrees(msg.angle_increment)
        num_readings = len(ranges)

        def avg_distance(center_deg, window=5):
            center_deg %= 360
            start_deg = (center_deg - window) % 360
            end_deg = (center_deg + window) % 360
            if start_deg < end_deg:
                indices = np.arange(int(start_deg / angle_increment_deg),
                                    int(end_deg / angle_increment_deg))
            else:
                indices = np.concatenate([
                    np.arange(0, int(end_deg / angle_increment_deg)),
                    np.arange(int(start_deg / angle_increment_deg), num_readings)
                ])
            values = ranges[indices]
            finite_values = values[np.isfinite(values)]
            return np.nanmean(finite_values) if len(finite_values) > 0 else np.nan

        directions = {
            "Front": 0,
            "Front Left": 45,
            "Left": 90,
            "Rear Left": 135,
            "Rear": 180,
            "Rear Right": 225,
            "Right": 270,
            "Front Right": 315
        }

        distances = {name: avg_distance(angle) for name, angle in directions.items()}

        def colorize(value):
            if np.isnan(value):
                return f"{Fore.MAGENTA}---{Style.RESET_ALL}"
            if value < 0.6:
                color = Fore.RED
            elif value < 1.5:
                color = Fore.YELLOW
            else:
                color = Fore.GREEN
            return f"{color}{value:.2f}{Style.RESET_ALL}"

        print("\n--- LIDAR AVERAGE DISTANCES (m) ---")
        print(f"Front Left:  {colorize(distances['Front Left'])}   "
              f"Front:   {colorize(distances['Front'])}   "
              f"Front Right: {colorize(distances['Front Right'])}")
        print(f"Left:        {colorize(distances['Left'])}   "
              f"               "
              f"Right:       {colorize(distances['Right'])}")
        print(f"Rear Left:   {colorize(distances['Rear Left'])}   "
              f"Rear:   {colorize(distances['Rear'])}   "
              f"Rear Right:  {colorize(distances['Rear Right'])}")

def main(args=None):
    rclpy.init(args=args)
    node = LidarDebug()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()