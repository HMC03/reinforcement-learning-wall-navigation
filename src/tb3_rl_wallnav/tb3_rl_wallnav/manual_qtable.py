#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import TwistStamped
import numpy as np
import itertools

class ManualQTableNode(Node):
    def __init__(self):
        super().__init__('manual_qtable_node')

        # Publisher for robot velocity
        self.cmd_vel_pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)

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

        # Define q_table
        self.q_table = {}

        for state in itertools.product([0, 1, 2], repeat=4):
            rear_left, left, front_left, front = state

            # Default action
            action = 0  # forward

            # --- Front obstacle logic with context ---
            if front == 0:
                if left > 1:       # left is open
                    action = 1     # turn left
                else:              # left also blocked
                    action = 2     # turn right

            # --- Wall too close on left ---
            elif left == 0 or front_left == 0:
                action = 2

            # --- Wall too far on left ---
            elif left == 2 and front_left == 2:
                action = 1

            # Store in Q-table
            self.q_table[state] = action

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
        """Compute 10th percentile LIDAR distances for key 8-way segments."""
        if self.lidar_ranges is None:
            return None

        deg = np.mod(self.lidar_angles, 360)  # Wrap angles to [0, 360)
        ranges = self.lidar_ranges

        def min10_in_range(low, high):
            mask = (deg >= low) & (deg < high)
            if not np.any(mask):
                return 3.5
            return np.percentile(ranges[mask], 10)

        lidar_segments = {
            "front":       min(min10_in_range(0, 20), min10_in_range(340, 360)),
            "front_left":  min10_in_range(20, 60),
            "left":        min10_in_range(60, 120),
            "rear_left":   min10_in_range(120, 160),
            "rear":        min10_in_range(160, 200),
            "rear_right":  min10_in_range(200, 240),
            "right":       min10_in_range(240, 300),
            "front_right": min10_in_range(300, 340), 
        }

        return lidar_segments

    def get_state(self, segments):
        """Convert segment distances into a discrete state tuple."""
        def categorize(dist):
            if dist < 0.4:
                return 0  # Short
            elif dist > 0.6:
                return 2  # Long
            else:
                return 1  # Ideal

        return (
            categorize(segments['rear_left']),
            categorize(segments['left']),
            categorize(segments['front_left']),
            categorize(segments['front'])
        )
    
    def control_loop(self):
        """Called periodically to decide motion using the Q-table."""
        if self.lidar_ranges is None:
            return  # Wait until we have data

        # 1. Get LIDAR segments
        segments = self.get_lidar_segments()
        if segments is None:
            return

        # 2. Convert segments to discrete state
        state = self.get_state(segments)

        # 3. Look up action in Q-table
        action = self.q_table.get(state, 0)  # default to forward if state unknown

        # 4. Create Twist command based on action
        cmd = TwistStamped()
        if action == 0:          # Forward
            cmd.twist.linear.x = 0.1
            cmd.twist.angular.z = 0.0
        elif action == 1:        # Turn left
            cmd.twist.linear.x = 0.0
            cmd.twist.angular.z = 0.2
        elif action == 2:        # Turn right
            cmd.twist.linear.x = 0.0
            cmd.twist.angular.z = -0.2

        # 5. Publish the command
        self.cmd_vel_pub.publish(cmd)

        # 6. log for debugging
        self.get_logger().info(f"State: {state} -> Action: {action}")

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
