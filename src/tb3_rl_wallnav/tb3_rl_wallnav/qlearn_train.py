#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import TwistStamped
import numpy as np
import itertools

class QLearnTrainNode(Node):
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

        # Define Actions
        self.actions = {
            0: "forward",
            1: "forward_left",
            2: "forward_right",
            3: "rotate_left",
            4: "rotate_right",
        }

        # Define q_table
        self.q_table = {}

        for state in itertools.product([0, 1, 2, 3], repeat=3):
            front, front_left, rear_left = state
            action = 0  # Default = forward

            # --- Case 1: Dead-end / U-turn needed ---
            if front == 0 and front_left == 0 and rear_left == 0:
                action = 4  # rotate right

            # --- Case 2: Wall directly ahead ---
            elif front == 0:
                if front_left > 1:     # open to left
                    action = 3         # rotate left
                else:
                    action = 4         # rotate right

            # --- Case 3: Too close to wall (any left sensors short) ---
            elif front_left == 0 or rear_left == 0:
                action = 2  # forward right (veer away)

            # --- Case 4: Wall too far (both left sensors far) ---
            elif front_left >= 2 and rear_left >= 2:
                action = 1  # forward left (hug wall)

            # --- Case 5: Angled toward wall ---
            elif front_left == 0 and rear_left > 1:
                action = 2  # forward right

            # --- Case 6: Angled away from wall ---
            elif front_left > 1 and rear_left == 0:
                action = 1  # forward left

            # --- Case 7: Approaching a turn (front far, front_left close) ---
            elif front > 1 and front_left == 0:
                action = 1  # curve left (anticipate turn)

            # --- Case 8: Default straight path ---
            else:
                action = 0  # forward

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
            "front_left":  min10_in_range(40, 90),
            "rear_left":   min10_in_range(90, 140),
        }

        return lidar_segments

    def get_state(self, segments):
        """Convert segment distances into a discrete state tuple."""
        def categorize(dist):
            if dist < 0.4:
                return 0  # Close
            elif dist < 0.6:
                return 1  # Medium
            elif dist < 1:
                return 2  # Far
            else:
                return 3 # Very Far

        return (
            categorize(segments['front']),
            categorize(segments['front_left']),
            categorize(segments['rear_left'])
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

        # 4. Map discrete actions to motion commands
        cmd = TwistStamped()
        fwd_speed = 0.1
        turn_speed = 0.25
        curve_turn = 0.15

        if action == 0:          # Forward
            cmd.twist.linear.x = fwd_speed
            cmd.twist.angular.z = 0.0
        elif action == 1:        # Forward Left
            cmd.twist.linear.x = fwd_speed
            cmd.twist.angular.z = curve_turn
        elif action == 2:        # Forward Right
            cmd.twist.linear.x = fwd_speed
            cmd.twist.angular.z = -curve_turn
        elif action == 3:        # Rotate Left
            cmd.twist.linear.x = 0.0
            cmd.twist.angular.z = turn_speed
        elif action == 4:        # Rotate Right
            cmd.twist.linear.x = 0.0
            cmd.twist.angular.z = -turn_speed

        # 5. Publish the command
        self.cmd_vel_pub.publish(cmd)

        # 6. Log for debugging
        self.get_logger().info(f"State: {state} -> Action: {self.actions[action]}")

def main(args=None):
    rclpy.init(args=args)
    node = QLearnTrainNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
