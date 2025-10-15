#!/usr/bin/env python3
import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import TwistStamped
from ros_gz_interfaces.srv import SetEntityPose
from std_msgs.msg import Header
import numpy as np
import itertools
import random
import csv
import time

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

        # Service client for resetting robot pose
        self.set_pose_client = self.create_client(SetEntityPose, '/world/default/set_pose')

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

        # Action parameters
        self.target_dist = 0.5
        self.collision_threshold = 0.2
        self.fwd_speed = 0.1
        self.turn_speed = 0.2
        self.prev_state = None
        self.prev_action = None

        # Episodes
        self.max_episodes = 500
        self.max_steps = 1000
        self.episode = 0
        self.episode_steps = 0
        self.episode_reward = 0.0
        self.lost_count = 0
        self.avg_rewards = []

        # Declare ROS parameters
        self.declare_parameter('alpha', 0.1)  # Learning rate
        self.declare_parameter('gamma', 0.99)  # Discount factor
        self.declare_parameter('epsilon', 0.1)  # Exploration rate
        self.declare_parameter('mode', 'train')  # Mode: train or run

        # Get parameter values
        self.alpha = self.get_parameter('alpha').get_parameter_value().double_value
        self.gamma = self.get_parameter('gamma').get_parameter_value().double_value
        self.epsilon = self.get_parameter('epsilon').get_parameter_value().double_value
        self.mode = self.get_parameter('mode').get_parameter_value().string_value

        # Log parameters for debugging
        self.get_logger().info(f"Parameters: alpha={self.alpha}, gamma={self.gamma}, epsilon={self.epsilon}, mode={self.mode}")

        # Define q_table
        self.q_table = {}
        for state in itertools.product([0, 1, 2, 3], repeat=3):
            self.q_table[state] = np.zeros(5)  # Initialize Q-values to zeros for all actions
        self.get_logger().info(f"Q-table size: {len(self.q_table)}, Sample: {self.q_table[(0,0,0)]}")

        # Create directories in package source directory
        script_path = os.path.abspath(__file__)  # Path to qlearn.py
        self.get_logger().info(f"Script path: {script_path}")  # Debug script location
        package_root = os.path.dirname(script_path)  # Start with tb3_rl_wallnav/tb3_rl_wallnav
        if 'build' in package_root or 'install' in package_root:
            # Navigate to src/tb3_rl_wallnav from build or install
            package_root = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(package_root))), 'src', 'tb3_rl_wallnav')
        self.qtable_dir = os.path.join(package_root, 'qtables')
        self.reward_dir = os.path.join(package_root, 'rewards')
        os.makedirs(self.qtable_dir, exist_ok=True)
        os.makedirs(self.reward_dir, exist_ok=True)

        # Log paths for debugging
        self.get_logger().info(f"Q-table dir: {self.qtable_dir}")
        self.get_logger().info(f"Reward dir: {self.reward_dir}")

        # Load Q-table if it exists
        self.q_table_file = os.path.join(self.qtable_dir, 'qlearn_qtable.npy')
        if os.path.exists(self.q_table_file):
            loaded = np.load(self.q_table_file, allow_pickle=True).item()
            self.q_table.update(loaded)
            self.get_logger().info(f"Loaded Q-table from {self.q_table_file}")
        else:
            self.get_logger().info(f"Q-table file {self.q_table_file} not found, starting with empty Q-table")
        self.get_logger().info(f"Q-table size: {len(self.q_table)}, Sample: {self.q_table[(0,0,0)]}")

        # Load last episode from rewards CSV
        self.reward_file = os.path.join(self.reward_dir, 'qlearn_rewards.csv')
        if os.path.exists(self.reward_file):
            with open(self.reward_file, 'r') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                episodes = [int(row[0]) for row in reader if row]
                if episodes:
                    self.episode = max(episodes) + 1
                    self.get_logger().info(f"Resuming from episode {self.episode}")
        else:
            self.get_logger().info(f"Reward file {self.reward_file} not found, starting from episode 0")

        # Initialize CSV file for reward logging (only write header if new)
        if not os.path.exists(self.reward_file):
            with open(self.reward_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Episode', 'Average_Reward'])

        # Reset Locations
        self.last_reset = 0
        self.reset_locations = [
            {'x_pose': -2.0, 'y_pose': -0.5},
            {'x_pose': 1.0, 'y_pose': 1.5}
        ]

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

    def get_reward(self, segments, prev_action):
        """Compute reward based on LIDAR segments and previous action for left-wall following."""
        # Extract segment values
        front_dist = segments['front']
        front_left_dist = segments['front_left']
        rear_left_dist = segments['rear_left']
        
        # Initialize reward
        reward = 0

        # --- Basic Rewards ---
        if prev_action < 3: # Moving forward = Good
            reward += 2
            if prev_action == 0: # Pure forward = Better
                reward += 1
        if front_dist > 0.6: # Open space in front = Good
            reward += 5
        if 0.4 < front_left_dist < 0.6: # Front left is at ideal distance = Good
            reward += 5
        if 0.4 < rear_left_dist < 0.6: # Rear left is at ideal distance = Good
            reward += 5
        if abs(front_left_dist - rear_left_dist) < 0.2:  # Parallel to wall = Good
            reward += 2
        
        # --- Basic Penalties ---
        if front_dist < 0.4: # Near front collision = Bad
            reward -= 2
        if front_left_dist > 1: # Front left is lost = Bad
            reward -= 1
        if rear_left_dist > 1: # Rear left is lost = Bad
            reward -= 1
        if front_left_dist < 0.2: # Near front left collision = Bad
            reward -= 1
        if rear_left_dist < 0.2: # Near rear left collision = Bad
            reward -= 1

        # --- Continuous Adjustments ---
        # Fine-tune wall distance (target: 0.5m)
        front_left_error = abs(front_left_dist - 0.5)
        rear_left_error = abs(rear_left_dist - 0.5)
        reward -= 2.0 * front_left_error  # Penalize deviation from 0.5m
        reward -= 2.0 * rear_left_error   # Penalize deviation from 0.5m

        # Encourage parallel alignment
        parallel_error = abs(front_left_dist - rear_left_dist)
        reward -= 2.0 * parallel_error  # Penalize misalignment

        # --- Action-specific adjustments ---
        if prev_action == 1:  # Forward-left
            if front_left_dist > 0.6:  # Moving toward wall when too far = Good
                reward += 1.0
            elif front_left_dist < 0.4:  # Moving toward wall when too close = Bad
                reward -= 1.0
        if prev_action == 2:  # Forward-right
            if front_left_dist < 0.4:  # Moving away from wall when too close = Good
                reward += 1.0
            elif front_left_dist > 0.6:  # Moving away from wall when too far = Bad
                reward -= 1.0
        if prev_action in [3, 4]:  # Rotate left/right
            if parallel_error > 0.5:  # Correcting misalignment = Good
                reward += 1.0

        return reward
    
    def is_terminal_state(self, segments, state):
        """Check if the current state is terminal."""
        # Collision check
        if segments['front'] < self.collision_threshold:
            self.get_logger().info(f"Collision Detected! Episode {self.episode} over")
            return True
        
        # Lost state (3,3,3) check
        if state == (3, 3, 3):
            self.lost_count += 1
            if self.lost_count >= 10:
                self.get_logger().info(f"Lost for 10 steps! Episode {self.episode} over")
                return True
        else:
            self.lost_count = 0  # Reset if not in lost state

        # Step limit check
        if self.episode_steps >= self.max_steps:
            self.get_logger().info(f"Reached max steps ({self.max_steps})! Episode {self.episode} over")
            return True

        return False

    def reset_environment(self):
        """Reset the robot to one of the two spawn locations."""
        if not self.set_pose_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("SetEntityPose service not available")
            return

        # Alternate between spawn locations
        self.last_reset = (self.last_reset + 1) % 2
        location = self.reset_locations[self.last_reset]

        # Prepare request
        request = SetEntityPose.Request()
        request.entity.name = 'burger'
        request.entity.type = 2
        request.pose.position.x = location['x_pose']
        request.pose.position.y = location['y_pose']
        request.pose.position.z = 0.0
        request.pose.orientation.w = 1.0  # Default orientation (no rotation)

        # Send request
        self.set_pose_client.call_async(request)
        self.get_logger().info(f"Reset robot to ({location['x_pose']}, {location['y_pose']})")
        time.sleep(0.5)

        # Stop robot motion immediately
        stop_twist = TwistStamped()
        stop_twist.header = Header(stamp=self.get_clock().now().to_msg())
        stop_twist.twist.linear.x = 0.0
        stop_twist.twist.angular.z = 0.0
        self.cmd_vel_pub.publish(stop_twist)
        self.get_logger().info("Published stop command")

        # Short delay to allow physics to stabilize
        time.sleep(0.5)
        self.get_logger().info("Stabilization delay complete")

        # Reset episode variables
        self.episode_steps = 0
        self.episode_reward = 0.0
        self.lost_count = 0
        self.prev_state = None
        self.prev_action = None
        self.lidar_ranges = None


    def save_q_table(self):
        """Save the Q-table to a file."""
        np.save(self.q_table_file, self.q_table)
        self.get_logger().info(f"Saved Q-table to {self.q_table_file}")

    def control_loop(self):
        """Called periodically to decide motion using the Q-table."""
        if self.episode_steps == 0:
            self.get_logger().info(f"Starting Episode {self.episode} after reset.")

        if self.lidar_ranges is None:
            return  # Wait until we have data

        # 0. Check if training is complete
        if self.episode >= self.max_episodes:
            self.get_logger().info(f"Completed {self.max_episodes} episodes. Stopping training.")
            self.save_q_table()
            self.destroy_node()
            return
        
        # 1. Get LIDAR segments
        segments = self.get_lidar_segments()
        if segments is None:
            return

        # 2. Convert segments to discrete state
        state = self.get_state(segments)

        # 3. Select action (epsilon greedy)
        if state in self.q_table:
            if self.mode == 'train' and random.random() < self.epsilon:
                action = random.randint(0, 4)  # Random action (0-4)
            else:
                action = np.argmax(self.q_table[state])  # Best action
        else:
            action = 0  # Default to forward
            self.get_logger().warn(f"State {state} not in Q-table")

        # 4. Execute action
        cmd = TwistStamped()
        cmd.header = Header(stamp=self.get_clock().now().to_msg())
        if action == 0:          # Forward
            cmd.twist.linear.x = self.fwd_speed
            cmd.twist.angular.z = 0.0
        elif action == 1:        # Forward Left
            cmd.twist.linear.x = self.fwd_speed
            cmd.twist.angular.z = self.turn_speed
        elif action == 2:        # Forward Right
            cmd.twist.linear.x = self.fwd_speed
            cmd.twist.angular.z = -self.turn_speed
        elif action == 3:        # Rotate Left
            cmd.twist.linear.x = 0.0
            cmd.twist.angular.z = self.turn_speed
        elif action == 4:        # Rotate Right
            cmd.twist.linear.x = 0.0
            cmd.twist.angular.z = -self.turn_speed
        
        self.cmd_vel_pub.publish(cmd)
        self.get_logger().info(f"Episode {self.episode}, Step {self.episode_steps}: State {state}, Action {self.actions[action]}")

        # 5. Update Q-table (in train mode)
        if self.mode == 'train':
            # Compute reward based on current segments and previous action
            if self.prev_action is not None:
                reward = self.get_reward(segments, self.prev_action)
                self.episode_reward += reward
            
                # Update Q-table if we have a previous state/action
                if self.prev_state is not None:
                    next_state = state
                    q_old = self.q_table[self.prev_state][self.prev_action]
                    q_max = np.max(self.q_table[next_state])
                    self.q_table[self.prev_state][self.prev_action] = q_old + self.alpha * (reward + self.gamma * q_max - q_old)
                    self.get_logger().info(f"Q-update: State {self.prev_state}, Action {self.actions[self.prev_action]}, Reward {reward}, New Q {self.q_table[self.prev_state][self.prev_action]}")
            
        # Store current state/action for next iteration
        self.prev_state = state
        self.prev_action = action
        self.episode_steps += 1
        
        # Check for terminal state & save average reward
        if self.mode == 'train' and self.is_terminal_state(segments, state):
            avg_reward = self.episode_reward / max(self.episode_steps, 1)
            self.get_logger().info(f"Episode {self.episode} ended. Average Reward: {avg_reward:.2f}")
            self.avg_rewards.append([self.episode, avg_reward])
            with open(self.reward_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([self.episode, avg_reward])
            self.save_q_table()
            self.episode += 1
            self.reset_environment()

    def destroy_node(self):
            """Override to save Q-table on shutdown."""
            self.save_q_table()
            self.get_logger().info(f"Saved rewards to {self.reward_file}")
            super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = QLearnTrainNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down node")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
