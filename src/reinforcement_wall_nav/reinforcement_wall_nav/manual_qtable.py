#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan

# -----------------------
# Config: Discretization
# -----------------------
DIST_THRESHOLDS = [0.3, 0.6, 1.0]  # meters
FRONT_ANGLE = (-10, 10)            # degrees
RIGHT_ANGLE = (-100, -80)
LEFT_ANGLE = (80, 100)

# Example actions
# 0 = forward, 1 = turn left, 2 = turn right, 3 = stop
ACTIONS = [0, 1, 2, 3]

# -----------------------
# Q-table placeholder
# -----------------------
# Fill this manually later: q_table[front_level, right_level, left_level] = action
# key = (front, right, left), value = action
q_table = {
    # Front far → move forward
    (2, 2, 2): 0, (2, 2, 1): 0, (2, 2, 0): 2,
    (2, 1, 2): 0, (2, 1, 1): 0, (2, 1, 0): 2,
    (2, 0, 2): 1, (2, 0, 1): 1, (2, 0, 0): 1,

    # Front medium → avoid obstacles
    (1, 2, 2): 0, (1, 2, 1): 0, (1, 2, 0): 2,
    (1, 1, 2): 0, (1, 1, 1): 1, (1, 1, 0): 2,
    (1, 0, 2): 1, (1, 0, 1): 1, (1, 0, 0): 1,

    # Front close → turn away
    (0, 2, 2): 1, (0, 2, 1): 1, (0, 2, 0): 1,
    (0, 1, 2): 1, (0, 1, 1): 1, (0, 1, 0): 1,
    (0, 0, 2): 1, (0, 0, 1): 1, (0, 0, 0): 1,
}

class TurtleBot3Node(Node):
    def __init__(self):
        super().__init__('turtlebot3_node')

        # Publisher
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Subscriber
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        # Timer
        self.timer = self.create_timer(0.1, self.control_loop)

        self.latest_scan = None
        self.get_logger().info("TurtleBot3 Q-learning test node initialized!")

    # -----------------------
    # Laser scan callback
    # -----------------------
    def scan_callback(self, msg: LaserScan):
        self.latest_scan = msg

    # -----------------------
    # Helper: angle → index
    # -----------------------
    def angle_to_index(self, angle_deg, scan: LaserScan):
        angle_rad = angle_deg * 3.14159265 / 180.0
        index = int((angle_rad - scan.angle_min) / scan.angle_increment)
        return max(0, min(index, len(scan.ranges)-1))

    # -----------------------
    # Helper: discretize distance
    # -----------------------
    def discretize_range(self, dist):
        for i, th in enumerate(DIST_THRESHOLDS):
            if dist < th:
                return i
        return len(DIST_THRESHOLDS) - 1  # always 0,1,2

    # -----------------------
    # Get discrete state
    # -----------------------
    def get_discrete_state(self):
        if self.latest_scan is None:
            return None

        scan = self.latest_scan

        def avg_distance(angle_range):
            start_idx = self.angle_to_index(angle_range[0], scan)
            end_idx = self.angle_to_index(angle_range[1], scan)
            values = [r for r in scan.ranges[start_idx:end_idx+1] if r > 0]
            return sum(values)/len(values) if values else float('inf')

        front_dist = avg_distance(FRONT_ANGLE)
        right_dist = avg_distance(RIGHT_ANGLE)
        left_dist = avg_distance(LEFT_ANGLE)

        return (self.discretize_range(front_dist),
                self.discretize_range(right_dist),
                self.discretize_range(left_dist))

    # -----------------------
    # Control loop
    # -----------------------
    def control_loop(self):
        state = self.get_discrete_state()
        if state is None:
            return

        # -----------------------
        # Q-table lookup
        # -----------------------
        action = q_table.get(state, 3)  # default to stop if state not defined

        twist = Twist()
        if action == 0:        # forward
            twist.linear.x = 0.15
            twist.angular.z = 0.0
        elif action == 1:      # turn left
            twist.linear.x = 0.0
            twist.angular.z = 0.5
        elif action == 2:      # turn right
            twist.linear.x = 0.0
            twist.angular.z = -0.5
        else:                  # stop
            twist.linear.x = 0.0
            twist.angular.z = 0.0

        self.cmd_pub.publish(twist)
        self.get_logger().info(f"State: {state} | Action: {action}")

# -----------------------
# Main
# -----------------------
def main(args=None):
    rclpy.init(args=args)
    node = TurtleBot3Node()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
