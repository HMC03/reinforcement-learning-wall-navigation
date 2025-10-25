# tb3_rl_wallnav Package

## Overview
The `tb3_rl_wallnav` package is a ROS2 package implementing reinforcement learning (Q-learning and SARSA) and a manual Q-table policy for TurtleBot3 to navigate a maze by following the left wall in a Gazebo simulation. It includes scripts for training, debugging, visualization, and analysis, using a custom large maze environment.

**License**: MIT

## Dependencies
- **ROS2 Jazzy**: Install via [official instructions](https://docs.ros.org/en/jazzy/Installation.html).
- **Gazebo**: Version 11 or later (via `ros-jazzy-gazebo-ros`).
- **Python Libraries**: `numpy`, `matplotlib`, `colorama`, handled via:
  ```bash
  rosdep install --from-paths src --ignore-src -r -y
  ```
- **ROS Packages**: `rclpy`, `gazebo_ros`, `turtlebot3_gazebo`, `sensor_msgs`, `geometry_msgs`, `nav_msgs`, `ros_gz_bridge`, `robot_state_publisher`, `rviz2`, `ros_gz_interfaces`, `ament_index_python`, etc. (see `package.xml`).

## Directory Structure
- `launch/`: Launch files for Gazebo and RViz simulations.
- `media/`: Images and videos for environment, results, and demos (see [workspace README](../../README.md)).
- `qtables/`: Q-tables (binary and text) and conversion script (see `qtables/README.md`).
- `rewards/`: Reward CSVs and plotting script (see `rewards/README.md`).
- `rviz/`: RViz configuration for visualization.
- `worlds/`: Custom Gazebo world file (`largemaze.world`).
- `tb3_rl_wallnav/`: Python scripts for control and debugging.

## Scripts
The `tb3_rl_wallnav/` directory contains:
- **`lidar_debug.py`**: Prints color-coded LIDAR distances (front, front_left, etc.) for debugging.
  ```bash
  ros2 run tb3_rl_wallnav lidar_debug
  ```
- **`manual_qtable.py`**: Runs a predefined Q-table policy for wall-following.
  ```bash
  ros2 run tb3_rl_wallnav manual_qtable
  ```
- **`qlearn.py`**: Trains or runs a Q-learning agent. Saves Q-tables to `qtables/` and rewards to `rewards/`.
  - Train: `ros2 run tb3_rl_wallnav qlearn --ros-args -p mode:=train -p epsilon:=1.0`
  - Run: `ros2 run tb3_rl_wallnav qlearn --ros-args -p mode:=run`
- **`sarsa.py`**: Similar to `qlearn.py` but uses SARSA (on-policy RL).
  - Train: `ros2 run tb3_rl_wallnav sarsa --ros-args -p mode:=train -p epsilon:=1.0`
  - Run: `ros2 run tb3_rl_wallnav sarsa --ros-args -p mode:=run`

Training resets the robot to one of four base locations (cycling deterministically, with ±0.1m x/y and ±25° yaw noise) upon termination (collision, lost state, or max steps).

## Launch Files
- **`gz_tb3_largemaze.launch.py`**: Launches Gazebo with the `largemaze.world` environment, spawns TurtleBot3 Burger at (-2.0, -0.5), publishes robot state and TF, and bridges the SetEntityPose service for resets.
  ```bash
  ros2 launch tb3_rl_wallnav gz_tb3_largemaze.launch.py
  ```
- **`gz_rviz_tb3_largemaze.launch.py`**: Launches Gazebo as above and adds RViz with `tb3_gazebo.rviz` for visualization.
  ```bash
  ros2 launch tb3_rl_wallnav gz_rviz_tb3_largemaze.launch.py
  ```

## Media
Visuals (e.g., `turtlebot3_burger.png`, `lidar_state_space.png`, demo videos) are in `media/`. See [workspace README](../../README.md) or `media/README.md` for details.

## Analysis Tools
- **Rewards**: In `rewards/`, use `plot_rewards.py` to visualize training progress from CSV files.
  ```bash
  python3 plot_rewards.py average 10  # Moving average, window=10
  ```
- **Q-Tables**: In `qtables/`, use `convert_qtable_to_txt.py` for readable Q-tables.
  ```bash
  python3 convert_qtable_to_txt.py qlearn_qtable.npy qlearn_qtable.txt
  ```

## Configuration
- **`rviz/tb3_gazebo.rviz`**: Configures RViz with displays for Grid, TF, LaserScan, Odometry, and RobotModel. Uses an Orbit view camera for TurtleBot3 visualization in Gazebo.
- **`worlds/largemaze.world`**: Custom large maze environment for Gazebo simulations, designed for wall-following navigation training.