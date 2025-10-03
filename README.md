# Reinforcement Learning Wall Navigation
This repository demonstrates a TurtleBot3 robot learning to follow walls using reinforcement learning. It highlights autonomous navigation, Python programming, ROS2 Jazzy, Gazebo simulation, and LiDAR-based sensing.

## Features
* Wall-following capabilities – Maintain a desired distance from walls and avoid collisions.
* Reinforcement learning algorithms – Q-Learning and SARSA implementations.
* Customizable reward functions – Reward both distance traveled and accuracy of wall-following.
* Simulation-ready – Works with TurtleBot3 in Gazebo using ROS2 Jazzy.
* Modular workspace – Built with colcon for easy ROS2 development.

## Installation
1. Install ROS2 Jazzy

    Follow official instructions: [ROS2 Jazzy Installation](https://docs.ros.org/en/jazzy/Installation.html#)

2. Create a ROS2 Workspace
    ```bash
    mkdir -p ~/ros2_ws/src
    cd ~/ros2_ws/
    ```

3. Clone Repository
    ```bash
    git clone --recurse-submodules <your-repo-url>
    cd src
    git submodule update --init --recursive
    ```

4. Build Workspace
    ```bash
    cd ~/ros2_ws/
    colcon build --symlink-install
    source install/setup.bash
    ```

## Running the Simulation

Launch wall-following simulation:
```bash
ros2 launch wall_following_ros2 wall_following.launch
```

Run reinforcement learning scripts:
```bash
ros2 run wall_following_ros2 <script_name>.py
```

Available scripts:
* q_td_train.py – Q-Learning training
* q_td_run.py – Q-Learning testing
* sarsa_train.py – SARSA training
* sarsa_run.py – SARSA testing

📈 Results (To Be Added)

Learning curves, reward plots, and simulation videos will be added as available.