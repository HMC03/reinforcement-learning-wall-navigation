# Rewards Directory

## Overview
This directory contains data and tools for analyzing the performance of reinforcement learning algorithms used in the wall navigation project. It includes CSV files storing episode-wise reward data for Q-learning and SARSA algorithms, along with a Python script for visualizing this data.

## Contents
- `qlearn_rewards.csv`: Reward data for the Q-learning algorithm, with columns for episode number and total reward.
- `sarsa_rewards.csv`: Reward data for the SARSA algorithm, with columns for episode number and total reward.
- `plot_rewards.py`: Python script to generate plots of reward data, supporting raw and moving-average visualizations.

## Dependencies
To run the plotting script, ensure the following Python packages are installed:
- `numpy`
- `matplotlib`

Install them using:
```bash
sudo apt install python3-numpy python3-matplotlib
```

## Usage
The plot_rewards.py script visualizes reward data from CSV files in the directory. It supports two modes:

* Raw: Plots raw reward data.
* Average: Plots a moving average of rewards with a configurable window size.

## Command Examples

1. Plot raw rewards:
    ```bash
    python3 plot_rewards.py raw
    ```


2. Plot moving average with a window size of 50:
    ```bash
    python3 plot_rewards.py average 50
    ```



## Notes

* The script automatically detects all .csv files in the directory.
* If an invalid window size is provided for the average mode, it defaults to 50.
* The generated plot displays episode numbers on the x-axis and total rewards on the y-axis, with a legend identifying each CSV file.
