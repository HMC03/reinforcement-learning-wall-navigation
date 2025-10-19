import sys
import csv
import os
import numpy as np
import matplotlib.pyplot as plt

def read_rewards(filename):
    episodes, rewards = [], []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                try:
                    episodes.append(int(row[0]))
                    rewards.append(float(row[1]))
                except ValueError:
                    continue
    return np.array(episodes), np.array(rewards)

def moving_average(data, window_size):
    if window_size <= 1:
        return data
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def main():
    # ---- Handle CLI args ----
    mode = "raw"
    window_size = 50  # default moving average window

    if len(sys.argv) >= 2:
        mode = sys.argv[1].lower()
    if len(sys.argv) >= 3:
        try:
            window_size = int(sys.argv[2])
        except ValueError:
            print("Invalid window size argument; using default 50")

    # ---- Find all CSV files ----
    csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
    if not csv_files:
        print("No CSV files found in current directory.")
        sys.exit(0)

    # ---- Plot ----
    plt.figure(figsize=(10, 6))
    plt.title(f"Training Rewards ({mode.capitalize()})")

    for csv_file in csv_files:
        episodes, rewards = read_rewards(csv_file)
        if len(rewards) == 0:
            continue

        if mode == "average":
            averaged_rewards = moving_average(rewards, window_size)
            averaged_episodes = episodes[:len(averaged_rewards)]
            plt.plot(
                averaged_episodes,
                averaged_rewards,
                label=f"{csv_file} (avg {window_size})"
            )
        else:
            plt.plot(episodes, rewards, label=csv_file)

    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
