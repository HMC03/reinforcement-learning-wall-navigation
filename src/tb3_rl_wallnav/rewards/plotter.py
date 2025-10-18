import pandas as pd
import matplotlib.pyplot as plt
import os

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Find all CSV files in the same directory
csv_files = [f for f in os.listdir(script_dir) if f.endswith('.csv')]

# Check if any CSV files were found
if not csv_files:
    print("No CSV files found in directory:", script_dir)
    exit()

plt.figure(figsize=(10, 6))

# Plot each CSV file
for file in csv_files:
    path = os.path.join(script_dir, file)
    try:
        data = pd.read_csv(path)
        if 'Episode' in data.columns and 'Total_Reward' in data.columns:
            plt.plot(data['Episode'], data['Total_Reward'], label=file[:-4])
        else:
            print(f"Skipping {file} (missing 'Episode' or 'Total_Reward' columns)")
    except Exception as e:
        print(f"Error reading {file}: {e}")

# Label the axes and title
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Machine Learning Total Reward per Episode')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()