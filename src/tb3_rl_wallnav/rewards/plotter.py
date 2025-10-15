import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('<path_to_workspace>/install/tb3_rl_wallnav/share/tb3_rl_wallnav/rewards/qlearn_rewards.csv')
plt.plot(data['Episode'], data['Average_Reward'])
plt.xlabel('Episode')
plt.ylabel('Average Reward')
plt.title('Q-Learning Average Reward per Episode')
plt.show()