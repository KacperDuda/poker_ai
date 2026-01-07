import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_learning_curve(csv_path="docs/training_log.csv", save_path="docs/learning_curve.png"):
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)

    plt.figure(figsize=(10, 5))
    
    # Plot Win Rate
    plt.subplot(1, 2, 1)
    plt.plot(df['Episode'], df['WinRate'], label='Win Rate (%)', color='green')
    plt.xlabel('Episode')
    plt.ylabel('Win Rate (%)')
    plt.title('Win Rate over Time')
    plt.grid(True)
    
    # Plot Avg Reward
    plt.subplot(1, 2, 2)
    plt.plot(df['Episode'], df['AvgReward'], label='Avg Reward', color='blue')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title('Reward over Time')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved plot to {save_path}")

if __name__ == "__main__":
    plot_learning_curve()
