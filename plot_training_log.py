import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_context("notebook", font_scale=1.2)

# Load data
log_file = "docs/training_log.csv"
if not os.path.exists(log_file):
    print(f"Error: {log_file} not found!")
    exit(1)

df = pd.read_csv(log_file)

# Create output directory if not exists
output_dir = "docs/plots"
os.makedirs(output_dir, exist_ok=True)

# 1. Win Rate & Epsilon (Dual Axis)
fig, ax1 = plt.subplots(figsize=(12, 6))

color = 'tab:blue'
ax1.set_xlabel('Episode')
ax1.set_ylabel('Win Rate (%)', color=color)
sns.lineplot(data=df, x='Episode', y='WinRate', ax=ax1, color=color, label='Win Rate')
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim(0, 100)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:orange'
ax2.set_ylabel('Epsilon (Exploration)', color=color)
sns.lineplot(data=df, x='Episode', y='Epsilon', ax=ax2, color=color, linestyle='--', label='Epsilon')
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim(0, 1.0)

plt.title('Win Rate vs Exploration Decay')
fig.tight_layout()
plt.savefig(f"{output_dir}/win_rate_epsilon.png")
print(f"Saved {output_dir}/win_rate_epsilon.png")
plt.close()

# 2. Average Profit (Moving Average for smoothness)
plt.figure(figsize=(12, 6))
df['AvgProfit_Smooth'] = df['AvgProfit'].rolling(window=10, min_periods=1).mean()
sns.lineplot(data=df, x='Episode', y='AvgProfit', alpha=0.3, color='gray', label='Raw')
sns.lineplot(data=df, x='Episode', y='AvgProfit_Smooth', color='green', linewidth=2, label='Smoothed (10-point MA)')
plt.axhline(0, color='red', linestyle='--', linewidth=1)
plt.title('Average Profit per 100 Episodes')
plt.ylabel('Average Profit ($)')
plt.legend()
plt.tight_layout()
plt.savefig(f"{output_dir}/average_profit.png")
print(f"Saved {output_dir}/average_profit.png")
plt.close()

# 3. Action Distribution (Stacked Area Chart)
plt.figure(figsize=(12, 6))
plt.stackplot(df['Episode'], 
              df['Fold%'], df['Call%'], df['Raise%'], 
              labels=['Fold', 'Call', 'Raise'],
              colors=['#e74c3c', '#f1c40f', '#2ecc71'], 
              alpha=0.8)
plt.legend(loc='upper left')
plt.title('Action Distribution Evolution')
plt.ylabel('Percentage (%)')
plt.xlabel('Episode')
plt.margins(0, 0)
plt.tight_layout()
plt.savefig(f"{output_dir}/action_distribution.png")
print(f"Saved {output_dir}/action_distribution.png")
plt.close()

print("\n✅ All plots generated successfully!")
