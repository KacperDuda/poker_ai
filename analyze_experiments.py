import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

experiments = {
    'baseline': 'Baseline (4 players, hidden=512)',
    'long_decay': 'Long Decay (slower epsilon)',
    'bignet': 'Big Network (hidden=1024)',
    'heads_up': 'Heads-Up (2 players)'
}

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Training Results Comparison', fontsize=16, fontweight='bold')

colors = {'baseline': '#2E86AB', 'long_decay': '#A23B72', 'bignet': '#F18F01', 'heads_up': '#C73E1D'}

for idx, (exp_name, exp_label) in enumerate(experiments.items()):
    df = pd.read_csv(f'docs/experiments/log_{exp_name}.csv')
    
    window = 100
    df['AvgReward_smooth'] = df['AvgReward'].rolling(window=window, min_periods=1).mean()
    df['WinRate_smooth'] = df['WinRate'].rolling(window=window, min_periods=1).mean()

axes[0, 0].plot(df['Episode'], df['AvgReward_smooth'], label=exp_label, color=colors[exp_name], linewidth=2, alpha=0.8)
axes[0, 1].plot(df['Episode'], df['WinRate_smooth'], label=exp_label, color=colors[exp_name], linewidth=2, alpha=0.8)

axes[0, 0].set_xlabel('Episode', fontsize=12)
axes[0, 0].set_ylabel('Average Reward', fontsize=12)
axes[0, 0].set_title('Average Reward Over Time', fontsize=13, fontweight='bold')
axes[0, 0].legend(fontsize=9, loc='lower right')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].set_xlabel('Episode', fontsize=12)
axes[0, 1].set_ylabel('Win Rate (%)', fontsize=12)
axes[0, 1].set_title('Win Rate Over Time', fontsize=13, fontweight='bold')
axes[0, 1].legend(fontsize=9, loc='lower right')
axes[0, 1].grid(True, alpha=0.3)

final_stats = []
for exp_name, exp_label in experiments.items():
    df = pd.read_csv(f'docs/experiments/log_{exp_name}.csv')
    last_1000 = df.tail(1000)
    final_stats.append({
        'Experiment': exp_label,
        'Final Avg Reward': last_1000['AvgReward'].mean(),
        'Final Win Rate': last_1000['WinRate'].mean(),
        'Peak Win Rate': df['WinRate'].max()
    })

stats_df = pd.DataFrame(final_stats)

x = np.arange(len(experiments))
width = 0.35

axes[1, 0].bar(x - width/2, stats_df['Final Avg Reward'], width, label='Avg Reward', color='#2E86AB', alpha=0.8)
axes[1, 0].bar(x + width/2, stats_df['Final Win Rate'] * 10, width, label='Win Rate (×10)', color='#F18F01', alpha=0.8)
axes[1, 0].set_xlabel('Experiment', fontsize=12)
axes[1, 0].set_ylabel('Value', fontsize=12)
axes[1, 0].set_title('Final Performance (Last 1000 Episodes)', fontsize=13, fontweight='bold')
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels([exp.split('(')[0].strip() for exp in experiments.values()], rotation=15, ha='right', fontsize=9)
axes[1, 0].legend(fontsize=10)
axes[1, 0].grid(True, alpha=0.3, axis='y')

table_data = []
for _, row in stats_df.iterrows():
    table_data.append([
        row['Experiment'].split('(')[0].strip(),
        f"{row['Final Avg Reward']:.1f}",
        f"{row['Final Win Rate']:.1f}%",
        f"{row['Peak Win Rate']:.1f}%"
    ])

axes[1, 1].axis('tight')
axes[1, 1].axis('off')
table = axes[1, 1].table(cellText=table_data,
                          colLabels=['Experiment', 'Avg Reward', 'Win Rate', 'Peak WR'],
                          cellLoc='center',
                          loc='center',
                          colWidths=[0.35, 0.22, 0.22, 0.22])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2.5)

for i in range(len(table_data) + 1):
    for j in range(4):
        cell = table[(i, j)]
        if i == 0:
            cell.set_facecolor('#2E86AB')
            cell.set_text_props(weight='bold', color='white')
        else:
            cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')

axes[1, 1].set_title('Summary Statistics', fontsize=13, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('docs/experiments_comparison.png', dpi=300, bbox_inches='tight')
print("Comparison plot saved to docs/experiments_comparison.png")

print("\n=== EXPERIMENT RESULTS ===")
print(stats_df.to_string(index=False))
print("\n=== BEST CONFIGURATION ===")
best_idx = stats_df['Final Avg Reward'].idxmax()
print(f"Best by Avg Reward: {stats_df.loc[best_idx, 'Experiment']}")
best_wr_idx = stats_df['Final Win Rate'].idxmax()
print(f"Best by Win Rate: {stats_df.loc[best_wr_idx, 'Experiment']}")
