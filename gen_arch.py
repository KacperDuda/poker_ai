import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

layer_configs = [
    {'name': 'Input\n(State Vector)', 'dim': '134', 'y': 8, 'color': '#E8F4F8'},
    {'name': 'Hidden Layer 1\n(ReLU)', 'dim': '512', 'y': 6.5, 'color': '#B3D9E8'},
    {'name': 'Hidden Layer 2\n(ReLU)', 'dim': '512', 'y': 5, 'color': '#7FB8D4'},
    {'name': 'Hidden Layer 3\n(ReLU)', 'dim': '256', 'y': 3.5, 'color': '#4A90A4'},
]

def draw_layer(ax, name, dim, y, color, width=3.0, height=0.8):
    rect = patches.FancyBboxPatch((3.5, y - height/2), width, height,
                                   boxstyle="round,pad=0.05", 
                                   edgecolor='#2C3E50', facecolor=color, linewidth=2.5)
    ax.add_patch(rect)
    ax.text(5, y, name, ha='center', va='center', fontsize=16, fontweight='bold', color='#2C3E50')
    ax.text(7.2, y, f'dim={dim}', ha='left', va='center', fontsize=14, 
            style='italic', color='#34495E', bbox=dict(boxstyle='round,pad=0.3', 
            facecolor='white', edgecolor='#BDC3C7', linewidth=1.5))

for layer in layer_configs:
    draw_layer(ax, layer['name'], layer['dim'], layer['y'], layer['color'])

for i in range(len(layer_configs) - 1):
    y1 = layer_configs[i]['y'] - 0.4
    y2 = layer_configs[i + 1]['y'] + 0.4
    ax.annotate('', xy=(5, y2), xytext=(5, y1),
                arrowprops=dict(arrowstyle='->', lw=2.5, color='#34495E'))

action_y = 1.8
action_box = patches.FancyBboxPatch((2.2, action_y - 0.35), 2.2, 0.7,
                                     boxstyle="round,pad=0.05",
                                     edgecolor='#27AE60', facecolor='#ABEBC6', linewidth=2.5)
ax.add_patch(action_box)
ax.text(3.3, action_y, 'Action Head\nQ(s, a)', ha='center', va='center', 
        fontsize=15, fontweight='bold', color='#27AE60')
ax.text(3.3, action_y - 0.7, 'dim=3 (Fold/Call/Raise)', ha='center', va='top', 
        fontsize=13, style='italic', color='#1E8449')

sizing_box = patches.FancyBboxPatch((5.6, action_y - 0.35), 2.2, 0.7,
                                     boxstyle="round,pad=0.05",
                                     edgecolor='#E67E22', facecolor='#F8C471', linewidth=2.5)
ax.add_patch(sizing_box)
ax.text(6.7, action_y, 'Sizing Head\nσ(s)', ha='center', va='center', 
        fontsize=15, fontweight='bold', color='#D35400')
ax.text(6.7, action_y - 0.7, 'dim=1 (Bet Size)', ha='center', va='top', 
        fontsize=13, style='italic', color='#BA4A00')

ax.annotate('', xy=(3.3, action_y + 0.35), xytext=(4.5, 3.1),
            arrowprops=dict(arrowstyle='->', lw=2.5, color='#27AE60', 
                          connectionstyle="arc3,rad=0.3"))
ax.annotate('', xy=(6.7, action_y + 0.35), xytext=(5.5, 3.1),
            arrowprops=dict(arrowstyle='->', lw=2.5, color='#E67E22', 
                          connectionstyle="arc3,rad=-0.3"))

ax.text(5, 9.5, 'PokerNet Architecture', ha='center', va='center', 
        fontsize=22, fontweight='bold', color='#2C3E50',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#ECF0F1', 
                 edgecolor='#34495E', linewidth=2))

ax.text(0.5, 8, '134-dim features:\n• Hand strength\n• Position\n• Pot odds\n• Stack sizes\n• Betting history', 
        ha='left', va='top', fontsize=14, color='#566573',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#F8F9F9', 
                 edgecolor='#AEB6BF', linewidth=1.5))

plt.tight_layout()
plt.savefig('docs/model_architecture.png', dpi=300, bbox_inches='tight', facecolor='white')
print("Architecture diagram saved to docs/model_architecture.png (enhanced)")
