import subprocess
import sys

print("=" * 80)
print("🎰 POKER DQN TRAINING LAUNCHER 🎰".center(80))
print("=" * 80)
print("\n📝 Modifying train_rl.py for demo (500 episodes)...")

with open('train_rl.py', 'r') as f:
    content = f.read()

content = content.replace('num_episodes = 200000', 'num_episodes = 500')

content = content.replace(
    'print("--- START DQN TRAINING (FIXED) ---")',
    '''print("=" * 80)
    print("🎰 POKER DQN TRAINING 🎰".center(80))
    print("=" * 80)
    print(f"\\n🎯 Configuration:")
    print(f"   • Players: 3 (1 DQN + 2 Random)")
    print(f"   • Episodes: {num_episodes:,}")
    print(f"   • Learning Rate: {LEARNING_RATE}")
    print(f"   • Replay Buffer: {MEMORY_SIZE:,}")
    print(f"\\n🚀 Starting Training...")
    print("=" * 80)'''
)

content = content.replace(
    'print(f"Episode {episode}: Avg Reward = {avg:.2f}, Win Rate = {wr:.2f}%, eps = {eps_threshold:.4f}")',
    '''emoji = "🔥" if wr > 60 else "📈" if wr > 40 else "💪" if wr > 25 else "🎲"
            print(f"{emoji} Ep {episode:5d} | Avg: {avg:7.1f} | WR: {wr:5.1f}% | ε: {eps_threshold:.4f}")'''
)

content = content.replace(
    'print("Training finished. Model saved to poker_dqn.pth")',
    '''print("\\n" + "=" * 80)
    print("✅ TRAINING COMPLETE! ✅".center(80))
    print("=" * 80)
    print(f"\\n💾 Model saved: poker_dqn.pth")'''
)

with open('train_rl_demo.py', 'w') as f:
    f.write(content)

print("✅ Created train_rl_demo.py")
print("\n🚀 Launching training...")
print("=" * 80 + "\n")

subprocess.run([sys.executable, 'train_rl_demo.py'])
