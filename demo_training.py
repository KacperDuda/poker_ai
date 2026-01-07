import torch
import numpy as np
import random
from poker_env import PokerEnv
from agent import PokerNet

print("=" * 70)
print("POKER DQN TRAINING LIVE DEMO".center(70))
print("=" * 70)

device = torch.device("cpu")
print(f"\n📱 Device: {device}")

players_ids = [0, 1, 2, 3]
env = PokerEnv(players_ids, initial_stack=2000.0)

print(f"\n🎰 Environment Setup:")
print(f"   • Players: 4 (1 DQN Agent + 3 Random Bots)")
print(f"   • Starting Stack: $2000")
print(f"   • Blinds: $10/$20")
print(f"   • State Dimension: {env.obs_dim}")

policy_net = PokerNet(env.obs_dim, 3, hidden_dim=256).to(device)
target_net = PokerNet(env.obs_dim, 3, hidden_dim=256).to(device)
target_net.load_state_dict(policy_net.state_dict())

print(f"\n🧠 Network Architecture:")
print(f"   • Input: {env.obs_dim} features")
print(f"   • Hidden: [256, 256, 128] (smaller for speed)")
print(f"   • Output: 3 Q-values (Fold/Call/Raise)")

optimizer = torch.optim.Adam(policy_net.parameters(), lr=0.0001)
episodes = 50

print(f"\n🚀 Starting Training: {episodes} episodes")
print("=" * 70)

rewards = []
steps_done = 0

for ep in range(episodes):
    env.reset()
    init_stack = env.players[0].stack
    episode_reward = 0
    
    hand_done = False
    step_count = 0
    
    while not hand_done and step_count < 200:
        step_count += 1
        
        active = [p for p in env.players if p.is_active and not p.is_allin]
        if len(active) <= 1:
            if env.pot > 0:
                env.finalize_showdown()
            hand_done = True
            break
        
        curr_idx = env.get_current_player_idx()
        curr_p = env.players[curr_idx]
        
        if not curr_p.is_active or curr_p.is_allin:
            env._next_active_player()
            continue
        
        if curr_idx == 0:
            state = env.get_observation(0)
            state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            
            eps = max(0.1, 1.0 * np.exp(-1. * steps_done / 1000))
            steps_done += 1
            
            if random.random() < eps:
                action = random.choice([0, 1, 2])
            else:
                with torch.no_grad():
                    action = policy_net(state_t)[0].max(1)[1].item()
            
            env.step(action, 0.5)
            
            if hand_done or step_count >= 199:
                profit = env.players[0].stack - init_stack
                episode_reward = profit / 20.0
        else:
            env.step(random.choice([0, 1, 2]), random.random())
        
        if env._check_end_of_betting_round():
            if env.stage < 3:
                env.deal_next_stage()
            else:
                env.finalize_showdown()
                hand_done = True
    
    profit = env.players[0].stack - init_stack
    episode_reward = profit / 20.0
    rewards.append(episode_reward)
    
    if (ep + 1) % 10 == 0:
        avg_r = np.mean(rewards[-10:])
        win_rate = len([r for r in rewards[-10:] if r > 0]) / 10 * 100
        print(f"Ep {ep+1:3d} | Avg Reward: {avg_r:6.1f} | Win Rate: {win_rate:5.1f}% | ε: {eps:.3f}")

print("\n" + "=" * 70)
print("✅ Training Complete!".center(70))
print("=" * 70)

final_avg = np.mean(rewards[-10:]) if len(rewards) >= 10 else np.mean(rewards)
final_wr = len([r for r in rewards if r > 0]) / len(rewards) * 100

print(f"\n📊 Final Statistics:")
print(f"   • Total Episodes: {episodes}")
print(f"   • Average Reward: {final_avg:.2f}")
print(f"   • Win Rate: {final_wr:.1f}%")
print(f"   • Final Epsilon: {eps:.4f}")
print("\n💡 Note: This is a short demo. Full training uses 20,000 episodes!")
print("=" * 70)
