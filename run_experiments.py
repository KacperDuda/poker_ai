import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
from collections import deque, namedtuple
import os
import csv
import time

# Robust Device Check
try:
    if torch.cuda.is_available():
        t = torch.tensor([1.0], device="cuda")
        _ = t + t
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
except RuntimeError:
    device = torch.device("cpu")

from agent import PokerNet
from poker_env import PokerEnv
import evaluator

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    def push(self, *args):
        self.memory.append(Transition(*args))
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    def __len__(self):
        return len(self.memory)

def optimize_model(policy_net, target_net, memory, optimizer, batch_size=128, gamma=0.99):
    if len(memory) < batch_size: return
    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions))
    
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    
    state_vals = policy_net(state_batch)[0].gather(1, action_batch)
    next_vals = torch.zeros(batch_size, device=device)
    if non_final_mask.sum() > 0:
        with torch.no_grad():
            next_vals[non_final_mask] = target_net(non_final_next_states)[0].max(1)[0]
    
    expected = (next_vals.unsqueeze(1) * gamma) + reward_batch
    loss = nn.SmoothL1Loss()(state_vals, expected)
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
    optimizer.step()

def run_experiment(exp_name, config, num_episodes=20000):
    print(f"\n>>> STARTING EXPERIMENT: {exp_name}")
    print(f"    Config: {config}")
    
    # Config parameters
    n_players = config.get("n_players", 3)
    hidden_dim = config.get("hidden_dim", 512)
    lr = config.get("lr", 0.0001)
    eps_decay = config.get("eps_decay", 30000)
    
    # Setup
    os.makedirs("docs/experiments", exist_ok=True)
    log_path = f"docs/experiments/log_{exp_name}.csv"
    
    players_ids = list(range(n_players))
    env = PokerEnv(players_ids, initial_stack=2000.0)
    
    # Network
    policy_net = PokerNet(env.obs_dim, 3, hidden_dim=hidden_dim).to(device)
    target_net = PokerNet(env.obs_dim, 3, hidden_dim=hidden_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    memory = ReplayMemory(100000)
    
    # Logging
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Episode", "AvgReward", "WinRate", "Epsilon"])
    
    steps_done = 0
    results_window = deque(maxlen=100)
    
    start_time = time.time()
    
    for i_episode in range(num_episodes):
        env.reset()
        init_stack = env.players[0].stack
        last_state = None
        last_action = None
        
        hand_done = False
        while not hand_done:
            # Game Loop (simplified)
            # Check Round End
            active = [p for p in env.players if p.is_active and not p.is_allin]
            if len(active) <= 1: 
                if env.pot > 0: env.finalize_showdown()
                break
                
            curr_idx = env.get_current_player_idx()
            curr_p = env.players[curr_idx]
            
            if not curr_p.is_active or curr_p.is_allin:
                env._next_active_player()
                continue
            
            # Action
            if curr_idx == 0: # HERO
                state = env.get_observation(0)
                state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                
                # Store prev transition
                if last_state is not None:
                    memory.push(last_state, last_action, state_t, torch.tensor([[0.0]], device=device))
                    optimize_model(policy_net, target_net, memory, optimizer)
                
                # Select Action
                eps = 0.05 + (1.0 - 0.05) * np.exp(-1. * steps_done / eps_decay)
                steps_done += 1
                
                if random.random() < eps:
                    action = random.choice([0,1,2]) # Pure random exploration is properly unbiased
                else:
                    with torch.no_grad():
                        action = policy_net(state_t)[0].max(1)[1].item()
                        
                env.step(action, 0.5) # simplify slider
                
                last_state = state_t
                last_action = torch.tensor([[action]], device=device)
            else:
                # Opponent (Random)
                env.step(random.choice([0,1,2]), random.random())
            
            # Check Stage End
            if env._check_end_of_betting_round():
                if env.stage < 3: env.deal_next_stage()
                else: 
                    env.finalize_showdown()
                    hand_done = True
        
        # End Episode
        if last_state is not None:
            profit = env.players[0].stack - init_stack
            reward = np.clip(profit / 2000.0, -1, 1)
            money_tens = torch.tensor([[reward]], device=device, dtype=torch.float32)
            memory.push(last_state, last_action, None, money_tens)
            optimize_model(policy_net, target_net, memory, optimizer)
            results_window.append(profit)
            
        if i_episode % 500 == 0:
            target_net.load_state_dict(policy_net.state_dict())
            
        # Progress Log
        if i_episode % 100 == 0:
            avg = sum(results_window)/len(results_window) if results_window else 0
            wr = len([x for x in results_window if x > 0]) / len(results_window) * 100 if results_window else 0
            
            with open(log_path, 'a', newline='') as f:
                csv.writer(f).writerow([i_episode, avg, wr, eps])
            
            if i_episode % 1000 == 0 and i_episode > 0:
                elapsed = time.time() - start_time
                pct = (i_episode / num_episodes) * 100
                print(f"    [Exp {exp_name}] {pct:.1f}% | Ep {i_episode} | Avg {avg:.1f} | WR {wr:.1f}% | Time {elapsed:.1f}s")

    # SAVE MODEL
    torch.save(policy_net.state_dict(), f"docs/experiments/model_{exp_name}.pth")
    print(f"    [Exp {exp_name}] COMPLETED. Saved model.")


if __name__ == "__main__":
    experiments = [
        ("baseline", {"n_players": 3, "hidden_dim": 512, "eps_decay": 10000}),
        ("heads_up", {"n_players": 2, "hidden_dim": 512, "eps_decay": 10000}), 
        ("bignet",   {"n_players": 3, "hidden_dim": 1024, "eps_decay": 10000}),
        ("long_decay", {"n_players": 3, "hidden_dim": 512, "eps_decay": 30000})
    ]
    
    print("=== LAUNCHING BIG EXPERIMENTS (4 Configs) ===")
    print(f"Device: {device}")
    
    for name, cfg in experiments:
        run_experiment(name, cfg, num_episodes=20000) # Short enough for demo, long enough relevant
        
    print("\nALL EXPERIMENTS FINISHED.")
