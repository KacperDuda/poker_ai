import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, namedtuple
from typing import Dict, List, Any
from agent import PokerNet

from poker_env import PokerEnv
import evaluator

device = torch.device("cpu")
print(f"Training on device: {device}")

BATCH_SIZE = 128          
GAMMA = 0.99              
EPS_START = 1.0           
EPS_END = 0.05            
EPS_DECAY = 30000         
TARGET_UPDATE = 500       
LEARNING_RATE = 0.0001    
MEMORY_SIZE = 100000      

NORMALIZATION_FACTOR = 2000.0

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


def get_expert_action(player, community_cards, current_bet, max_bet_on_table):
    to_call = max_bet_on_table - current_bet
    can_check = (to_call <= 0.01)

    if not community_cards:
        if len(player.hand) < 2: return 0 
        c1, c2 = player.hand[0], player.hand[1]
        
        if c1.rank_idx == c2.rank_idx:
            return 2 if c1.rank_idx >= 6 else 1
        
        if c1.rank_idx >= 9 or c2.rank_idx >= 9:
            if c1.rank_idx >= 10 and c2.rank_idx >= 10: return 2
            return 1
            
        if c1.suit_idx == c2.suit_idx: return 1
        if to_call < 50: return 1
        if can_check: return 1
        return 0 

    score = evaluator.get_best_hand(player.hand, community_cards)
    hand_rank = score[0] 
    
    if hand_rank >= 1: return 2 
    if np.random.rand() < 0.4: return 1 
    if can_check: return 1
    return 0

def optimize_model(policy_net, target_net, memory, optimizer):
    if len(memory) < BATCH_SIZE:
        return 0.0
    
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    
    non_final_next_states = None
    if non_final_mask.sum() > 0:
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch)[0].gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    
    if non_final_next_states is not None:
        with torch.no_grad():
            next_state_values[non_final_mask] = target_net(non_final_next_states)[0].max(1)[0]

    expected_state_action_values = (next_state_values.unsqueeze(1) * GAMMA) + reward_batch

    criterion = nn.SmoothL1Loss() 
    loss = criterion(state_action_values, expected_state_action_values)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0) 
    optimizer.step()
    
    return loss.item()

def train_dqn():
    players_ids = [0, 1, 2]
    
    env = PokerEnv(players_ids, initial_stack=2000.0)
    
    n_actions = 3 
    
    policy_net = PokerNet(env.obs_dim, n_actions).to(device)
    target_net = PokerNet(env.obs_dim, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict()) 
    target_net.eval() 

    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    memory = ReplayMemory(MEMORY_SIZE)

    steps_done = 0
    num_episodes = 200000
    
    results_window = deque(maxlen=100)
    action_counts = {0: 0, 1: 0, 2: 0}

    print("--- START DQN TRAINING (FIXED) ---")

    for episode in range(num_episodes):
        env.reset()
        initial_stack = env.players[0].stack
        
        last_state = None
        last_action = None
        
        hand_done = False
        
        while not hand_done:
            betting_round_active = True
            while betting_round_active:
                
                active_players = [p for p in env.players if p.is_active and not p.is_allin]
                if len(active_players) <= 1:
                    betting_round_active = False
                    continue
                
                max_bet = max(p.current_bet for p in env.players)
                all_active_equal = all(p.current_bet == max_bet for p in env.players if p.is_active)
                curr_p = env.players[env.current_player_idx]
                to_call = max_bet - curr_p.current_bet
                
                if to_call <= 0.01 and all_active_equal:
                    betting_round_active = False
                    continue

                current_idx = env.get_current_player_idx()
                
                if not env.players[current_idx].is_active or env.players[current_idx].is_allin:
                    env._next_active_player()
                    continue

                if current_idx == 0:
                    state = env.get_observation(current_idx)
                    state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                    
                    if last_state is not None:
                        reward_placeholder = torch.tensor([[0.0]], device=device)
                        memory.push(last_state, last_action, state_tensor, reward_placeholder)
                        
                        optimize_model(policy_net, target_net, memory, optimizer)

                    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                        np.exp(-1. * steps_done / EPS_DECAY)
                    steps_done += 1
                    
                    action_idx = 0
                    if random.random() < eps_threshold:
                        player_obj = env.players[0]
                        max_bet_table = max(p.current_bet for p in env.players)
                        action_idx = get_expert_action(player_obj, env.community_cards, player_obj.current_bet, max_bet_table)
                    else:
                        with torch.no_grad():
                            q_values, _ = policy_net(state_tensor)
                            action_idx = q_values.max(1)[1].item()

                    action_counts[action_idx] += 1
                    
                    action_tensor = torch.tensor([[action_idx]], device=device, dtype=torch.long)
                    
                    slider = 0.0
                    if action_idx == 2: slider = 0.6
                    
                    env.step(action_idx, slider)
                    
                    last_state = state_tensor
                    last_action = action_tensor

                else:
                    act = random.choice([0, 1, 2])
                    slid = random.random()
                    env.step(act, slid)

                if sum(p.is_active for p in env.players) <= 1:
                    betting_round_active = False
                    hand_done = True

            if hand_done: break
            if env.stage < 3: env.deal_next_stage()
            else: hand_done = True

        if env.pot > 0 and len([p for p in env.players if p.is_active]) > 1:
             env.finalize_showdown() 

        if last_state is not None:
            raw_profit = env.players[0].stack - initial_stack
            reward_val = raw_profit / NORMALIZATION_FACTOR
            
            if reward_val > 0: reward_val *= 1.2
            reward_val = np.clip(reward_val, -1.0, 1.0)
            
            reward_tensor = torch.tensor([[reward_val]], device=device, dtype=torch.float32)
            
            memory.push(last_state, last_action, None, reward_tensor)
            
            optimize_model(policy_net, target_net, memory, optimizer)
            results_window.append(raw_profit)

        if steps_done % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if episode % 100 == 0 and episode > 0:
            avg_gain = sum(results_window) / len(results_window)
            wins = len([r for r in results_window if r > 0])
            win_rate = (wins / len(results_window)) * 100
            
            total_acts = sum(action_counts.values())
            actions_dist = {k: round(v/total_acts, 2) for k,v in action_counts.items()} if total_acts > 0 else {}
            
            print(f"Ep {episode}: Avg: {avg_gain:.1f}$ | WinRate: {win_rate:.1f}% | Epsilon: {eps_threshold:.2f} | Acts: {actions_dist}")
            action_counts = {0: 0, 1: 0, 2: 0}

        if episode % 5000 == 0:
            torch.save(policy_net.state_dict(), "poker_dqn.pth")

if __name__ == "__main__":
    train_dqn()