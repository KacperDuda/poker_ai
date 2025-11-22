import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Agent:
    def __init__(self, agent_id):
        self.id = agent_id

    def get_action(self, observation, legal_moves=None):
        raise NotImplementedError

# --- KLASYCZNI AGENCI ---

class RandomAgent(Agent):
    def get_action(self, observation, legal_moves=None):
        return random.choice([0, 1, 2]), random.random()

class ConservativeAgent(Agent):
    def get_action(self, observation, legal_moves=None):
        roll = random.random()
        if roll < 0.1: return 0, 0.0 # Fold
        elif roll < 0.9: return 1, 0.0 # Call
        else: return 2, 0.5 # Raise

# --- AGENT UCZĄCY SIĘ (TORCH) ---

class PokerNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(PokerNet, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        self.action_head = nn.Linear(hidden_dim, 3) # Fold, Call, Raise
        self.slider_head = nn.Linear(hidden_dim, 1) # Suwak 0-1

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        action_logits = self.action_head(x)
        slider_val = torch.sigmoid(self.slider_head(x))
        
        return action_logits, slider_val

class DeepAgent(Agent):
    def __init__(self, agent_id, input_dim, model_path=None, shared_net=None):
        super().__init__(agent_id)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # LOGIKA WSPÓŁDZIELENIA SIECI
        if shared_net is not None:
            # Jeśli podano wspólną sieć, używamy jej (referencja)
            self.net = shared_net
        else:
            # W przeciwnym razie tworzymy nową
            self.net = PokerNet(input_dim).to(self.device)
            if model_path:
                self.net.load_state_dict(torch.load(model_path))
                self.net.eval()

    def get_action(self, observation, legal_moves=None):
        obs_tensor = torch.FloatTensor(observation).to(self.device).unsqueeze(0)
        with torch.no_grad():
            action_logits, slider_val = self.net(obs_tensor)
        
        probs = F.softmax(action_logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action_idx = dist.sample().item()
        slider_amt = slider_val.item()
        
        return action_idx, slider_amt

    def get_action_and_log_prob(self, observation):
        obs_tensor = torch.FloatTensor(observation).to(self.device).unsqueeze(0)
        action_logits, slider_val = self.net(obs_tensor)
        
        probs = F.softmax(action_logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        
        action_idx = dist.sample()
        log_prob = dist.log_prob(action_idx)
        
        return action_idx.item(), slider_val, log_prob