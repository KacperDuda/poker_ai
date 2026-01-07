import random
import torch
import torch.nn as nn
import torch.nn.functional as F

class Agent:
    def __init__(self, agent_id):
        self.id = agent_id

    def get_action(self, observation, legal_moves=None):
        raise NotImplementedError

class RandomAgent(Agent):
    def get_action(self, observation, legal_moves=None):
        return random.choice([0, 1, 2]), random.random()

class ConservativeAgent(Agent):
    def get_action(self, observation, legal_moves=None):
        roll = random.random()
        if roll < 0.1: return 0, 0.0 
        elif roll < 0.9: return 1, 0.0 
        else: return 2, 0.5 

class PokerNet(nn.Module):
    def __init__(self, input_dim, n_actions=3):
        super(PokerNet, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)
        
        self.q_head = nn.Linear(256, n_actions)
        self.slider_head = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        q_values = self.q_head(x)
        slider_val = torch.sigmoid(self.slider_head(x))
        
        return q_values, slider_val

class DeepAgent(Agent):
    def __init__(self, agent_id, input_dim, model_path=None, shared_net=None):
        super().__init__(agent_id)
        self.device = torch.device("cpu")
        
        if shared_net is not None:
            self.net = shared_net
        else:
            self.net = PokerNet(input_dim).to(self.device)
            
            if model_path:
                try:
                    self.net.load_state_dict(torch.load(model_path, map_location=self.device))
                    self.net.eval() 
                    print(f"DeepAgent {agent_id}: Model loaded from {model_path}")
                except Exception as e:
                    print(f"DeepAgent {agent_id}: Could not load model from {model_path}. Starting fresh. Error: {e}")

    def get_action(self, observation, legal_moves=None):
        obs_tensor = torch.FloatTensor(observation).to(self.device).unsqueeze(0)
        
        with torch.no_grad():
            q_values, slider_val = self.net(obs_tensor)
        
        action_idx = q_values.argmax(dim=1).item()
        slider_amt = slider_val.item()
        
        if action_idx == 2 and slider_amt < 0.5:
            slider_amt = 0.5
            
        return action_idx, slider_amt