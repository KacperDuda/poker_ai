import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
import numpy as np
from typing import List, Optional

# Importujemy definicję typu z Twojego pliku środowiska
# Dzięki temu IDE wie, że funkcja zwraca krotkę (int, float)
from poker_env import PlayerAction 

# --- 1. SIEĆ NEURONOWA (MÓZG) ---
class PokerNetwork(nn.Module):
    """
    Sieć neuronowa.
    Wejście: Tensor obserwacji (stan gry).
    Wyjście: Logity akcji (co zrobić) oraz wartość suwaka (ile postawić).
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super(PokerNetwork, self).__init__()
        
        # Warstwy przetwarzające (Feature extraction)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # GŁOWICA 1: Decyzja dyskretna (Fold / Call / Raise)
        self.action_head = nn.Linear(hidden_dim, out_features=3)
        
        # GŁOWICA 2: Decyzja ciągła (Bet Sizing - % z min-max)
        self.slider_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Zwraca: (action_logits, slider_value)
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        action_logits = self.action_head(x)
        # Sigmoid zapewnia, że wartość jest zawsze między 0.0 a 1.0
        slider_val = torch.sigmoid(self.slider_head(x))
        
        return action_logits, slider_val

# --- 2. KLASY AGENTÓW ---

class Agent:
    """Klasa bazowa."""
    def __init__(self, agent_id: int):
        self.id = agent_id

    def get_action(self, observation: np.ndarray, legal_moves=None) -> PlayerAction:
        """
        Główna metoda interfejsu.
        Musi przyjąć obserwację (numpy array) i zwrócić PlayerAction (int, float).
        """
        raise NotImplementedError

class RandomAgent(Agent):
    """Agent wykonujący losowe ruchy - do testów."""
    def get_action(self, observation: np.ndarray, legal_moves=None) -> PlayerAction:
        action_idx = np.random.choice([0, 1, 2])
        amount = np.random.random() # float 0.0 - 1.0
        
        # Jawne rzutowanie na typy proste, by pasowało do definicji PlayerAction
        return int(action_idx), float(amount)

class DeepAgent(Agent):
    """
    Agent RL używający sieci neuronowej.
    Zbiera log_probs w self.log_probs dla algorytmu REINFORCE.
    """
    def __init__(self, agent_id: int, input_dim: int, shared_model: Optional[PokerNetwork] = None):
        super().__init__(agent_id)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Współdzielenie modelu (Self-Play): Wszyscy agenci DeepAgent używają tej samej instancji sieci.
        if shared_model:
            self.net = shared_model
        else:
            self.net = PokerNetwork(input_dim).to(self.device)
            
        # Pamięć epizodu na logarytmy prawdopodobieństw (potrzebne do liczenia gradientu)
        self.log_probs: List[torch.Tensor] = []

    def get_action(self, observation: np.ndarray, legal_moves=None) -> PlayerAction:
        # 1. Konwersja numpy -> tensor
        obs_tensor = torch.FloatTensor(observation).to(self.device).unsqueeze(0)
        
        # 2. Przejście przez sieć
        # Nie używamy torch.no_grad() tutaj, bo potrzebujemy gradientów w self.log_probs!
        logits, slider_tensor = self.net(obs_tensor)
        
        # 3. Wybór akcji (Action Head)
        probs = F.softmax(logits, dim=-1)
        dist = distributions.Categorical(probs)
        action_idx_tensor = dist.sample()
        
        # 4. Zapisanie log_prob do historii (kluczowe dla REINFORCE)
        # To łączy obecny krok z późniejszą funkcją straty (Loss)
        log_prob = dist.log_prob(action_idx_tensor)
        self.log_probs.append(log_prob)
        
        # 5. Przygotowanie wyniku zgodnego z typem PlayerAction (int, float)
        action_idx = action_idx_tensor.item()
        slider_val = slider_tensor.item()
        
        return int(action_idx), float(slider_val)

    def clear_memory(self):
        """Czyści historię log_probs przed nowym rozdaniem."""
        self.log_probs = []