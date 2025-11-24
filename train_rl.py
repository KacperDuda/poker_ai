import torch
import torch.optim as optim
import numpy as np
from typing import Dict, List

from poker_env import PokerEnv
from agent import Agent, DeepAgent, PokerNetwork, RandomAgent

# --- KONFIGURACJA ---
NUM_EPISODES = 5000
LEARNING_RATE = 0.001
STACK_SIZE = 1000.0

def calculate_reward(player_stack: float, initial_stack: float) -> float:
    """Prosta funkcja nagrody: Zysk/Strata netto w żetonach."""
    return player_stack - initial_stack

def train():
    print("=== START TRENINGU (REINFORCE) ===")
    
    # Inicjalizacja środowiska z 4 graczami
    players_ids = [0, 1, 2, 3]
    env = PokerEnv(players_ids, initial_stack=STACK_SIZE)
    
    # 1. WSPÓLNA SIEĆ (Shared Brain)
    # Pobieramy wymiar obserwacji z przykładowego wywołania
    dummy_obs = env.get_observation(0)
    input_dim = dummy_obs.shape[0] # np. 109 (karty + stan)
    
    shared_model = PokerNetwork(input_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shared_model.to(device)
    
    # Optymalizator aktualizuje wagi wspólnego modelu
    optimizer = optim.Adam(shared_model.parameters(), lr=LEARNING_RATE)
    
    # 2. TWORZENIE AGENTÓW Z TYPOWANIEM
    # Słownik mapuje ID gracza (int) na instancję klasy Agent
    agents: Dict[int, Agent] = {}
    
    # Agenci 0 i 1 to AI (uczą się tego samego modelu)
    agents[0] = DeepAgent(0, input_dim, shared_model=shared_model)
    agents[1] = DeepAgent(1, input_dim, shared_model=shared_model)
    # Agenci 2 i 3 to tło (losowi)
    agents[2] = RandomAgent(2)
    agents[3] = RandomAgent(3)
    
    # Lista ID agentów, którzy podlegają treningowi
    learning_ids: List[int] = [0, 1]

    # --- PĘTLA GŁÓWNA ---
    for episode in range(NUM_EPISODES):
        
        # Reset środowiska
        _ = env.reset() # state nie jest nam tu potrzebny, bo pobieramy go w pętli
        done = False
        
        # Wyczyszczenie pamięci gradientów agentów AI
        for pid in learning_ids:
            # Rzutowanie (casting), bo my wiemy że to DeepAgent, ale słownik ma typ Agent
            agent = agents[pid]
            if isinstance(agent, DeepAgent):
                agent.clear_memory()

        # --- PĘTLA ROZDANIA ---
        while not done:
            current_pid = env.current_player_idx
            current_agent = agents[current_pid]
            
            # Pobranie obserwacji (typ: np.ndarray)
            obs = env.get_observation(current_pid)
            
            # Decyzja agenta (zwraca typ: PlayerAction -> tuple[int, float])
            action_tuple = current_agent.get_action(obs)
            
            # Wykonanie ruchu w środowisku
            # step() oczekuje dokładnie PlayerAction
            
            _, _, done, _ = env.step(action_tuple)  # pyright: ignore[reportAssignmentType]

        # --- TRENING (PO ZAKOŃCZENIU ROZDANIA) ---
        optimizer.zero_grad()
        total_policy_loss = torch.tensor(0.0, device=device)
        updates_count = 0
        
        for pid in learning_ids:
            agent = agents[pid]
            # Upewniamy się, że to agent uczący się
            if not isinstance(agent, DeepAgent): 
                continue
            
            # Jeśli agent spasował od razu (np. BB w walkowerze), może nie mieć historii
            if not agent.log_probs:
                continue

            # 1. Oblicz nagrodę (skalowaną dla stabilności numerycznej)
            player_obj = env.players[pid]
            reward_val = calculate_reward(player_obj.stack, STACK_SIZE)
            scaled_reward = reward_val / 100.0 
            
            # 2. Oblicz Loss: Suma(-log_prob * reward)
            # Tworzymy listę strat dla każdej decyzji w tym epizodzie
            loss_list = []
            for log_prob in agent.log_probs:
                # Minus, bo gradient ascent -> gradient descent
                loss_list.append(-log_prob * scaled_reward)
            
            if loss_list:
                step_loss = torch.stack(loss_list).sum()
                total_policy_loss += step_loss
                updates_count += 1

        # 3. Aktualizacja wag (Backpropagation)
        if updates_count > 0:
            total_policy_loss.backward()
            optimizer.step()

        # --- LOGOWANIE ---
        if episode % 50 == 0:
            # Wyciągamy wartość liczbową z tensora lossa dla czytelności
            loss_val = total_policy_loss.item()
            print(f"Epizod {episode} | Loss: {loss_val:.4f}")
            # Podgląd stacków
            stack_str = ", ".join([f"P{p.id}:{int(p.stack)}" for p in env.players])
            print(f"   Stacks: [{stack_str}]")

    # Zapis modelu
    torch.save(shared_model.state_dict(), "poker_ai_model.pth")
    print("Trening zakończony.")

if __name__ == "__main__":
    train()