import torch
import torch.optim as optim
import numpy as np
from poker_env import PokerEnv
from agent import DeepAgent, RandomAgent, PokerNet

#do dopracowania, korzystamy tutaj ze starych metod od agenta oraz od poker_env

def calculate_reward(player, initial_stack):
    return player.stack - initial_stack

def train():
    # 1. Konfiguracja 6 graczy
    # ID: 0, 1, 2, 3, 4, 5
    # Agenci AI: 0, 2, 4
    # Randomy: 1, 3, 5
    players_ids = [0, 1, 2, 3, 4, 5]
    env = PokerEnv(players_ids, initial_stack=1000)
    
    # --- TWORZENIE AGENTÓW ---
    
    # Krok A: Tworzymy JEDNĄ sieć neuronową (Wspólny Mózg)
    # Dzięki temu wszyscy agenci AI uczą się tej samej strategii
    shared_brain = PokerNet(input_dim=env.obs_dim)
    # Przenosimy na GPU jeśli dostępne (w DeepAgent robimy to przez referencję, ale tu warto jawnie)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shared_brain.to(device)
    
    # Krok B: Inicjalizacja słownika agentów
    agents = {}
    ai_agent_ids = [0, 2, 4] # Lista ID naszych uczących się botów
    
    for pid in players_ids:
        if pid in ai_agent_ids:
            # Przekazujemy shared_net!
            agents[pid] = DeepAgent(pid, env.obs_dim, shared_net=shared_brain)
        else:
            agents[pid] = RandomAgent(pid)

    # Optimizer aktualizuje parametry wspólnego mózgu
    optimizer = optim.Adam(shared_brain.parameters(), lr=0.001)
    
    num_episodes = 1000
    print(f"--- START TRENINGU 6-OSOBOWEGO ({num_episodes} rozdań) ---")
    print(f"DeepAgents (Shared Brain): {ai_agent_ids}")
    print(f"RandomAgents: {[p for p in players_ids if p not in ai_agent_ids]}")

    for episode in range(num_episodes):
        env.reset()
        
        # Pamięć log_probs osobna dla każdego agenta AI
        # Struktura: { 0: [log_prob1, log_prob2], 2: [...], 4: [...] }
        episode_memory = {pid: [] for pid in ai_agent_ids}
        
        # Zapamiętujemy stacki początkowe, żeby policzyć nagrody
        initial_stacks = {p.id: p.stack for p in env.players}
        
        done = False
        
        while not done:
            current_idx = env.get_current_player_idx()
            current_agent = agents[current_idx]
            obs = env.get_observation(current_idx)
            
            if current_idx in ai_agent_ids:
                # To jest Agent AI - musimy zapisać log_prob do treningu
                action, slider, log_prob = current_agent.get_action_and_log_prob(obs)
                
                # Zapisujemy decyzję w pamięci KONKRETNEGO agenta
                episode_memory[current_idx].append(log_prob)
                
                # Wykonujemy ruch (konwersja slider tensor -> float)
                env.step(action, slider.item())
                
            else:
                # To jest Random - po prostu gra, nie uczymy go
                action, slider = current_agent.get_action(obs)
                env.step(action, slider)

            # Warunki końca gry
            active = [p for p in env.players if p.is_active]
            if len(active) == 1 or (env.stage == 3 and env.min_raise == 0):
                done = True
            if env.pot > 10000: done = True # Safety brake

        # --- KONIEC ROZDANIA: OBLICZANIE STRATY (LOSS) ---
        
        total_loss = 0
        rewards_summary = []

        # Iterujemy po każdym agencie AI z osobna
        for pid in ai_agent_ids:
            player_obj = env.players[pid]
            memory = episode_memory[pid]
            
            if not memory: 
                continue # Gracz mógł spasować od razu i nie podjąć żadnej decyzji (np. był na BB i wszyscy spasowali)

            # 1. Nagroda dla TEGO konkretnego gracza
            raw_reward = calculate_reward(player_obj, initial_stacks[pid])
            normalized_reward = raw_reward / 100.0
            rewards_summary.append(raw_reward)
            
            # 2. Strata dla TEGO gracza
            player_loss = []
            for log_prob in memory:
                player_loss.append(-log_prob * normalized_reward)
            
            if player_loss:
                # Sumujemy stratę tego gracza do wspólnego worka
                total_loss += torch.stack(player_loss).sum()

        # Aktualizacja wspólnej sieci (jeśli ktokolwiek grał)
        if isinstance(total_loss, torch.Tensor):
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        # Logowanie
        if episode % 50 == 0:
            avg_gain = sum(rewards_summary) / len(rewards_summary) if rewards_summary else 0
            loss_val = total_loss.item() if isinstance(total_loss, torch.Tensor) else 0
            print(f"Epizod {episode}: Śr. wynik AI: {avg_gain:.1f}$, Loss: {loss_val:.4f}")

    # Zapisz model (wystarczy zapisać shared_brain)
    torch.save(shared_brain.state_dict(), "poker_6max_shared.pth")
    print("Trening zakończony. Model zapisany.")

if __name__ == "__main__":
    train()