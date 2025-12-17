from poker_env import PokerEnv
import numpy as np
def debug_observation():
    players_ids = [0, 1, 2]
    env = PokerEnv(players_ids, initial_stack=1000.0)
    env.reset()
    
    # Pobierz obserwację dla naszego bota (0)
    obs = env.get_observation(0)
    
    print("\n--- DEBUG OBSERWACJI ---")
    print(f"Rozmiar wejścia (obs_dim): {len(obs)}")
    print(f"Przykładowe 20 wartości: {obs[:20]}")
    print(f"Czy są same zera? {'TAK' if np.all(obs == 0) else 'NIE'}")
    print(f"Max wartość w wejściu: {np.max(obs)}")
    print(f"Min wartość w wejściu: {np.min(obs)}")
    
    # Sprawdźmy karty gracza z silnika
    print(f"Karty gracza 0 w silniku: {env.players[0].hand}")
    print("------------------------\n")

debug_observation()