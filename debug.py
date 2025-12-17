from poker_env import PokerEnv
import numpy as np

def debug_observation():
    players_ids = [0, 1, 2]
    env = PokerEnv(players_ids, initial_stack=1000.0)
    env.reset()
    
    # Get observation for our bot (ID: 0)
    obs = env.get_observation(0)
    
    print("\n--- OBSERVATION DEBUG ---")
    print(f"Input size (obs_dim): {len(obs)}")
    print(f"First 20 values: {obs[:20]}")
    print(f"Are all values zero? {'YES' if np.all(obs == 0) else 'NO'}")
    print(f"Max value in input: {np.max(obs)}")
    print(f"Min value in input: {np.min(obs)}")
    
    # Check player cards directly from the engine
    print(f"Player 0 cards in engine: {env.players[0].hand}")
    print("------------------------\n")

debug_observation()