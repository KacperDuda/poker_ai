from agent import RandomAgent, DeepAgent
from poker_env import PokerEnv
from player import Player
from gui import PokerGUI
import pygame
import time
import os
import random

def _check_end_of_betting_round_logic(self):
    active_players = [p for p in self.players if p.is_active and not p.is_allin]
    if len(active_players) <= 1: return True 
    max_bet = max(p.current_bet for p in self.players)
    all_active_equal = all(p.current_bet == max_bet for p in self.players if p.is_active)
    next_player = self.players[self.current_player_idx]
    to_call = max_bet - next_player.current_bet
    if to_call <= 0.01 and all_active_equal: return True
    return False

setattr(PokerEnv, "_check_end_of_betting_round", _check_end_of_betting_round_logic)

def run_demo():
    print("=== STARTING ENHANCED DEMO (10 Games) ===")
    
    gui = PokerGUI()
    
    model_path = "poker_dqn.pth"
    use_ai = True 
    
    if os.path.exists(model_path):
        print(f"AI Model found at {model_path}. Player 0 will be Pre-Trained AI.")
    else:
        print("Model file not found. Player 0 will be Untrained AI (Random Init).")

    for game_idx in range(1, 11):
        n_players = random.randint(2, 6)
        starting_stack = random.choice([500.0, 1000.0, 2000.0, 5000.0])
        
        print(f"\n--- Game {game_idx}/10: {n_players} Players, Stack {starting_stack} ---")
        
        players_data = [Player(i, starting_stack) for i in range(n_players)]
        env = PokerEnv(players_data, sb=10, bb=20, debug=False)
        
        env.add_observer(gui.update_state)
        
        agents = {}
        for p in env.players:
            if p.id == 0 and use_ai:
                 agents[p.id] = DeepAgent(p.id, env.obs_dim, model_path=model_path)
            else:
                 agents[p.id] = RandomAgent(p.id)

        player_types = {p.id: ("AI" if (p.id==0 and use_ai) else "BOT") for p in env.players}
        env._notify("player_info", {"types": player_types})
        
        if not env.start_round():
            print("Failed to start round (not enough players?)")
            continue
            
        game_over = False
        steps = 0
        watchdog = 0
        
        while not game_over:
            pygame.event.pump()
            
            if not gui.running:
                print("GUI closed by user.")
                return

            current_p_idx = env.get_current_player_idx()
            current_p = env.players[current_p_idx]
            
            active_count = sum(1 for p in env.players if p.is_active)
            if active_count < 2:
                env.step(0,0) 
                if env.pot > 0: env.finalize_showdown() 
                break

            if watchdog > 200:
                print(" !!! WATCHDOG: Game Stuck. Breaking loop.")
                break

            if not current_p.is_active or current_p.is_allin:
               env._next_active_player()
               watchdog += 1
               continue

            obs = env.get_observation(current_p_idx)
            
            time.sleep(1.5)
            
            action, val = agents[current_p.id].get_action(obs)
            _, done, info = env.step(action, val)
            watchdog = 0 
            
            if done:
                print(f"  Hand finished. Winner: {info.get('winner')}")
                game_over = True
            
            if env._check_end_of_betting_round():
                 if env.stage < 4: 
                     env.deal_next_stage()
                 else:
                     env.finalize_showdown()
                     game_over = True
                     print("  Showdown!")

            steps += 1
            if steps > 100: 
                print("  Forcing end of hand (too many steps)")
                game_over = True
                
        time.sleep(2.0) 

    pygame.quit()

if __name__ == "__main__":
    run_demo()
