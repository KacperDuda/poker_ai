import settings
from poker_env import PokerEnv
from player import Player
from agent import RandomAgent
from gui import PokerGUI
import pygame
import time
import os

def run_gui_simulation():
    print("=== STARTING GUI SIMULATION ===")
    
    # Init GUI
    gui = PokerGUI()
    
    # Init Env
    players_pool = [
        Player(0, 1000.0),
        Player(1, 2000.0),
        Player(2, 1000.0),
        Player(3, 1000.0)
    ]
    env = PokerEnv(players_pool, sb=10, bb=20, debug=True)
    
    # Connect GUI to Env
    env.add_observer(gui.update_state)
    
    # Agents
    random_agents = {p.id: RandomAgent(p.id) for p in env.players}
    
    # Run a few steps
    env.start_round()
    
    # Simulate a few moves
    for _ in range(10):
        pygame.event.pump() # Process event queue
        
        current_p_idx = env.get_current_player_idx()
        current_p = env.players[current_p_idx]
        
        if not current_p.is_active or current_p.is_allin:
            env._next_active_player()
            continue
            
        obs = env.get_observation(current_p_idx)
        action, val = random_agents[current_p.id].get_action(obs)
        
        env.step(action, val)
        
        if env._check_end_of_betting_round():
            env.deal_next_stage()
            
        time.sleep(0.5) # Slow down for visibility
        
    # Take screenshot
    screenshot_path = os.path.abspath("poker_ui_test.png")
    gui.save_screenshot(screenshot_path)
    print(f"Screenshot saved to: {screenshot_path}")
    
    pygame.quit()

if __name__ == "__main__":
    # Monkey patch logic if needed (it was done in main.py)
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
    
    run_gui_simulation()
