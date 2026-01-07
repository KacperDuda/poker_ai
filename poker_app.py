import sys
import os
import pygame
import random
import copy
from collections import deque

from gui import PokerGUI
from poker_env import PokerEnv
from player import Player
from agent import RandomAgent, DeepAgent

# MODE CONSTANTS
MODE_MENU = "MENU"
MODE_GAME = "GAME"
MODE_REPLAY = "REPLAY"

class PokerApp:
    def __init__(self):
        self.gui = PokerGUI()
        self.mode = MODE_MENU
        
        # MENU
        self.menu_options = ["Play 1v1 (AI vs Random)", "Watch Simulation (4 Bots)", "Exit"]
        self.menu_idx = 0
        
        # GAME / REPLAY STATE
        self.env = None
        self.agents = {}
        self.history = [] # List of (event, data) tuples for exact replay
        self.current_step = 0
        self.paused = False
        self.held_left_time = 0
        self.held_right_time = 0
        self.frame_timer = 0
        
        self.clock = pygame.time.Clock()
        self.running = True

    def run(self):
        while self.running:
            if self.mode == MODE_MENU:
                self.handle_menu()
            elif self.mode == MODE_GAME:
                self.run_game_step()
            elif self.mode == MODE_REPLAY:
                self.handle_replay()
            
            pygame.display.flip()
            self.clock.tick(60)
            
        pygame.quit()
        sys.exit()

    def handle_menu(self):
        self.gui.draw_menu(self.menu_options, self.menu_idx)
        
        # Calculate button rects (Same logic as gui.draw_menu)
        # TODO: Ideally gui.draw_menu should return these or we share logic.
        # Hardcoding the layout here for interaction:
        # btn_rect = pygame.Rect(self.gui.target_w//2 - 300, 300 + i*80 - 15, 600, 60)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.VIDEORESIZE:
                self.gui.screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
                self.gui.width = event.w
                self.gui.height = event.h
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    self.menu_idx = (self.menu_idx - 1) % len(self.menu_options)
                elif event.key == pygame.K_DOWN:
                    self.menu_idx = (self.menu_idx + 1) % len(self.menu_options)
                elif event.key == pygame.K_RETURN:
                    self.execute_menu_choice()
            elif event.type == pygame.MOUSEMOTION:
                cx, cy = self.gui.map_window_to_canvas(event.pos)
                # Check collision with buttons
                start_y = 300
                for i in range(len(self.menu_options)):
                    rect = pygame.Rect(self.gui.target_w//2 - 300, start_y + i*80 - 15, 600, 60)
                    if rect.collidepoint(cx, cy):
                        self.menu_idx = i
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1: # Left Click
                    cx, cy = self.gui.map_window_to_canvas(event.pos)
                    # Check collision and execute
                    start_y = 300
                    for i in range(len(self.menu_options)):
                        rect = pygame.Rect(self.gui.target_w//2 - 300, start_y + i*80 - 15, 600, 60)
                        if rect.collidepoint(cx, cy):
                             self.menu_idx = i
                             self.execute_menu_choice()

    def execute_menu_choice(self):
        choice = self.menu_options[self.menu_idx]
        if choice == "Exit":
            self.running = False
        elif choice == "Play 1v1 (AI vs Random)":
            self.setup_game(mode="1v1")
        elif choice == "Watch Simulation (4 Bots)":
            self.setup_game(mode="sim")

    def setup_game(self, mode):
        self.history = []
        self.current_step = 0
        self.paused = False
        
        if mode == "1v1":
            players_data = [Player(0, 2000), Player(1, 2000)]
        else:
            players_data = [Player(i, 2000) for i in range(4)]
            
        self.env = PokerEnv(players_data, sb=10, bb=20)
        
        self.env.add_observer(self.record_state)
        self.env.add_observer(self.gui.update_state)
        
        self.agents = {}
        for p in self.env.players:
            if p.id == 0:
                model_path = "poker_dqn.pth" 
                if os.path.exists(model_path):
                    self.agents[p.id] = DeepAgent(p.id, self.env.obs_dim, model_path=model_path)
                else:
                     self.agents[p.id] = RandomAgent(p.id)
            else:
                self.agents[p.id] = RandomAgent(p.id)

        player_types = {}
        for p in self.env.players:
            if isinstance(self.agents[p.id], DeepAgent): player_types[p.id] = "AI"
            else: player_types[p.id] = "BOT"
            
        self.env._notify("player_info", {"types": player_types})
        
        self.env.start_round()
        self.mode = MODE_REPLAY

    def record_state(self, event, data):
        import copy
        snapshot = copy.deepcopy(data)
        self.history.append((event, snapshot))
        if not self.paused:
            self.current_step = len(self.history) - 1

    def run_game_logic_step(self):
        env = self.env
        
        active = [p for p in env.players if p.is_active and not p.is_allin]
        if len(active) <= 1:
             if env.pot > 0: env.finalize_showdown()
             else: env.start_round() 
             return

        curr_idx = env.get_current_player_idx()
        curr_p = env.players[curr_idx]
            
        if not curr_p.is_active or curr_p.is_allin:
            env._next_active_player()
            return

        obs = env.get_observation(curr_idx)
        action, val = self.agents[curr_p.id].get_action(obs)
        env.step(action, val)
        
        if env._check_end_of_betting_round():
            if env.stage < 3: env.deal_next_stage()
            else: env.finalize_showdown()

    def handle_replay(self):
        # 1. Handle Input (Toggle Pause, Exit)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.VIDEORESIZE:
                self.gui.screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
                self.gui.width = event.w
                self.gui.height = event.h
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_ESCAPE:
                    self.mode = MODE_MENU
        
        # Continuous Scrolling (Held Keys) with Acceleration
        keys = pygame.key.get_pressed()
        
        if keys[pygame.K_LEFT]:
            self.held_left_time += 1
            if self.held_left_time == 1:
                # Immediate step on press
                self.paused = True
                self.current_step = max(0, self.current_step - 1)
                self.apply_history_state()
            elif self.held_left_time > 60:
                # Accelerating speed: Interval decreases from 10 to 1
                interval = max(1, 10 - int((self.held_left_time - 60) / 15))
                if self.held_left_time % interval == 0:
                    self.paused = True
                    self.current_step = max(0, self.current_step - 1)
                    self.apply_history_state()
        else:
            self.held_left_time = 0
            
        if keys[pygame.K_RIGHT]:
            self.held_right_time += 1
            if self.held_right_time == 1:
                self.paused = True
                self.current_step = min(len(self.history) - 1, self.current_step + 1)
                self.apply_history_state()
            elif self.held_right_time > 60:
                interval = max(1, 10 - int((self.held_right_time - 60) / 15))
                if self.held_right_time % interval == 0:
                    self.paused = True
                    self.current_step = min(len(self.history) - 1, self.current_step + 1)
                    self.apply_history_state()
        else:
            self.held_right_time = 0

        # 2. Game Logic Update (only if "Live" / not paused and at end of history)
        # 2. Game Logic / Auto-Play
        if not self.paused:
            self.frame_timer += 1
            # Deterministic speed: ~30 frames (0.5s) per step
            if self.frame_timer >= 30:
                self.frame_timer = 0
                
                # Case A: Playback
                if self.current_step < len(self.history) - 1:
                    self.current_step += 1
                    self.apply_history_state()
                # Case B: Live
                elif self.current_step >= len(self.history) - 1:
                    self.run_game_logic_step()
        
        # 3. Draw
        # Ensure GUI has the state corresponding to current_step
        if self.paused and 0 <= self.current_step < len(self.history):
            self.apply_history_state()
            
        self.gui.draw()
        self.gui.draw_timeline(self.current_step + 1, len(self.history), self.paused)

    def apply_history_state(self):
        # We need to find the last state-bearing event up to current_step
        # History contains (event, data). 
        # 'player_info' is persistent. 
        # 'step', 'new_round', 'stage_change' all update the visual state.
        # Ideally, every history item is a full state snapshot?
        # My poker_env._notify generates a FULL snapshot every time!
        # So we just take history[current_step].data
        
        if not self.history: return
        
        evt, data = self.history[self.current_step]
        # We need to manually set gui state
        self.gui.state = data
        # We also need to help gui recreate 'last_actions' if jumping randomly?
        # The GUI logic accumulates last_actions. Jumping back breaks this.
        # Ideally, snapshot should contain 'last_actions'. 
        # Currently it doesn't. 
        # Allow GUI to be imperfect on rewind (actions might disappear), or fixing it properly:
        # We can replay 'step' events from the start of the current round.
        # But for now, let's accept that 'last_actions' might be glitchy on rewind. 
        # However, `draw()` calls clear it on `new_round`. 
        pass


if __name__ == "__main__":
    app = PokerApp()
    app.run()
