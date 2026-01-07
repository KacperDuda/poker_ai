import pygame
import sys
import math

class PokerGUI:
    def __init__(self, width=1400, height=900):
        pygame.init()
        # Actual resizing logic is handled by scaling
        # Virtual Resolution
        self.target_w = 1400
        self.target_h = 900
        
        # Initial Window Size (can be resized)
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
        pygame.display.set_caption("Poker AI Visualization")
        
        # Virtual Canvas
        self.canvas = pygame.Surface((self.target_w, self.target_h))
        
        # Background is now generated procedurally
        self.background_img = None
        
        self.GREEN_FELT = (30, 100, 30)
        self.WHITE = (245, 245, 245)
        self.BLACK = (20, 20, 20)
        self.RED = (200, 30, 30)
        self.BLUE = (30, 30, 200)
        self.GRAY = (100, 100, 100)
        self.YELLOW = (255, 215, 0)
        self.DARK_RED = (100, 0, 0)
        
        self.ACT_FOLD = (100, 100, 100)
        self.ACT_CHECK = (50, 50, 200)
        self.ACT_CALL = (50, 50, 200)
        self.ACT_RAISE = (50, 200, 50)
        self.ACT_ALLIN = (200, 50, 50)

        self.font_small = pygame.font.SysFont("Arial", 18, bold=True)
        self.font_card = pygame.font.SysFont("Arial", 42, bold=True)
        self.font_ui = pygame.font.SysFont("Arial", 22)
        self.large_font = pygame.font.SysFont("Arial", 48, bold=True)
        
        self.state = None
        self.last_actions = {} 
        self.player_types = {} 
        self.running = True
        
        self.CARD_W = 100
        self.CARD_H = 140
        self.CARD_RADIUS = 10
        
        # Initialize scaling params for map_window_to_canvas
        self.scale = 1.0
        self.start_x = 0
        self.start_y = 0

    def update_state(self, event, data):
        self.state = data
        
        if event == "player_info":
            self.player_types = data.get("types", {})
            return
        
        if event == "step":
            pid = data.get("player_id")
            act_type = data.get("action")
            if act_type == 0:
                self.last_actions[pid] = ("FOLD", self.ACT_FOLD)
            elif act_type == 1:
                self.last_actions[pid] = ("CALL", self.ACT_CALL)
            elif act_type == 2:
                self.last_actions[pid] = ("RAISE", self.ACT_RAISE)
        
        if event == "new_round" or event == "stage_change":
            if event == "new_round": self.last_actions.clear()

    def draw_chip_stack(self, amount, x, y):
        if amount <= 0: return
        
        chips = [
            (500, (200, 150, 50), (100, 70, 0)), 
            (100, (20, 20, 20), (200, 200, 200)), 
            (25, (20, 100, 20), (150, 200, 150)), 
            (5, (150, 20, 20), (250, 100, 100)), 
            (1, (200, 200, 250), (50, 50, 150)), 
        ]
        
        remaining = amount
        y_cursor = y
        
        chip_counts = []
        for val, fill_col, line_col in chips:
            count = int(remaining // val)
            remaining %= val
            if count > 0:
                chip_counts.append((count, (fill_col, line_col)))
        
        max_height_px = 60
        total_chips = sum(c[0] for c in chip_counts)
        if total_chips == 0 and amount > 0: 
             chip_counts.append((1, chips[-1][1]))
        
        step_y = 4
        current_y = y
        
        drawn = 0
        for count, (fill_col, line_col) in chip_counts:
            for _ in range(count):
                if drawn > 15: break 
                
                pygame.draw.ellipse(self.canvas, fill_col, (x - 15, current_y - 10, 30, 20))
                pygame.draw.ellipse(self.canvas, line_col, (x - 15, current_y - 10, 30, 20), 2)
                
                current_y -= step_y
                drawn += 1

    def draw_card(self, card_str, x, y, w=None, h=None):
        if w is None: w = self.CARD_W
        if h is None: h = self.CARD_H
        
        shadow_offset = 4
        pygame.draw.rect(self.canvas, (0,0,0, 50), (x+shadow_offset, y+shadow_offset, w, h), border_radius=self.CARD_RADIUS) 
        
        pygame.draw.rect(self.canvas, self.WHITE, (x, y, w, h), border_radius=self.CARD_RADIUS)
        pygame.draw.rect(self.canvas, self.BLACK, (x, y, w, h), 2, border_radius=self.CARD_RADIUS)
        
        if card_str == "BACK":
            inner_rect = pygame.Rect(x+6, y+6, w-12, h-12)
            pygame.draw.rect(self.canvas, (180, 40, 40), inner_rect, border_radius=5)
            pygame.draw.rect(self.canvas, (220, 60, 60), inner_rect, 2, border_radius=5)
            return

        if not card_str: return
        
        if len(card_str) == 3: rank, suit = card_str[:2], card_str[2]
        else: rank, suit = card_str[0], card_str[1]
        
        color = self.RED if suit in ['h', 'd'] else self.BLACK
        suit_sym = {'h': '♥', 'd': '♦', 's': '♠', 'c': '♣'}.get(suit, suit)
        
        rank_text = self.font_card.render(f"{rank}", True, color)
        self.canvas.blit(rank_text, (x + 6, y + 4))
        
        suit_text_small = self.font_small.render(suit_sym, True, color)
        self.canvas.blit(suit_text_small, (x + 8, y + 45))
        
        br_surf = self.font_card.render(suit_sym, True, color)
        br_rect = br_surf.get_rect(bottomright=(x+w-6, y+h-4))
        self.canvas.blit(br_surf, br_rect)

    def draw_table_and_cards(self, state):
        if self.background_img:
            self.canvas.blit(self.background_img, (0, 0))
        else:
            self.canvas.fill(self.GREEN_FELT)
            table_w, table_h = 1000, 500
            table_rect = pygame.Rect((self.target_w - table_w)//2, (self.target_h - table_h)//2, table_w, table_h)
            pygame.draw.ellipse(self.canvas, (40, 130, 40), table_rect) 
            pygame.draw.ellipse(self.canvas, (100, 50, 20), table_rect, 15)
        
        comm_cards = state.get("community_cards", [])
        card_spacing = 15
        total_w = 5 * self.CARD_W + 4 * card_spacing
        start_x = (self.target_w - total_w) // 2
        cards_y = self.target_h // 2 - self.CARD_H // 2
        
        for i in range(5):
             bx = start_x + i * (self.CARD_W + card_spacing)
             pygame.draw.rect(self.canvas, (35, 90, 35), (bx, cards_y, self.CARD_W, self.CARD_H), border_radius=10)
             pygame.draw.rect(self.canvas, (50, 110, 50), (bx, cards_y, self.CARD_W, self.CARD_H), 2, border_radius=10)

        for i, c in enumerate(comm_cards):
            cx = start_x + i * (self.CARD_W + card_spacing)
            self.draw_card(c, cx, cards_y)
            
        pot_amount = state.get('pot', 0)
        pot_y = self.target_h // 2 + 100
        if pot_amount > 0:
             self.draw_chip_stack(pot_amount, self.target_w//2, pot_y)
             pot_text = self.large_font.render(f"POT: {pot_amount:.1f}", True, self.YELLOW)
             pot_rect = pot_text.get_rect(midtop=(self.target_w//2, pot_y + 10))
             self.canvas.blit(pot_text, pot_rect)
        
        players = state.get("players", [])
        num_players = len(players)
        center_x, center_y = self.target_w // 2, self.target_h // 2
        radius_x, radius_y = 500, 280 
        
        for i, p in enumerate(players):
            angle = (2 * math.pi / num_players) * i + math.pi/2
            px = center_x + int(radius_x * math.cos(angle))
            py = center_y + int(radius_y * math.sin(angle))
            
            box_w, box_h = 160, 100

            if p['stack'] > 0:
                dir_x = (px - center_x)
                dir_y = (py - center_y)
                dist = math.hypot(dir_x, dir_y)
                if dist < 1: dist = 1
                norm_x, norm_y = dir_x/dist, dir_y/dist
                
                stack_dist = dist + 100 
                stack_x = center_x + int(norm_x * stack_dist)
                stack_y = center_y + int(norm_y * stack_dist)
                
                self.draw_chip_stack(p['stack'], stack_x, stack_y)
            
            bet_val = p.get('bet', 0)
            if bet_val > 0:
                 dir_x = (center_x - px)
                 dir_y = (center_y - py)
                 dist = math.hypot(dir_x, dir_y)
                 norm_x, norm_y = dir_x/dist, dir_y/dist
                 
                 chip_x = px + norm_x * 90
                 chip_y = py + norm_y * 90
                 
                 self.draw_chip_stack(bet_val, chip_x, chip_y)

            cards = p.get('cards', [])
            card_scale = 0.6
            small_w = int(self.CARD_W * card_scale)
            small_h = int(self.CARD_H * card_scale)
            for ci, c in enumerate(cards):
                c_x = px - box_w//2 + 30 + ci * 35
                c_y = py + 55 
                self.draw_card(c, c_x, c_y, small_w, small_h)
            
            if i == state.get("dealer"):
                db_x, db_y = px + box_w//2 - 20, py - box_h//2 - 10
                pygame.draw.circle(self.canvas, self.WHITE, (db_x, db_y), 15)
                pygame.draw.circle(self.canvas, self.BLACK, (db_x, db_y), 15, 2)
                d_text = self.font_ui.render("D", True, self.BLACK)
                d_rect = d_text.get_rect(center=(db_x, db_y))
                self.canvas.blit(d_text, d_rect)

    def render_to_screen(self):
        win_w, win_h = self.screen.get_size()
        
        curr_aspect = win_w / win_h
        target_aspect = self.target_w / self.target_h
        
        if curr_aspect > target_aspect:
             self.scale = win_h / self.target_h
             new_w = int(self.target_w * self.scale)
             new_h = win_h
             self.start_x = (win_w - new_w) // 2
             self.start_y = 0
        else:
             self.scale = win_w / self.target_w
             new_w = win_w
             new_h = int(self.target_h * self.scale)
             self.start_x = 0
             self.start_y = (win_h - new_h) // 2
             
        scaled_surf = pygame.transform.smoothscale(self.canvas, (new_w, new_h))
        
        self.screen.fill(self.BLACK)
        self.screen.blit(scaled_surf, (self.start_x, self.start_y))

    def map_window_to_canvas(self, pos):
        wx, wy = pos
        if not hasattr(self, 'scale'): return wx, wy
        
        cx = (wx - self.start_x) / self.scale
        cy = (wy - self.start_y) / self.scale
        return int(cx), int(cy)

    def draw_hud(self, state):
        players = state.get("players", [])
        num_players = len(players)
        center_x, center_y = self.target_w // 2, self.target_h // 2
        radius_x, radius_y = 500, 280 
        
        for i, p in enumerate(players):
            angle = (2 * math.pi / num_players) * i + math.pi/2
            px = center_x + int(radius_x * math.cos(angle))
            py = center_y + int(radius_y * math.sin(angle))
            
            box_w, box_h = 160, 100
            
            last_act, last_col = self.last_actions.get(p['id'], (None, None))
            is_turn = (i == state.get("current_player"))
            
            if is_turn:
                 pygame.draw.rect(self.canvas, self.YELLOW, (px - box_w//2-5, py - box_h//2-5, box_w+10, box_h+10), border_radius=12)
            elif last_col and last_act != "FOLD":
                 pygame.draw.rect(self.canvas, last_col, (px - box_w//2-3, py - box_h//2-3, box_w+6, box_h+6), border_radius=10)

            bg_color = self.WHITE
            if not p['active'] or (last_act == "FOLD"): bg_color = (180, 180, 180)
            if p['allin']: bg_color = (255, 230, 230)
            
            p_rect = pygame.Rect(px - box_w//2, py - box_h//2, box_w, box_h)
            pygame.draw.rect(self.canvas, bg_color, p_rect, border_radius=8)
            pygame.draw.rect(self.canvas, self.BLACK, p_rect, 3, border_radius=8)
            
            id_text = self.font_ui.render(f"Player {p['id']}", True, self.BLACK)
            stack_text = self.font_ui.render(f"${p['stack']:.1f}", True, self.DARK_RED)
            
            p_type = self.player_types.get(p['id'], "")
            if p_type:
                type_col = (100, 0, 200) if p_type == "AI" else (100, 100, 100)
                type_surf = self.font_small.render(p_type, True, type_col)
                self.canvas.blit(type_surf, (px + box_w//2 - 40, py - box_h//2 + 10))

            self.canvas.blit(id_text, (px - box_w//2 + 10, py - box_h//2 + 10))
            self.canvas.blit(stack_text, (px - box_w//2 + 10, py - box_h//2 + 40))
            
            if last_act:
                 act_surf = self.font_ui.render(last_act, True, last_col)
                 self.canvas.blit(act_surf, (px - box_w//2 + 10, py - box_h//2 + 70))
            
            bet_val = p.get('bet', 0)
            if bet_val > 0:
                 dir_x = (center_x - px)
                 dir_y = (center_y - py)
                 dist = math.hypot(dir_x, dir_y)
                 norm_x, norm_y = dir_x/dist, dir_y/dist
                 
                 chip_x = px + norm_x * 90
                 chip_y = py + norm_y * 90
                 # Chips are drawn in table pass, just text here? 
                 # Actually better to redraw everything or keep text?
                 # Let's draw text here
                 b_text = self.font_small.render(f"{bet_val}", True, self.WHITE)
                 b_rect = b_text.get_rect(center=(chip_x, chip_y + 20))
                 self.canvas.blit(b_text, b_rect)

    def draw(self):
        if not self.state: return
        self.canvas.fill(self.GREEN_FELT) # This will be covered by background_img if present in draw_table_and_cards
        self.draw_table_and_cards(self.state)
        self.draw_hud(self.state)
        
        # Stage Info
        stage_map = {0: "Pre-Flop", 1: "Flop", 2: "Turn", 3: "River", 5: "Showdown"}
        st = self.state.get("stage", 0)
        st_text = self.large_font.render(stage_map.get(st, ""), True, (200, 200, 200))
        self.canvas.blit(st_text, (20, 20))
        
        # Stage Indicators ... (keep existing code for indicators/winners)
        # ... copying indicator logic for brevity or reuse?
        # Let's keep it here for now as it's HUD.
        if st == 5: current_idx = 4
        else: current_idx = st
        indicator_x, indicator_y = 40, 100
        stages_labels = ["Pre", "Flop", "Turn", "River", "Show"]
        
        for i in range(5):
            if i < 4:
                pygame.draw.line(self.canvas, (100, 100, 100), (indicator_x + i*60 + 20, indicator_y), (indicator_x + (i+1)*60 - 20, indicator_y), 2)
            
            color = (100, 100, 100)
            if i <= current_idx: color = self.YELLOW
            
            pygame.draw.circle(self.canvas, color, (indicator_x + i*60, indicator_y), 12)
            if i == current_idx:
                pygame.draw.circle(self.canvas, self.WHITE, (indicator_x + i*60, indicator_y), 12, 2)
            
            lbl = self.font_small.render(stages_labels[i], True, (150, 150, 150))
            lbl_rect = lbl.get_rect(center=(indicator_x + i*60, indicator_y + 25))
            self.canvas.blit(lbl, lbl_rect)

        if self.state.get("winners"):
            winners = self.state.get("winners")
            if len(winners) == 1: w_str = f"Winner: Player {winners[0]}"
            else: w_str = "Winners: " + ", ".join([f"P{w}" for w in winners])
            
            win_surf = self.large_font.render(w_str, True, self.YELLOW)
            banner_rect = pygame.Rect(0, 150, self.target_w, 60)
            overlay = pygame.Surface((self.target_w, 60), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.canvas.blit(overlay, (0, 150))
            
            win_rect = win_surf.get_rect(center=(self.target_w//2, 180))
            self.canvas.blit(win_surf, win_rect)

        # 2. Final Output: Scale Canvas to Window
        self.render_to_screen()

    def draw_menu_background(self):
        fake_state = {
            "community_cards": ['Qh', 'Jh', 'Th', '4c', '7d'],
            "pot": 12550.0,
            "players": [
                {"id": 0, "stack": 8450, "bet": 500, "cards": ['Ah', 'Kh'], "active": True, "allin": False},
                {"id": 1, "stack": 2200, "bet": 0, "cards": ["BACK", "BACK"], "active": True, "allin": False},
                {"id": 2, "stack": 0, "bet": 0, "cards": [], "active": False, "allin": False}, # Folded
                {"id": 3, "stack": 11000, "bet": 500, "cards": ["BACK", "BACK"], "active": True, "allin": False},
                {"id": 4, "stack": 4000, "bet": 0, "cards": ["BACK", "BACK"], "active": True, "allin": False},
                {"id": 5, "stack": 200, "bet": 0, "cards": ["BACK", "BACK"], "active": True, "allin": True},
            ],
            "dealer": 2,
            "current_player": -1,
            "stage": 5,
            "winners": None
        }
        self.draw_table_and_cards(fake_state)
        
        # Add a dark overlay for the menu
        overlay = pygame.Surface((self.target_w, self.target_h), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 160)) 
        self.canvas.blit(overlay, (0, 0))

    def render_to_screen(self):
        # Get current window size
        win_w, win_h = self.screen.get_size()
        
        # Calculate Scale
        curr_aspect = win_w / win_h
        target_aspect = self.target_w / self.target_h
        
        if curr_aspect > target_aspect:
             # Window is wider than target -> Fit height, black bars on sides
             scale = win_h / self.target_h
             new_w = int(self.target_w * scale)
             new_h = win_h
             start_x = (win_w - new_w) // 2
             start_y = 0
        else:
             # Window is taller than target -> Fit width, black bars on top/bottom
             scale = win_w / self.target_w
             new_w = win_w
             new_h = int(self.target_h * scale)
             start_x = 0
             start_y = (win_h - new_h) // 2
             
        scaled_surf = pygame.transform.smoothscale(self.canvas, (new_w, new_h))
        
        self.screen.fill(self.BLACK) # Clear borders
        self.screen.blit(scaled_surf, (start_x, start_y))

    def save_screenshot(self, filename="screenshot.png"):
        pygame.image.save(self.canvas, filename)

    def draw_menu(self, options, selected_idx):
        self.draw_menu_background()

        title_surf = self.large_font.render("POKER AI SIMULATOR", True, self.YELLOW)
        title_rect = title_surf.get_rect(center=(self.target_w//2, 150))
        self.canvas.blit(title_surf, title_rect)

        start_y = 300
        for i, opt in enumerate(options):
            color = self.WHITE
            if i == selected_idx: color = self.YELLOW
            
            # Highlight box - WIDER as requested (600px instead of 400px)
            if i == selected_idx:
                btn_rect = pygame.Rect(self.target_w//2 - 300, start_y + i*80 - 15, 600, 60)
                pygame.draw.rect(self.canvas, (50, 50, 50), btn_rect, border_radius=10)
                pygame.draw.rect(self.canvas, self.YELLOW, btn_rect, 2, border_radius=10)

            text_surf = self.font_card.render(opt, True, color)
            text_rect = text_surf.get_rect(center=(self.target_w//2, start_y + i*80 + 15))
            self.canvas.blit(text_surf, text_rect)
            
        self.render_to_screen()

    def draw_timeline(self, step, total, paused):
        # NOTE: This one is tricky. It normally draws ON TOP of draw(). 
        # But draw() calls render_to_screen().
        # So we should draw this on CANVAS first, then render_to_screen?
        # But draw() is called BEFORE this in poker_app.py
        # FIX: poker_app should call draw(), then draw_timeline(), and ONLY THEN flip?
        # NO. We need robust layering.
        # Solution: draw_timeline should ALSO draw to canvas, then we call render_to_screen AGAIN? 
        # Excessive scaling.
        # Better: draw() should NOT call render_to_screen immediately if we plan to add overlay.
        # But poker_app calls them sequentially.
        # To avoid API break, draw_timeline will assume self.canvas is ready, draw on it, and then call render_to_screen() ITSELF.
        # This implies draw() should NOT call render_to_screen, OR render_to_screen is cheap? Scaling is expensive.
        # Let's Modify draw() to NOT call render_to_screen, and add a method finalize() that poker_app calls?
        # OR: Just update poker_app to call a flush method?
        # OR: draw_timeline is the last thing called. so draw_timeline does the flip?
        # Let's make draw_timeline calls render_to_screen. But draw() updates the game view.
        # If I remove render_to_screen from draw(), then if timeline is NOT called (e.g. game mode?), screen is black.
        # Let's keep render in draw(), and in draw_timeline, we draw on canvas and render AGAIN? Sluggish.
        # Best: Separate 'update_canvas' and 'render'.
        # For now, I will effectively implement: draw_timeline draws to canvas, then renders.
        # AND I will modify draw() to NOT render if we are in replay mode? No, messy.
        # I will let draw() do its thing (render). Then draw_timeline draws on canvas and renders AGAIN. 
        # It's inefficient (double scale per frame in replay) but safe codebase-wise.
        
        # Graphical Control Panel
        panel_h = 60
        panel_y = self.target_h - panel_h - 5
        panel_w = 800
        panel_x = (self.target_w - panel_w) // 2
        
        # Background Panel
        panel_surf = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        # Draw rounded rect on the transparent surface
        pygame.draw.rect(panel_surf, (30, 30, 40, 200), (0, 0, panel_w, panel_h), border_radius=15)
        self.canvas.blit(panel_surf, (panel_x, panel_y))
        
        # Border
        pygame.draw.rect(self.canvas, (60, 60, 70), (panel_x, panel_y, panel_w, panel_h), 2, border_radius=15)
        
        # Progress Bar
        bar_w = panel_w - 60
        bar_h = 6
        bar_x = panel_x + 30
        bar_y = panel_y + 40
        
        # Bar Track
        pygame.draw.rect(self.canvas, (50, 50, 50), (bar_x, bar_y, bar_w, bar_h), border_radius=3)
        
        # Bar Fill
        progress = 0
        if total > 1: progress = (step - 1) / (total - 1)
        if progress < 0: progress = 0
        if progress > 1: progress = 1
        
        fill_w = int(bar_w * progress)
        pygame.draw.rect(self.canvas, self.YELLOW, (bar_x, bar_y, fill_w, bar_h), border_radius=3)
        
        # Knob
        knob_x = bar_x + fill_w
        pygame.draw.circle(self.canvas, self.WHITE, (knob_x, bar_y + bar_h//2), 6)
        
        # Controls / Status Text
        # Left: Status
        status_str = "PAUSED" if paused else "PLAYING ►"
        st_col = (255, 100, 100) if paused else (100, 255, 100)
        st_surf = self.font_ui.render(status_str, True, st_col)
        self.canvas.blit(st_surf, (panel_x + 30, panel_y + 10))
        
        # Center: Current Step
        s_text = self.font_ui.render(f"Step: {step} / {total}", True, self.WHITE)
        s_rect = s_text.get_rect(center=(panel_x + panel_w//2, panel_y + 20))
        self.canvas.blit(s_text, s_rect)
        
        # Right: Hints
        hint_str = "[<] RW  [SPACE]  FF [>]"
        hint = self.font_small.render(hint_str, True, (150, 150, 150))
        hint_rect = hint.get_rect(topright=(panel_x + panel_w - 30, panel_y + 12))
        self.canvas.blit(hint, hint_rect)
        
        # Re-render to update screen with timeline
        self.render_to_screen()
        

