import pygame
import sys
import math

class PokerGUI:
    def __init__(self, width=1400, height=900):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Poker AI Visualization")
        
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

        self.draw()
        
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                self.running = False
                pygame.quit()
                sys.exit()

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
                
                pygame.draw.ellipse(self.screen, fill_col, (x - 15, current_y - 10, 30, 20))
                pygame.draw.ellipse(self.screen, line_col, (x - 15, current_y - 10, 30, 20), 2)
                
                current_y -= step_y
                drawn += 1

    def draw_card(self, card_str, x, y, w=None, h=None):
        if w is None: w = self.CARD_W
        if h is None: h = self.CARD_H
        
        shadow_offset = 4
        pygame.draw.rect(self.screen, (0,0,0, 50), (x+shadow_offset, y+shadow_offset, w, h), border_radius=self.CARD_RADIUS) 
        
        pygame.draw.rect(self.screen, self.WHITE, (x, y, w, h), border_radius=self.CARD_RADIUS)
        pygame.draw.rect(self.screen, self.BLACK, (x, y, w, h), 2, border_radius=self.CARD_RADIUS)
        
        if not card_str: return
        
        if len(card_str) == 3: rank, suit = card_str[:2], card_str[2]
        else: rank, suit = card_str[0], card_str[1]
        
        color = self.RED if suit in ['h', 'd'] else self.BLACK
        suit_sym = {'h': '♥', 'd': '♦', 's': '♠', 'c': '♣'}.get(suit, suit)
        
        rank_text = self.font_card.render(f"{rank}", True, color)
        self.screen.blit(rank_text, (x + 6, y + 4))
        
        suit_text_small = self.font_small.render(suit_sym, True, color)
        self.screen.blit(suit_text_small, (x + 8, y + 45))
        
        br_surf = self.font_card.render(suit_sym, True, color)
        br_rect = br_surf.get_rect(bottomright=(x+w-6, y+h-4))
        self.screen.blit(br_surf, br_rect)

    def draw(self):
        if not self.state: return
        
        self.screen.fill(self.GREEN_FELT)
        
        table_w, table_h = 1000, 500
        table_rect = pygame.Rect((self.width - table_w)//2, (self.height - table_h)//2, table_w, table_h)
        pygame.draw.ellipse(self.screen, (40, 130, 40), table_rect) 
        pygame.draw.ellipse(self.screen, (100, 50, 20), table_rect, 15)
        
        comm_cards = self.state.get("community_cards", [])
        card_spacing = 15
        total_w = 5 * self.CARD_W + 4 * card_spacing
        start_x = (self.width - total_w) // 2
        cards_y = self.height // 2 - self.CARD_H // 2
        
        for i in range(5):
             bx = start_x + i * (self.CARD_W + card_spacing)
             pygame.draw.rect(self.screen, (35, 90, 35), (bx, cards_y, self.CARD_W, self.CARD_H), border_radius=10)
             pygame.draw.rect(self.screen, (50, 110, 50), (bx, cards_y, self.CARD_W, self.CARD_H), 2, border_radius=10)

        for i, c in enumerate(comm_cards):
            cx = start_x + i * (self.CARD_W + card_spacing)
            self.draw_card(c, cx, cards_y)
            
        pot_amount = self.state.get('pot', 0)
        pot_y = self.height // 2 + 100
        if pot_amount > 0:
             self.draw_chip_stack(pot_amount, self.width//2, pot_y)
             pot_text = self.large_font.render(f"POT: {pot_amount:.1f}", True, self.YELLOW)
             pot_rect = pot_text.get_rect(midtop=(self.width//2, pot_y + 10))
             self.screen.blit(pot_text, pot_rect)
        
        players = self.state.get("players", [])
        num_players = len(players)
        center_x, center_y = self.width // 2, self.height // 2
        radius_x, radius_y = 500, 280 
        
        for i, p in enumerate(players):
            angle = (2 * math.pi / num_players) * i + math.pi/2
            px = center_x + int(radius_x * math.cos(angle))
            py = center_y + int(radius_y * math.sin(angle))
            
            box_w, box_h = 160, 100
            
            last_act, last_col = self.last_actions.get(p['id'], (None, None))
            is_turn = (i == self.state.get("current_player"))
            
            if is_turn:
                 pygame.draw.rect(self.screen, self.YELLOW, (px - box_w//2-5, py - box_h//2-5, box_w+10, box_h+10), border_radius=12)
            elif last_col and last_act != "FOLD":
                 pygame.draw.rect(self.screen, last_col, (px - box_w//2-3, py - box_h//2-3, box_w+6, box_h+6), border_radius=10)

            bg_color = self.WHITE
            if not p['active'] or (last_act == "FOLD"): bg_color = (180, 180, 180)
            if p['allin']: bg_color = (255, 230, 230)
            
            p_rect = pygame.Rect(px - box_w//2, py - box_h//2, box_w, box_h)
            pygame.draw.rect(self.screen, bg_color, p_rect, border_radius=8)
            pygame.draw.rect(self.screen, self.BLACK, p_rect, 3, border_radius=8)
            
            id_text = self.font_ui.render(f"Player {p['id']}", True, self.BLACK)
            stack_text = self.font_ui.render(f"${p['stack']:.1f}", True, self.DARK_RED)
            
            p_type = self.player_types.get(p['id'], "")
            if p_type:
                type_col = (100, 0, 200) if p_type == "AI" else (100, 100, 100)
                type_surf = self.font_small.render(p_type, True, type_col)
                self.screen.blit(type_surf, (px + box_w//2 - 40, py - box_h//2 + 10))

            self.screen.blit(id_text, (px - box_w//2 + 10, py - box_h//2 + 10))
            self.screen.blit(id_text, (px - box_w//2 + 10, py - box_h//2 + 10))
            self.screen.blit(stack_text, (px - box_w//2 + 10, py - box_h//2 + 40))
            
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
            
            if last_act:
                 act_surf = self.font_ui.render(last_act, True, last_col)
                 self.screen.blit(act_surf, (px - box_w//2 + 10, py - box_h//2 + 70))
            
            bet_val = p.get('bet', 0)
            if bet_val > 0:
                 dir_x = (center_x - px)
                 dir_y = (center_y - py)
                 dist = math.hypot(dir_x, dir_y)
                 norm_x, norm_y = dir_x/dist, dir_y/dist
                 
                 chip_x = px + norm_x * 90
                 chip_y = py + norm_y * 90
                 
                 self.draw_chip_stack(bet_val, chip_x, chip_y)
                 
                 b_text = self.font_small.render(f"{bet_val}", True, self.WHITE)
                 b_rect = b_text.get_rect(center=(chip_x, chip_y + 20))
                 self.screen.blit(b_text, b_rect)

            if i == self.state.get("dealer"):
                db_x, db_y = px + box_w//2 - 20, py - box_h//2 - 10
                pygame.draw.circle(self.screen, self.WHITE, (db_x, db_y), 15)
                pygame.draw.circle(self.screen, self.BLACK, (db_x, db_y), 15, 2)
                d_text = self.font_ui.render("D", True, self.BLACK)
                d_rect = d_text.get_rect(center=(db_x, db_y))
                self.screen.blit(d_text, d_rect)

            cards = p.get('cards', [])
            card_scale = 0.6
            small_w = int(self.CARD_W * card_scale)
            small_h = int(self.CARD_H * card_scale)
            for ci, c in enumerate(cards):
                c_x = px - box_w//2 + 80 + ci * 25 
                c_y = py - 40 
                c_x = px - box_w//2 + 30 + ci * 35
                c_y = py + 55 
                self.draw_card(c, c_x, c_y, small_w, small_h)

        stage_map = {0: "Pre-Flop", 1: "Flop", 2: "Turn", 3: "River", 5: "Showdown"}
        st = self.state.get("stage", 0)
        
        st_text = self.large_font.render(stage_map.get(st, ""), True, (200, 200, 200))
        self.screen.blit(st_text, (20, 20))
        
        if st == 5: current_idx = 4
        else: current_idx = st
        
        indicator_x = 40
        indicator_y = 100 
        stages_labels = ["Pre", "Flop", "Turn", "River", "Show"]
        
        for i in range(5):
            if i < 4:
                pygame.draw.line(self.screen, (100, 100, 100), (indicator_x + i*60 + 20, indicator_y), (indicator_x + (i+1)*60 - 20, indicator_y), 2)
            
            color = (100, 100, 100)
            if i <= current_idx: color = self.YELLOW
            
            pygame.draw.circle(self.screen, color, (indicator_x + i*60, indicator_y), 12)
            if i == current_idx:
                pygame.draw.circle(self.screen, self.WHITE, (indicator_x + i*60, indicator_y), 12, 2)
            
            lbl = self.font_small.render(stages_labels[i], True, (150, 150, 150))
            lbl_rect = lbl.get_rect(center=(indicator_x + i*60, indicator_y + 25))
            self.screen.blit(lbl, lbl_rect)

        winners = self.state.get("winners")
        if winners:
            if len(winners) == 1: w_str = f"Player {winners[0]}"
            else: w_str = ", ".join([f"P{w}" for w in winners])
            
            win_text = self.large_font.render(f"WINNER: {w_str}", True, self.YELLOW)
            win_rect = win_text.get_rect(center=(self.width//2, self.height//2))
            
            pygame.draw.rect(self.screen, (0,0,0,220), win_rect.inflate(80, 50), border_radius=20)
            pygame.draw.rect(self.screen, self.YELLOW, win_rect.inflate(80, 50), 3, border_radius=20)
            self.screen.blit(win_text, win_rect)

        pygame.display.flip()

    def save_screenshot(self, filename="screenshot.png"):
        pygame.image.save(self.screen, filename)
