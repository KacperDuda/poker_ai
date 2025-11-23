import numpy as np
from card import Card
import settings
from deck import Deck
from player import Player
import evaluator  # <--- IMPORTUJEMY LOGIKĘ DO ŚRODOWISKA

class PokerEnv:
    def __init__(self, players_data, initial_stack=1000.0, sb=settings.SB, bb=settings.BB):
        if not isinstance(players_data, list) or len(players_data) < 2:
            raise ValueError("players_data musi być listą z minimum 2 graczami.")

        self.sb_amount = float(sb)
        self.bb_amount = float(bb)
        
        self.players = []
        self.starting_stacks = {}

        for item in players_data:
            if isinstance(item, Player):
                self.players.append(item)
                self.starting_stacks[item.id] = item.stack
            elif isinstance(item, int):
                new_player = Player(item, initial_stack)
                self.players.append(new_player)
                self.starting_stacks[item] = float(initial_stack)
            else:
                raise ValueError(f"Player type unsupported: {type(item)}")
        
        self.num_players = len(self.players)
        self.player_ids = [p.id for p in self.players]

        self.deck = Deck()
        self.community_cards: list[Card] = []
        self.pot = 0.0
        self.stage = 0 
        self.dealer_pos = -1 
        self.current_player_idx = 0
        self.min_raise = 0.0

        self.obs_dim: int = (
            settings.N_CARDS + settings.N_CARDS +
            ((settings.MAX_SEATS - 1) * 4 + 3)
        )

    # --- NOWE METODY DO OBSŁUGI WYGRANEJ ---

    def _distribute_pot(self, winners_ids):
        """
        Prywatna metoda: Dzieli pulę równo między zwycięzców i aktualizuje ich stacki.
        Zeruje pulę na końcu.
        """
        if not winners_ids:
            return

        split_amount = self.pot / len(winners_ids)
        
        for w_id in winners_ids:
            # Znajdź obiekt gracza (zakładamy, że ID są spójne z indeksami lub szukamy)
            # W naszej prostej implementacji self.players[i].id == i zazwyczaj, 
            # ale bezpieczniej jest poszukać
            winner = next((p for p in self.players if p.id == w_id), None)
            if winner:
                winner.stack += split_amount
        
        # Reset puli po wypłacie
        self.pot = 0.0

    def finalize_showdown(self):
        """
        Publiczna metoda wywoływana na końcu Rivera.
        Automatycznie sprawdza karty, wyłania zwycięzcę i aktualizuje stacki.
        Zwraca listę ID zwycięzców.
        """
        # 1. Użyj evaluator, aby znaleźć najlepsze układy
        winners_ids = evaluator.determine_winner(self)
        
        # 2. Wypłać pieniądze
        self._distribute_pot(winners_ids)
        
        return winners_ids

    # --- ZMODYFIKOWANY STEP ---

    def step(self, action_type, action_amt_pct=0.0):
        """
        Wykonuje ruch i AUTOMATYCZNIE sprawdza, czy gra się skończyła przez 'Walkower'.
        Zwraca: next_obs (tutaj None/Dummy), reward (0), done (True/False), info
        """
        player = self.players[self.current_player_idx]
        current_max_bet = max(p.current_bet for p in self.players)
        to_call = current_max_bet - player.current_bet

        # --- LOGIKA RUCHU (Fold/Call/Raise) ---
        if action_type == 0: # FOLD
            if to_call == 0: 
                action_type = 1 
            else: 
                player.is_active = False

        if action_type == 1: # CALL / CHECK
            amt = min(player.stack, to_call)
            self._post_bet(player, amt)

        elif action_type == 2: # RAISE
            target_min_bet = current_max_bet + self.min_raise
            target_max_bet = player.current_bet + player.stack

            if target_max_bet <= target_min_bet: 
                amt = player.stack
                self._post_bet(player, amt) 
            else:
                amount_to_add_min = target_min_bet - player.current_bet
                amount_to_add_max = player.stack
                amt = amount_to_add_min + (amount_to_add_max - amount_to_add_min) * action_amt_pct
                amt = float(int(amt))

                new_raise_size = (player.current_bet + amt) - current_max_bet
                if new_raise_size > self.min_raise:
                    self.min_raise = new_raise_size
                self._post_bet(player, amt)

        # --- AUTOMATYCZNE WYKRYWANIE KOŃCA (WALKOWER) ---
        active_players = [p for p in self.players if p.is_active]
        
        if len(active_players) == 1:
            # Wszyscy inni spasowali!
            winner = active_players[0]
            self._distribute_pot([winner.id])
            
            # Zwracamy done=True
            return 0, True, {"winner": winner.id, "method": "walkower"}

        # Przesunięcie wskaźnika
        self._next_active_player()

        # Gra toczy się dalej
        return 0, False, {}

    # --- RESZTA METOD BEZ ZMIAN ---
    
    def get_current_player_idx(self):
        return self.current_player_idx

    def get_observation(self, player_idx):
        hero = self.players[player_idx]
        hero_vec = np.zeros(settings.N_CARDS, dtype=np.float32)
        for c in hero.hand: hero_vec += c.to_one_hot()
        
        board_vec = np.zeros(settings.N_CARDS, dtype=np.float32)
        for c in self.community_cards: board_vec += c.to_one_hot()
        
        opp_vec = []
        for i in range(1, settings.MAX_SEATS):
            seat_idx = (player_idx + i) % self.num_players
            if seat_idx >= self.num_players:
                opp_vec.extend([0.0]*4)
                continue
            opp = self.players[seat_idx]
            opp_stats = [
                opp.stack / settings.MAX_STACK,
                opp.current_bet / settings.MAX_STACK,
                1.0 if opp.is_active else 0.0,
                1.0 if seat_idx == self.dealer_pos else 0.0
            ]
            opp_vec.extend(opp_stats)
            
        glob_vec = [
            hero.stack / settings.MAX_STACK, 
            self.pot / settings.MAX_STACK, 
            hero.current_bet / settings.MAX_STACK
        ]
        return np.concatenate([hero_vec, board_vec, np.array(opp_vec, dtype=np.float32), np.array(glob_vec, dtype=np.float32)])

    def reset(self):
        self.dealer_pos = -1
        for p in self.players:
            p.stack = self.starting_stacks[p.id]
            p.reset_for_hand()
        self.start_round()
        return None

    def start_round(self):
        self.deck = Deck()
        self.community_cards = []
        self.pot = 0.0
        self.stage = 0
        self.min_raise = self.bb_amount

        active_count = 0
        for p in self.players:
            p.reset_for_hand()
            if p.stack > 0.01: 
                p.is_active = True
                active_count += 1
            else:
                p.is_active = False

        if active_count < 2:
            print("Game Over! Not enough players.")
            return

        if self.dealer_pos == -1:
            self.dealer_pos = 0 if self.players[0].is_active else self._get_next_active_seat(0)
        else:
            self.dealer_pos = self._get_next_active_seat(self.dealer_pos)

        for p in self.players:
            if p.is_active:
                p.hand = self.deck.draw(2)

        sb_pos, bb_pos, first_action_idx = self._calculate_positions(active_count)
        self._post_blind(self.players[sb_pos], self.sb_amount)
        self._post_blind(self.players[bb_pos], self.bb_amount)

        self.current_player_idx = first_action_idx
        self._ensure_active_player()

    def deal_next_stage(self):
        if self.stage == 0: self.community_cards.extend(self.deck.draw(3)); self.stage = 1
        elif self.stage == 1: self.community_cards.extend(self.deck.draw(1)); self.stage = 2
        elif self.stage == 2: self.community_cards.extend(self.deck.draw(1)); self.stage = 3
        elif self.stage >= 3: return

        for p in self.players:
            p.current_bet = 0.0
        self.min_raise = self.bb_amount
        self.current_player_idx = self._get_next_active_seat(self.dealer_pos)
        self._ensure_active_player()

    def _get_next_active_seat(self, current_seat):
        for i in range(1, self.num_players):
            idx = (current_seat + i) % self.num_players
            if self.players[idx].is_active: return idx
        return current_seat

    def _post_bet(self, player, amount):
        player.stack -= amount
        player.current_bet += amount
        self.pot += amount
        if player.stack <= 0.01: player.is_allin = True

    def _calculate_positions(self, active_count):
        if active_count == 2:
            sb_pos = self.dealer_pos
            bb_pos = self._get_next_active_seat(sb_pos)
            first_action = sb_pos
        else:
            sb_pos = self._get_next_active_seat(self.dealer_pos)
            bb_pos = self._get_next_active_seat(sb_pos)
            first_action = self._get_next_active_seat(bb_pos)
        return sb_pos, bb_pos, first_action

    def _post_blind(self, player, amount):
        if not player.is_active: return
        actual = min(player.stack, amount)
        player.stack -= actual
        player.current_bet += actual
        self.pot += actual
        if player.stack <= 0.01: player.is_allin = True

    def _next_active_player(self):
        start = self.current_player_idx
        while True:
            self.current_player_idx = (self.current_player_idx + 1) % self.num_players
            p = self.players[self.current_player_idx]
            if p.is_active and not p.is_allin and p.stack > 0: break 
            if self.current_player_idx == start: break 

    def _ensure_active_player(self):
        p = self.players[self.current_player_idx]
        if not p.is_active or p.is_allin: self._next_active_player()

    def render(self):
        print("\n" + "-"*50)
        print(f"BOARD: {self.community_cards} | POT: {self.pot:.1f} | MinRaise: {self.min_raise}")
        for i, p in enumerate(self.players):
            marker = "D" if i == self.dealer_pos else " "
            act = "<--" if i == self.current_player_idx else ""
            status = ""
            if not p.is_active: status = "[OUT]"
            elif p.is_allin: status = "[ALLIN]"
            print(f"{marker} {p} {status} {act}")
        print("-" * 50)