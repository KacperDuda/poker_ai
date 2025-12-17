from numpy._typing._array_like import NDArray
from numpy import floating
from numpy._typing._nbit_base import _32Bit
from typing import Any, Literal, Self

import numpy as np
from card import Card
import settings
from deck import Deck
from player import Player
import evaluator

class PokerEnv:
    def __init__(self, players_data, initial_stack=1000.0, sb=settings.SB, bb=settings.BB, chip_step=5.0, debug = False) -> None:
        if not isinstance(players_data, list) or len(players_data) < 2:
            raise ValueError("players_data must be a list with at least 2 players.")

        self.sb_amount: float = float(sb)
        self.bb_amount: float = float(bb)
        self.chip_step: float = float(chip_step)
        self.debug: bool = debug
        self.observers = []  # List of callbacks for UI/Logging
        
        self.players: list[Player] = []
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
        
        self.num_players: int = len(self.players)
        self.player_ids: list[int] = [p.id for p in self.players]

        self.deck: Deck = Deck()
        self.community_cards: list[Card] = []
        self.pot: float = 0.0
        self.stage: int = 0 
        self.dealer_pos: int = -1 
        self.current_player_idx: int = 0
        self.min_raise: float = 0.0

        # --- ZMIANA OBS_DIM (Teraz jeszcze większy) ---
        # 52 (Hero) + 52 (Board) + Opponents + Global 
        # + 9 (Hand Category One-Hot) 
        # + 5 (Exact Card Ranks Normalized) <--- NOWOŚĆ
        self.obs_dim: int = (
            settings.N_CARDS + settings.N_CARDS +
            ((settings.MAX_SEATS - 1) * 4 + 3) + 
            9 + 
            5 
        )

    def add_observer(self, callback):
        """Register a callback (func) that accepts (event_name, data_dict)."""
        self.observers.append(callback)

    def _notify(self, event: str, data: dict = None):
        if data is None: data = {}
        # Snapshot basic state for the UI
        state_snapshot = {
            "pot": self.pot,
            "community_cards": [str(c) for c in self.community_cards],
            "stage": self.stage,
            "current_player": self.current_player_idx,
            "dealer": self.dealer_pos,
            "players": [
                {
                    "id": p.id,
                    "stack": p.stack,
                    "bet": p.current_bet,
                    "active": p.is_active,
                    "allin": p.is_allin,
                    "cards": [str(c) for c in p.hand] if hasattr(p, 'hand') else []
                }
                for p in self.players
            ]
        }
        full_data = {**state_snapshot, **data}
        for obs in self.observers:
            obs(event, full_data)
    def _round_to_chip(self, amount) -> float:
        if self.chip_step <= 0: return amount
        steps: int = round(amount / self.chip_step)
        return steps * self.chip_step

    def _distribute_pot(self, winners_ids) -> None:
        if not winners_ids: return
        split_amount = self.pot / len(winners_ids)
        for w_id in winners_ids:
            winner = next((p for p in self.players if p.id == w_id), None)
            if winner: winner.stack += split_amount
        self.pot = 0.0

    def finalize_showdown(self):
        self.community_cards.extend(self.deck.draw(5 - len(self.community_cards)))
        self.stage = 5
        winners_ids = evaluator.determine_winner(self)
        self._distribute_pot(winners_ids)

        self._notify("showdown", {"winners": winners_ids})
        return winners_ids

    def step(self, action_type, action_amt_pct=0.0):
        player: Player = self.players[self.current_player_idx]
        current_max_bet: float = max(p.current_bet for p in self.players)
        to_call: float = current_max_bet - player.current_bet 

        if action_type == 0: # FOLD
            player.is_active = False

        if action_type == 1: # CALL / CHECK
            amt: float = min(player.stack, to_call)
            self._post_bet(player, amt)

        elif action_type == 2: # RAISE
            target_min_bet = current_max_bet + self.min_raise
            target_max_bet = player.current_bet + player.stack 

            if target_max_bet <= target_min_bet: 
                self._post_bet(player, player.stack)
            else:
                amount_to_add_min = target_min_bet - player.current_bet
                amount_to_add_max = player.stack
                raw_amt = amount_to_add_min + (amount_to_add_max - amount_to_add_min) * action_amt_pct
                amt = self._round_to_chip(raw_amt)
                
                new_raise_size = (player.current_bet + amt) - current_max_bet


                self._post_bet(player, amt)

        self._notify("step", {"player_id": player.id, "action": action_type, "amount_pct": action_amt_pct})

        active_players = [p for p in self.players if p.is_active]
        if len(active_players) == 1: 
            winner = active_players[0]
            self._distribute_pot([winner.id])
            return 0, True, {"winner": winner.id, "method": "walkover"}

        self._next_active_player()
        return 0, False, {}

    def get_current_player_idx(self):
        return self.current_player_idx

    def get_observation(self, player_idx) -> NDArray[floating[_32Bit]]:
        hero: Player = self.players[player_idx]
        
        # 1. Hero Cards (One-Hot)
        hero_vec = np.zeros(settings.N_CARDS, dtype=np.float32)
        for c in hero.hand: hero_vec += c.to_one_hot()
        
        # 2. Board Cards (One-Hot)
        board_vec = np.zeros(settings.N_CARDS, dtype=np.float32)
        for c in self.community_cards: board_vec += c.to_one_hot()
        
        # 3. Opponents Stats
        opp_vec: list[Any] = []
        for i in range(1, settings.MAX_SEATS):
            seat_idx: int = (player_idx + i) % self.num_players
            opp = self.players[seat_idx]
            opp_stats = [
                opp.stack / settings.MAX_STACK,
                opp.current_bet / settings.MAX_STACK,
                1.0 if opp.is_active else 0.0,
                1.0 if seat_idx == self.dealer_pos else 0.0
            ]
            opp_vec.extend(opp_stats)
            
        # 4. Global Stats
        glob_vec = [
            hero.stack / settings.MAX_STACK, 
            self.pot / settings.MAX_STACK, 
            hero.current_bet / settings.MAX_STACK
        ]

        # 5. FEATURE ENGINEERING: Hand Strength + Kickers
        # Evaluator returns: (score_int, (rank1, rank2, rank3, rank4, rank5))
        hand_score_tuple = evaluator.get_best_hand(hero.hand, self.community_cards)
        
        # A. Hand Category One-Hot (0-8)
        hand_rank_idx = hand_score_tuple[0]
        hand_strength_vec = np.zeros(9, dtype=np.float32)
        hand_strength_vec[hand_rank_idx] = 1.0
        
        # B. Exact Card Ranks (Normalized 0-1)
        # Ranks are 0 (Two) to 12 (Ace). We divide by 12.0.
        ranks_tuple = hand_score_tuple[1]
        ranks_vec = np.zeros(5, dtype=np.float32)
        
        # Fill ranks (sometimes tuple is smaller than 5 if not enough cards, though get_best_hand handles 5)
        for i, r in enumerate(ranks_tuple):
            if i < 5:
                ranks_vec[i] = r / 12.0 # Normalize 0.0 to 1.0
        
        # Concatenate EVERYTHING
        return np.concatenate([
            hero_vec, 
            board_vec, 
            np.array(opp_vec, dtype=np.float32), 
            np.array(glob_vec, dtype=np.float32),
            hand_strength_vec,
            ranks_vec # Adding the kickers!
        ])

    def reset(self):
        self.dealer_pos = -1
        for p in self.players:
            p.stack = self.starting_stacks[p.id]
            p.reset_for_hand()
        self.start_round()

    def start_round(self) -> bool:
        self.deck = Deck()
        self.community_cards = []
        self.pot = 0.0
        self.stage = 0
        self.min_raise = self.bb_amount

        for p in self.players: p.reset_for_hand()

        active_count = sum(p.is_active for p in self.players)
        if active_count < 2:
            if self.debug: print("Game Over! Not enough players.")
            return False

        self.dealer_pos = self._get_next_active_seat(self.dealer_pos)
        for p in self.players: p.hand = self.deck.draw(2) if p.is_active else []

        sb_pos, bb_pos, first_action_idx = self._calculate_positions(active_count)
        self._post_blind(self.players[sb_pos], self.sb_amount)
        self._post_blind(self.players[bb_pos], self.bb_amount)

        self.current_player_idx = first_action_idx
        self._ensure_active_player()

        self._notify("new_round", {})
        return True

    def deal_next_stage(self):
        if self.stage == 0: self.community_cards.extend(self.deck.draw(3)); self.stage = 1
        elif self.stage == 1: self.community_cards.extend(self.deck.draw(1)); self.stage = 2
        elif self.stage == 2: self.community_cards.extend(self.deck.draw(1)); self.stage = 3
        elif self.stage >= 3: return

        for p in self.players: p.current_bet = 0.0
        self.min_raise = self.bb_amount
        self.current_player_idx = self._get_next_active_seat(self.dealer_pos) 

        self._notify("stage_change", {"new_stage": self.stage})

    def _get_next_active_seat(self, current_seat):
        for i in range(1, self.num_players):
            idx = (current_seat + i) % self.num_players
            if self.players[idx].is_active: return idx
        return current_seat

    def _post_bet(self, player: Player, amount: float):
        player.stack -= amount
        player.current_bet += amount
        player.total_wagered += amount
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

    def _post_blind(self, player: Player, amount):
        if not player.is_active: return
        actual = min(player.stack, amount)
        player.stack -= actual
        player.total_wagered += actual
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

    def __repr__(self) -> str:
        ret = "\n" + "-"*67 + "\n"
        ret += f"BOARD: {self.community_cards} | POT: {self.pot:.1f} $ | MinRaise: {self.min_raise}\n" 
        for i, p in enumerate(self.players):
            marker = "D" if i == self.dealer_pos else " "
            act = "<--" if i == self.current_player_idx else ""
            ret += f"{marker} {p} {act}\n"
        ret += "-" * 67
        return ret
    
    def _check_end_of_betting_round(self) -> bool:
        active_players = [p for p in self.players if p.is_active and not p.is_allin]
        if len(active_players) <= 1: return True 
        max_bet = max(p.current_bet for p in self.players)
        all_active_equal = all(p.current_bet == max_bet for p in self.players if p.is_active)
        next_player = self.players[self.current_player_idx]
        to_call = max_bet - next_player.current_bet
        if to_call <= 0.01 and all_active_equal: return True
        return False