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

        # --- CHANGE OBS_DIM (Now even larger) ---
        # 52 (Hero) + 52 (Board) + Opponents + Global 
        # + 9 (Hand Category One-Hot) 
        # + 5 (Exact Card Ranks Normalized) <--- NEW
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

    def _distribute_pot(self, winners_ids=None) -> None:
        """
        Distributes the pot correctly handling side-pots for all-in players.
        Uses 'total_wagered' to determine eligibility.
        """
        # 1. Identify all candidates (Active + All-in at showdown)
        # Note: self.players includes everyone. We filter for those who put money in.
        
        candidates = [p for p in self.players if p.total_wagered > 0 and (p.is_active or p.is_allin)]
        
        if not candidates:
            # Should not happen, but if so, return pot to... the house? or button?
            return 

        # 2. Sort unique wager amounts to create pot "levels"
        # We process the pot from bottom up. 
        # e.g. P1(100), P2(1000), P3(1000). unique levels: [100, 1000]
        # Level 1 (0-100): P1, P2, P3 compete. Pot size = 100*3 = 300. P1 Wins.
        # Level 2 (100-1000): P2, P3 compete. Pot size = 900*2 = 1800. P2 Wins.
        
        unique_wagers = sorted(list(set(p.total_wagered for p in candidates)))
        
        current_level_base = 0.0
        
        for level_cap in unique_wagers:
            contribution_from_this_level = level_cap - current_level_base
            if contribution_from_this_level <= 0.001: 
                continue
                
            # Who contributed to this level?
            contributors = [p for p in candidates if p.total_wagered >= level_cap]
            if not contributors: 
                continue
            
            # Pot size for this slice
            # Everyone who wagered 'level_cap' or more contributes 'contribution_from_this_level'
            # (Note: players who wagered LESS than level_cap but MORE than current_level_base 
            # are also mathematically part of a lower side pot, but our `unique_wagers` iteration handles valid caps.
            # Wait, players with wages BETWEEN levels? unique_wagers covers ALL unique amounts.
            # So if P4 wagered 500 (between 100 and 1000), 500 would be in the list.
            
            pot_slice = 0.0
            eligible_for_win = []
            
            for p in self.players: # iterate all original players to catch folded money too?
                 # Even folded players contributed to the pot at this level!
                 # But they are NOT eligible to win.
                 # Contribution is: min(max(0, p.total_wagered - current_level_base), contribution_from_this_level)
                 
                 effective_contribution = min(max(0.0, p.total_wagered - current_level_base), contribution_from_this_level)
                 pot_slice += effective_contribution
                 
                 # Eligibility check: Must be in 'candidates' (active/allin) AND have wagered enough
                 if p in candidates and p.total_wagered >= level_cap:
                     eligible_for_win.append(p)

            if pot_slice < 0.01:
                current_level_base = level_cap
                continue

            # Determine winner for this slice
            # We need to re-evaluate best hand among ONLY 'eligible_for_win'
            if not eligible_for_win:
                # Everyone folded who reached this level? Refund to highest contributor?
                # In PokerEnv, unexpected. Give to dealer/active.
                # Just leave in pot? No, distribute to someone.
                # Assuming one player usually wins walkover before this.
                pass
            else:
                 # Evaluate hands
                 best_rank = -1
                 slice_winners = []
                 
                 # If we have pre-calculated winners_ids (main pot), we can't reuse blindly because side pot might exclude main winner.
                 # So we evaluate locally.
                 
                 player_scores = []
                 for p in eligible_for_win:
                     score = evaluator.get_best_hand(p.hand, self.community_cards)
                     player_scores.append((p, score))
                 
                 # Sort by score (descending, assuming higher tuple is better)
                 # evaluator returns (rank_int, kickers_tuple). Python tuple comparison works nicely.
                 # rank_int: 8 (StrFlush) > ... > 0 (HighCard).
                 
                 player_scores.sort(key=lambda x: x[1], reverse=True)
                 
                 best_score = player_scores[0][1]
                 slice_winners = [p for p, s in player_scores if s == best_score]
                 
                 # Distribute 'pot_slice' equally among 'slice_winners'
                 share = pot_slice / len(slice_winners)
                 for w in slice_winners:
                     w.stack += share
                     
            current_level_base = level_cap

        # Reset global pot and wagered
        self.pot = 0.0
        for p in self.players:
            p.total_wagered = 0.0

    def finalize_showdown(self):
        self.community_cards.extend(self.deck.draw(5 - len(self.community_cards)))
        self.stage = 5
        # winners_ids = evaluator.determine_winner(self) # Deprecated for distribution, used for logging only
        # We let _distribute_pot calculate per-side-pot winners.
        
        # For logging, we'll just log the "Survivor" or best overall hand
        active = [p for p in self.players if p.is_active or p.is_allin]
        best_p = active[0].id if active else -1
        
        self._distribute_pot() 

        self._notify("showdown", {"winners": "calculated_per_pot"})
        return []

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
                
                # new_raise_size = (player.current_bet + amt) - current_max_bet

                self._post_bet(player, amt)

        self._notify("step", {"player_id": player.id, "action": action_type, "amount_pct": action_amt_pct})

        active_players = [p for p in self.players if p.is_active]
        if len(active_players) == 1: 
            # Walkover - check if anyone is all-in?
            # If others are all-in, we MUST Showdown.
            # If no one is all-in (just folded), then it is a clean walkover.
            allin_players = [p for p in self.players if p.is_allin and p.is_active] # is_active is usually True for allins in this env? 
            # implementation detail: usually all-in flag is separate. p.is_active might be true or false.
            # Let's check lines 269/289: when allin, is_active stays True.
            
            # Correct logic:
            # If remaining active players + allin players > 1, we continue (to showdown or next street)
            # If only 1 player has chips and cards, and others are Folded (not allin), he wins pot.
            
            contestants = [p for p in self.players if (p.is_active or p.is_allin) and p.stack >= 0] 
            # Actually simplest check:
            # If everyone else FOLDED, then 1 winner.
            # If someone is All-IN, they haven't folded.
            
            not_folded = [p for p in self.players if p.is_active or p.is_allin] # Assuming Folds set is_active=False AND is_allin=False?
            # Look at FOLD logic: player.is_active = False. is_allin remains?
            # We need to ensure FOLD clears is_allin if it was set? No, you can't fold if you are all-in usually (unless valid action).
            # But here standard FOLD sets is_active=False.
            
            if len(not_folded) == 1:
                winner = not_folded[0]
                self._distribute_pot() # Will identify just one candidate
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