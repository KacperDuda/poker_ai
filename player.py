from typing import Any

from card import Card


class Player:
    def __init__(self, player_id, initial_stack):
        self.id = player_id
        self.stack = float(initial_stack)
        self.hand: list[Card] = []
        self.current_bet: float = 0.0
        self.total_wagered: float = 0.0
        self.is_active = True
        self.is_allin = False

    def reset_for_hand(self):
        self.hand = []
        self.current_bet = 0.0
        self.total_wagered = 0.0
        self.is_active = True if self.stack > 0.05 else False
        self.is_allin = False

    def __repr__(self) -> str:
        status = ""
        if not self.is_active: status = "[FOLD]"
        elif self.is_allin: status = "[ALL-IN]"

        hand_str = str(self.hand) if self.hand else "[]"
        return f"P{self.id} {hand_str:<10}|{self.stack:>6.0f} $ |{self.total_wagered:>7.0f} $ |{self.current_bet:>7.0f} $ {status:>11}"
        