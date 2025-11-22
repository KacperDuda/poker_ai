class Player:
    def __init__(self, player_id, initial_stack):
        self.id = player_id
        self.stack = float(initial_stack)
        self.hand = []
        self.current_bet = 0.0
        self.is_active = True
        self.is_allin = False

    def reset_for_hand(self):
        self.hand = []
        self.current_bet = 0.0
        self.is_active = True if self.stack > 0 else False
        self.is_allin = False

    def __repr__(self):
        status = ""
        if not self.is_active: status = "[FOLD]"
        elif self.is_allin: status = "[ALL-IN]"

        hand_str = str(self.hand) if self.hand else "[]"
        return f"P{self.id} {hand_str:<10} ${self.stack:<6.0f} (+${self.current_bet}) {status}"
        