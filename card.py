import numpy as np
import settings

class Card:
    
    __slots__ = ['rank_idx', 'suit_idx']

    def __init__(self, rank_idx, suit_idx):
        self.rank_idx = rank_idx
        self.suit_idx = suit_idx

    def to_one_hot(self):
        vec = np.zeros(settings.N_CARDS, dtype=np.float32)
        idx = self.rank_idx * settings.N_SUITS + self.suit_idx
        vec[idx] = 1.0
        return vec

    def __repr__(self):
        r_str = settings.RANKS[self.rank_idx]
        s_str = settings.CHAR_SUITS[self.suit_idx]
        return f"{r_str}{s_str}"