import random
import settings
from card import Card

class Deck:
    def  __init__(self):
        self.cards = [
            Card(r, s)
            for r in range(settings.N_RANKS)
            for s in range(settings.N_SUITS)
        ]
        
        self.shuffle()

    def shuffle(self):
        random.shuffle(self.cards)

    def draw(self, n=1):
        drawn = []
        for _ in range(n):
            drawn.append(self.cards.pop())
        return drawn

    def __repr__(self):
        return ' '.join([str(c) for c in self.cards])