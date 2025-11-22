RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
SUITS = ["Spades", "Hearts", "Diamonds", "Clubs"]
CHAR_SUITS = ['s', 'h', 'd', 'c']

N_RANKS = len(RANKS)
N_SUITS = len(SUITS)
N_CARDS = N_RANKS * N_SUITS

MAX_SEATS = 6
MAX_STACK = 2000.0 # do normalizacji 0-1

SB = 10.0
BB = 20.0