
import itertools
from collections import Counter

def evaluate_5_cards(cards) -> tuple[int, tuple[int, ...]]:
    ranks = sorted([c.rank_idx for c in cards], reverse=True)
    suits = [c.suit_idx for c in cards]
    
    is_flush = len(set(suits)) == 1
    
    is_straight = (max(ranks) - min(ranks) == 4) and (len(set(ranks)) == 5)
    
    if set(ranks) == {12, 3, 2, 1, 0}:
        is_straight = True
        ranks = [3, 2, 1, 0] 

    ranks_tuple = tuple(ranks)

    if is_straight and is_flush:
        return (8, ranks_tuple)

    counts = Counter(ranks)
    sorted_counts = sorted(counts.items(), key=lambda x: (x[1], x[0]), reverse=True)
    
    score_ranks = tuple([r for r, count in sorted_counts])

    if sorted_counts[0][1] == 4:
        return (7, score_ranks)
    
    if sorted_counts[0][1] == 3 and sorted_counts[1][1] == 2:
        return (6, score_ranks)
    
    if is_flush:
        return (5, ranks_tuple)
    
    if is_straight:
        return (4, ranks_tuple) 
    
    if sorted_counts[0][1] == 3:
        return (3, score_ranks)
    
    if sorted_counts[0][1] == 2 and sorted_counts[1][1] == 2:
        return (2, score_ranks)
    
    if sorted_counts[0][1] == 2:
        return (1, score_ranks)
    
    return (0, ranks_tuple)

def get_best_hand(hole_cards, community_cards):
    all_cards = hole_cards + community_cards
    best_score = (-1, ())
    
    for combo in itertools.combinations(all_cards, 5):
        score = evaluate_5_cards(combo)
        if score > best_score:
            best_score = score
            
    return best_score

def determine_winner(env):
    active_players = [p for p in env.players if p.is_active]
    
    if len(active_players) == 1:
        return [active_players[0].id]
    
    best_score = (-1, ())
    winners = []
    
    for p in active_players:
        score = get_best_hand(p.hand, env.community_cards)
        
        if score > best_score:
            best_score = score
            winners = [p.id]
        elif score == best_score:
            winners.append(p.id) 
            
    return winners

def get_hand_name(score_tuple):
    names = {
        0: "High Card",
        1: "Pair",
        2: "Two Pair",
        3: "Three of a Kind",
        4: "Straight",
        5: "Flush",
        6: "Full House",
        7: "Four of a Kind",
        8: "Straight Flush"
    }
    return names.get(score_tuple[0], "Unknown")