import itertools
from collections import Counter

# Hand hierarchy (higher is better)
# 8: Straight Flush
# 7: Four of a Kind
# 6: Full House
# 5: Flush
# 4: Straight
# 3: Three of a Kind
# 2: Two Pair
# 1: Pair
# 0: High Card

def evaluate_5_cards(cards) -> tuple[int, tuple[int, ...]]:
    """
    Evaluates a 5-card hand.
    Returns a tuple: (hand_ranking, tuple_of_kickers_descending)
    Using tuples fixes Pyright error and facilitates comparison.
    """
    # Get ranks and suits
    ranks = sorted([c.rank_idx for c in cards], reverse=True)
    suits = [c.suit_idx for c in cards]
    
    # Check for flush
    is_flush = len(set(suits)) == 1
    
    # Check for straight
    # Normal straight: difference between max and min is 4, and 5 unique cards
    is_straight = (max(ranks) - min(ranks) == 4) and (len(set(ranks)) == 5)
    
    # Special case: "Wheel" Straight A-2-3-4-5 (Ranks: 12, 3, 2, 1, 0)
    # This might not work for shorthand, BUT for shorthand there will be a completely different function to resolve the tie :)
    if set(ranks) == {12, 3, 2, 1, 0}:
        is_straight = True
        ranks = [3, 2, 1, 0] # In the Wheel, 5 is the highest (rank 3)

    # Convert ranks to tuple (for comparison and typing)
    ranks_tuple = tuple(ranks)

    # 1. Straight Flush
    if is_straight and is_flush:
        return (8, ranks_tuple)

    # Rank counter (for quads, full house, pairs)
    counts = Counter(ranks)
    # Sort: first by count (descending), then by rank (descending)
    sorted_counts = sorted(counts.items(), key=lambda x: (x[1], x[0]), reverse=True)
    
    # Ranks sorted by importance - immediately convert to tuple
    score_ranks = tuple([r for r, count in sorted_counts])

    # 2. Four of a Kind (4 repetitions)
    if sorted_counts[0][1] == 4:
        return (7, score_ranks)
    
    # 3. Full House (3 + 2 repetitions)
    if sorted_counts[0][1] == 3 and sorted_counts[1][1] == 2:
        return (6, score_ranks)
    
    # 4. Flush
    if is_flush:
        return (5, ranks_tuple)
    
    # 5. Straight
    if is_straight:
        return (4, ranks_tuple) # ranks_tuple already includes the Wheel correction
    
    # 6. Three of a Kind (3 repetitions)
    if sorted_counts[0][1] == 3:
        return (3, score_ranks)
    
    # 7. Two Pair (2 + 2 repetitions)
    if sorted_counts[0][1] == 2 and sorted_counts[1][1] == 2:
        return (2, score_ranks)
    
    # 8. Pair (2 repetitions)
    if sorted_counts[0][1] == 2:
        return (1, score_ranks)
    
    # 9. High Card
    return (0, ranks_tuple)

def get_best_hand(hole_cards, community_cards):
    """
    Finds the best 5-card hand from the available 7 cards (2 hole + 5 board).
    """
    all_cards = hole_cards + community_cards
    best_score = (-1, ())
    
    # Iterate through all combinations of 5 cards from 7 (there are 21)
    for combo in itertools.combinations(all_cards, 5):
        score = evaluate_5_cards(combo)
        if score > best_score:
            best_score = score
            
    return best_score

def determine_winner(env):
    """
    Main function to determine the winner in the PokerEnv environment.
    Returns a list of winner indices (usually one, but possible tie/split pot).
    """
    active_players = [p for p in env.players if p.is_active]
    
    # Safety check: If only one player remains (everyone else folded earlier)
    if len(active_players) == 1:
        return [active_players[0].id]
    
    best_score = (-1, ())
    winners = []
    
    # Evaluate each active player
    for p in active_players:
        # Player must use their cards + community cards
        # If the board is not full (e.g., simulation ends early), 
        # it evaluates what's there, but this is mainly relevant on the River.
        score = get_best_hand(p.hand, env.community_cards)
        
        # Debug (optional)
        # print(f"Player {p.id} Score: {score}")

        if score > best_score:
            best_score = score
            winners = [p.id]
        elif score == best_score:
            winners.append(p.id) # Tie (Split Pot)
            
    return winners

def get_hand_name(score_tuple):
    """Helper function to convert score (int) to a text name"""
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