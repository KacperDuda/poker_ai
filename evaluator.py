from typing import Any


import itertools
from collections import Counter

# Hierarchia układów (im wyżej tym lepiej)
# 8: Straight Flush (Poker)
# 7: Four of a Kind (Kareta)
# 6: Full House (Full)
# 5: Flush (Kolor)
# 4: Straight (Strit)
# 3: Three of a Kind (Trójka)
# 2: Two Pair (Dwie Pary)
# 1: Pair (Para)
# 0: High Card (Wysoka Karta)

def evaluate_5_cards(cards) -> tuple[int, tuple[int, ...]]:
    """
    Ocenia układ 5 kart.
    Zwraca krotkę: (ranking_układu, krotka_kickrów_malejąco)
    Użycie krotek (tuples) naprawia błąd Pyright i ułatwia porównywanie.
    """
    # Pobieramy rangi i kolory
    ranks = sorted([c.rank_idx for c in cards], reverse=True)
    suits = [c.suit_idx for c in cards]
    
    # Sprawdzamy kolor
    is_flush = len(set(suits)) == 1
    
    # Sprawdzanie strita
    # Normalny strit: różnica między max a min to 4 oraz 5 unikalnych kart
    is_straight = (max(ranks) - min(ranks) == 4) and (len(set(ranks)) == 5)
    
    # Specjalny przypadek: Strit "Wheel" A-2-3-4-5 (Rangi: 12, 3, 2, 1, 0)
    # dla shorthand to nie działa, ALE dla shorthend będzie w ogóle inna funkcja do rozstrzygnięcia :)
    if set(ranks) == {12, 3, 2, 1, 0}:
        is_straight = True
        ranks = [3, 2, 1, 0] # W Wheelu 5 jest najwyższa (rank 3)

    # Konwertujemy ranks na krotkę (dla porównywania i typowania)
    ranks_tuple = tuple(ranks)

    # 1. Straight Flush
    if is_straight and is_flush:
        return (8, ranks_tuple)

    # Licznik rang (dla karety, fulla, par)
    counts = Counter(ranks)
    # Sortujemy: najpierw po liczbie wystąpień (malejąco), potem po randze (malejąco)
    sorted_counts = sorted(counts.items(), key=lambda x: (x[1], x[0]), reverse=True)
    
    # Rangi posortowane wg ważności - konwertujemy od razu na krotkę
    score_ranks = tuple([r for r, count in sorted_counts])

    # 2. Kareta (4 powtórzenia)
    if sorted_counts[0][1] == 4:
        return (7, score_ranks)
    
    # 3. Full House (3 + 2 powtórzenia)
    if sorted_counts[0][1] == 3 and sorted_counts[1][1] == 2:
        return (6, score_ranks)
    
    # 4. Flush
    if is_flush:
        return (5, ranks_tuple)
    
    # 5. Straight
    if is_straight:
        return (4, ranks_tuple) # ranks_tuple uwzględnia już poprawkę dla Wheel
    
    # 6. Trójka (3 powtórzenia)
    if sorted_counts[0][1] == 3:
        return (3, score_ranks)
    
    # 7. Dwie Pary (2 + 2 powtórzenia)
    if sorted_counts[0][1] == 2 and sorted_counts[1][1] == 2:
        return (2, score_ranks)
    
    # 8. Para (2 powtórzenia)
    if sorted_counts[0][1] == 2:
        return (1, score_ranks)
    
    # 9. Wysoka Karta
    return (0, ranks_tuple)

def get_best_hand(hole_cards, community_cards):
    """
    Znajduje najlepszy układ 5-kartowy z dostępnych 7 kart (2 ręka + 5 stół).
    """
    all_cards = hole_cards + community_cards
    best_score = (-1, ())
    
    # Iterujemy przez wszystkie kombinacje 5 kart z 7 (jest ich 21)
    for combo in itertools.combinations(all_cards, 5):
        score = evaluate_5_cards(combo)
        if score > best_score:
            best_score = score
            
    return best_score

def determine_winner(env) -> list[int]:
    """
    Główna funkcja do wyłonienia zwycięzcy w środowisku PokerEnv.
    Zwraca listę indeksów zwycięzców (zwykle jeden, ale możliwy remis/split pot).
    """
    active_players = [p for p in env.players if p.is_active]
    
    # Zabezpieczenie: Jeśli został tylko jeden gracz (wszyscy spasowali wcześniej)
    if len(active_players) == 1:
        return [active_players[0].id]
    
    best_score = (-1, ())
    winners = []
    
    # Oceniamy każdego aktywnego gracza
    for p in active_players:
        # Gracz musi użyć swoich kart + kart ze stołu
        # Jeśli board nie jest pełny (np. symulacja kończy się wcześniej), 
        # oceni to co jest, ale to ma sens głównie na Riverze.
        score = get_best_hand(p.hand, env.community_cards)
        
        # Debug (opcjonalne)
        # print(f"Gracz {p.id} Score: {score}")

        if score > best_score:
            best_score = score
            winners = [p.id]
        elif score == best_score:
            winners.append(p.id) # Remis (Split Pot)
            
    return winners

def get_hand_name(score_tuple):
    """Pomocnicza funkcja zamieniająca wynik (int) na nazwę tekstową"""
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