import time
from poker_env import PokerEnv
import evaluator
import settings

def pause(sec=0):
    """Krótka pauza dla czytelności w terminalu."""
    time.sleep(sec)

def manual_step(env, action_type, amt_pct=0.0, comment=""):
    """
    Wykonuje ruch dla AKTUALNEGO gracza.
    
    Parametr amt_pct dla RAISE (action_type=2):
      - 0.0 -> MIN RAISE (minimalna dozwolona kwota przebicia)
      - 1.0 -> ALL-IN (wrzucenie całego stacka)
      - 0.5 -> Połowa zakresu między Min a Max
    """
    player_idx = env.get_current_player_idx()
    p = env.players[player_idx]
    
    # --- Generowanie opisu ruchu do logów ---
    act_desc = "FOLD"
    if action_type == 1: 
        act_desc = "CHECK/CALL"
    elif action_type == 2:
        # Interpretacja suwaka dla logów
        if amt_pct == 0.0:
            act_desc = "RAISE (MIN)"
        elif amt_pct == 1.0:
            act_desc = "RAISE (ALL-IN)"
        else:
            act_desc = f"RAISE (Suwak: {amt_pct*100:.0f}%)"

    print(f"\n>> RUCH: Gracz P{player_idx} ({p.stack:.0f}$) -> {act_desc} {comment}")
    
    # --- Wykonanie kroku w środowisku ---
    # env.step sam przeliczy amt_pct na konkretną kwotę żetonów
    env.step(action_type, amt_pct)
    env.render()
    pause()

def run_scripted_game():
    print("=== SCENARIUSZ: Test Suwaka (Min vs All-in) ===")
    
    # 1. Inicjalizacja: 3 graczy, stacki 1000$
    players_ids = [0, 1, 2]
    # Upewnij się, że masz settings.py, deck.py, card.py, player.py i poker_env.py
    env = PokerEnv(players_ids, initial_stack=1000.0)
    
    env.reset()
    env.render()
    print("Start rozdania. Dealer: P0, SB: P1, BB: P2")
    pause()

    # ------------------------------------------------
    # FAZA 1: PREFLOP
    # Kolejność: P0 (D) -> P1 (SB) -> P2 (BB)
    # ------------------------------------------------
    print("\n--- FAZA: PREFLOP ---")
    
    # P0: Chce przebić, ale minimalnie, żeby tanio zobaczyć flop.
    # Używamy suwaka 0.0 -> to jest MIN RAISE.
    manual_step(env, action_type=2, amt_pct=0.0, comment="(Chcę przebić tylko o minimum!)")
    
    # P1: Sprawdza (Call)
    manual_step(env, action_type=1, comment="(Call)")
    
    # P2: Pasuje (Fold)
    manual_step(env, action_type=0, comment="(Fold)")
    
    # ------------------------------------------------
    # FAZA 2: FLOP
    # ------------------------------------------------
    print("\n>>> ROZDAJEMY FLOPA...")
    env.deal_next_stage()
    env.render()
    pause()
    
    print("\n--- FAZA: FLOP ---")
    # Zostali P0 i P1. Zaczyna P1 (SB).
    
    # P1: Czeka
    manual_step(env, action_type=1, comment="(Check)")
    
    # P0: Ma silną rękę, chce przebić solidnie.
    # Ustawia suwak na 50% zakresu między min-raise a swoim max stackiem.
    manual_step(env, action_type=2, amt_pct=0.5, comment="(Mocny Raise - 50% suwaka)")
    
    # P1: Sprawdza
    manual_step(env, action_type=1, comment="(Call)")

    # ------------------------------------------------
    # FAZA 3: TURN
    # ------------------------------------------------
    print("\n>>> ROZDAJEMY TURNA...")
    env.deal_next_stage()
    env.render()
    pause()
    
    print("\n--- FAZA: TURN ---")
    
    # Obaj czekają (Check-Check)
    manual_step(env, action_type=1, comment="(Check)")
    manual_step(env, action_type=1, comment="(Check)")

    # ------------------------------------------------
    # FAZA 4: RIVER
    # ------------------------------------------------
    print("\n>>> ROZDAJEMY RIVERA...")
    env.deal_next_stage()
    env.render()
    pause()
    
    print("\n--- FAZA: RIVER ---")
    
    # P1: Próbuje ukraść pulę minimalnym zakładem (Suwak 0.0)
    manual_step(env, action_type=2, amt_pct=0.0, comment="(Bet MIN - próba kradzieży)")
    
    # P0: Wchodzi ALL-IN! (Suwak 1.0)
    # Niezależnie od kwoty w stacku, 1.0 oznacza 'wrzucam wszystko co mam'
    manual_step(env, action_type=2, amt_pct=1.0, comment="(ALL-IN! Suwak na 100%)")
    
    # P1: Sprawdza (Hero Call)
    manual_step(env, action_type=1, comment="(Call All-In)")

    # ------------------------------------------------
    # SHOWDOWN
    # ------------------------------------------------
    print("\n=== SHOWDOWN (Koniec gry) ===")
    
    active_players = [p for p in env.players if p.is_active]
    
    # Pokazujemy karty
    for p in active_players:
        score_tuple = evaluator.get_best_hand(p.hand, env.community_cards)
        hand_name = evaluator.get_hand_name(score_tuple)
        print(f"Gracz P{p.id}: {p.hand} -> {hand_name}")
        
    # Wyłaniamy zwycięzcę
    winners = evaluator.determine_winner(env)
    print(f"\n$$$ ZWYCIĘZCA: P{winners} zgarnia pulę {env.pot:.1f}$ $$$")

if __name__ == "__main__":
    run_scripted_game()