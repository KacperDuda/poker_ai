import helpers
import settings
from poker_env import PokerEnv
from player import Player
import evaluator
from helpers import conv_observation

# Stałe akcji
ACTION_FOLD = 0
ACTION_CALL_CHECK = 1
ACTION_RAISE_BET = 2
ACTION_DEAL = 999  # Sygnał dla skryptu do zmiany etapu

def run_simulation():
    print("=== START SYMULACJI (R1: Start, R2: Kontynuacja, R3: Reset) ===\n")

    players_pool = [
        Player(0, 1000.0),
        Player(1, 2000.0),
        Player(2, 1000.0),
        Player(3, 1000.0)
    ]
    
    env = PokerEnv(players_pool, sb=10, bb=20, debug=True)
    
    # ================= RUNDA 1 =================
    print(f">>> RUNDA 1: Start (P0 wygrywa walkowerem na Flopie)")
    env.start_round()

    script_r1 = [
        (3, ACTION_FOLD, 0.0,      "P3 (UTG) pasuje"),
        (0, ACTION_RAISE_BET, 0.1, "P0 (BTN) podbija"),
        (1, ACTION_FOLD, 0.0,      "P1 (SB) pasuje"),
        (2, ACTION_CALL_CHECK, 0.0, "P2 (BB) sprawdza"),
        (None, ACTION_DEAL, 0.0,   "--- DEAL FLOP ---"),
        (2, ACTION_CALL_CHECK, 0.0, "P2 (BB) czeka"),
        (0, ACTION_RAISE_BET, 0.3, "P0 (BTN) betuje"),
        (2, ACTION_FOLD, 0.0,      "P2 (BB) pasuje -> Walkower dla P0")
    ]
    play_hand(env, script_r1)
    
    print(f"\n>>> RUNDA 2: Kontynuacja (Start Round - Stacki zachowane)")
    
    env.start_round()

    script_r2 = [
        (0, ACTION_CALL_CHECK, 0.0, "P0 (UTG) Limp"),
        (1, ACTION_FOLD, 0.0,       "P1 (BTN) Fold"),
        (2, ACTION_CALL_CHECK, 0.0, "P2 (SB) Complete"), 
        (3, ACTION_RAISE_BET, 1.0,  "P3 (BB) ALL-IN!"),  
        (0, ACTION_CALL_CHECK, 0.0, "P0 sprawdza All-in"),
        (2, ACTION_CALL_CHECK, 0.0, "P2 sprawdza All-in"),
    ]
    play_hand(env, script_r2)
    
    # ================= KONIEC =================
    print_final_stacks(env.players)


def play_hand(env: PokerEnv, scripted_actions):
    step_num = 1
    game_over = False
    
    for p_id, action, val, desc in scripted_actions:
        if action == ACTION_DEAL:
            print(f"  [{desc}] Zmiana etapu...")
            env.deal_next_stage()
            print(env)
            continue

        current_p = env.players[env.current_player_idx]
        
        if p_id is not None and current_p.id != p_id:
            if action == ACTION_FOLD: continue 
            print(f"     [!] Uwaga: Kolej P{current_p.id}, skrypt oczekiwał P{p_id}.")

        print(f"  [Krok {step_num}] P{current_p.id} ({desc}) -> Action: {action}")
        
        _, done, info = env.step(action, val)
        print(env)
        print(helpers.conv_observation(env.get_observation(current_p.id)))
        
        if done:
            game_over = True
            if "method" in info and info["method"] == "walkower":
                print(f"  [KONIEC] Walkower! Wygrywa Gracz {info.get('winner')}")
            break
        step_num += 1

    if not game_over:
        print("\n  --- Koniec licytacji, Showdown ---")
        if env.pot > 0:
            winners = env.finalize_showdown()
            print(env)
            print(f"  [WYNIK] Zwycięzcy ID: {winners}")

    print_round_summary(env)


def print_round_summary(env):
    print("  Stany graczy (Bieżące Stacki):")
    for p in env.players:
        hand_desc = "Folded"
        if p.is_active:
            try:
                if len(env.community_cards) >= 3:
                    score_tuple = evaluator.get_best_hand(p.hand, env.community_cards)
                    hand_desc = evaluator.get_hand_name(score_tuple)
                else:
                    hand_desc = "Pre-flop/Flop (No Showdown)"
            except:
                hand_desc = "Unknown"
            
        print(f"    P{p.id}: {p.stack:.2f}$ ({hand_desc})")


def print_final_stacks(players):
    print("\n=== PODSUMOWANIE KOŃCOWE ===")
    total = 0
    for p in players:
        print(f"Gracz {p.id}: {p.stack:.2f}$")
        total += p.stack
    print(f"Suma w systemie: {total:.2f}$")

if __name__ == "__main__":
    run_simulation()