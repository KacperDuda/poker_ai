import settings
from poker_env import PokerEnv
from player import Player
import evaluator
from agent import RandomAgent, DeepAgent 

def conv_observation(obs_array):
    n_cards = settings.N_CARDS 
    
    hand_part = obs_array[:n_cards]
    hand_str = "".join(['1' if x > 0.5 else '0' for x in hand_part])
    
    if len(obs_array) >= 2 * n_cards:
        board_part = obs_array[n_cards : 2*n_cards]
        board_str = "".join(['1' if x > 0.5 else '0' for x in board_part])
    else:
        board_str = ""
        
    remaining_part = obs_array[2*n_cards:]
    result_string = f"{hand_str}\n{board_str}\n{str(remaining_part)}"
    
    return result_string

ACTION_FOLD = 0
ACTION_CALL_CHECK = 1
ACTION_RAISE_BET = 2
ACTION_DEAL = 999 

def run_simulation():
    print("=== START SIMULATION (R1: Random Agents Training) ===\n")

    players_pool = [
        Player(0, 1000.0),
        Player(1, 2000.0),
        Player(2, 1000.0),
        Player(3, 1000.0)
    ]
    
    env = PokerEnv(players_pool, sb=10, bb=20, debug=True)
    
    print(f">>> ROUND 1: Random Agents playing 5 hands.")
    
    NUM_RANDOM_HANDS = 5 
    
    run_random_game(env, num_hands=NUM_RANDOM_HANDS) 
    
    print_final_stacks(env.players)


def run_random_game(env: PokerEnv, num_hands=1):
    random_agents = {p.id: RandomAgent(p.id) for p in env.players}
    
    for hand in range(num_hands):
        print(f"\n=== HAND {hand + 1} (Random Agents) ===")
        env.start_round()

        game_over = False
        step_num = 1
        
        while not game_over and env.stage <= 3:
            
            if env._check_end_of_betting_round(): 
                if env.stage < 3:
                    stage_names = ['Preflop', 'Flop', 'Turn', 'River']
                    print(f"  [DEAL] Changing stage: {stage_names[env.stage + 1]}")
                    env.deal_next_stage()
                    print(env)
                    continue
                else:
                    game_over = True
                    break 

            current_p_idx = env.get_current_player_idx()
            current_p = env.players[current_p_idx]
            
            if not current_p.is_active or current_p.is_allin:
                env._next_active_player()
                continue
                
            obs = env.get_observation(current_p_idx)
            action, val = random_agents[current_p.id].get_action(obs)
            
            desc = {0: "FOLD", 1: "CALL/CHECK", 2: "RAISE"}.get(action, "UNKNOWN")
            
            if action == ACTION_RAISE_BET:
                print(f"  [Step {step_num}] P{current_p.id} ({desc}) -> Action: {action}, Val: {val:.2f}")
            else:
                print(f"  [Step {step_num}] P{current_p.id} ({desc}) -> Action: {action}")
                
            _, done, info = env.step(action, val)
            print(env)

            if done:
                game_over = True
                if "method" in info and info["method"] == "walkover":
                    print(f"  [END] Walkover! Player {info.get('winner')} wins")
                break
                
            step_num += 1

        if not game_over:
            print("\n  --- End of Betting, Showdown ---")
            if env.pot > 0:
                winners = env.finalize_showdown()
                print(env)
                print(f"  [RESULT] Winner IDs: {winners}")

        print_round_summary(env)


def play_hand_scripted(env: PokerEnv, scripted_actions):
    step_num = 1
    game_over = False
    
    for p_id, action, val, desc in scripted_actions:
        if action == ACTION_DEAL:
            print(f"  [{desc}] Changing stage...")
            env.deal_next_stage()
            print(env)
            continue

        current_p = env.players[env.current_player_idx]
        
        if p_id is not None and current_p.id != p_id:
            if action == ACTION_FOLD: continue 
            print(f"     [!] Warning: P{current_p.id}'s turn, script expected P{p_id}.")

        print(f"  [Step {step_num}] P{current_p.id} ({desc}) -> Action: {action}")
        
        _, done, info = env.step(action, val)
        print(env)
        
        if done:
            game_over = True
            if "method" in info and info["method"] == "walkover":
                print(f"  [END] Walkover! Player {info.get('winner')} wins")
            break
        step_num += 1

    if not game_over:
        print("\n  --- End of Betting, Showdown ---")
        if env.pot > 0:
            winners = env.finalize_showdown()
            print(env)
            print(f"  [RESULT] Winner IDs: {winners}")

    print_round_summary(env)


def print_round_summary(env):
    print("  Player States (Current Stacks):")
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
    print("\n=== FINAL SUMMARY ===")
    total = 0
    for p in players:
        print(f"Player {p.id}: {p.stack:.2f}$")
        total += p.stack
    print(f"Total in system: {total:.2f}$")

def _check_end_of_betting_round_logic(self):
    active_players = [p for p in self.players if p.is_active and not p.is_allin]
    
    if len(active_players) <= 1:
        return True 

    max_bet = max(p.current_bet for p in self.players)
    all_active_equal = all(p.current_bet == max_bet for p in self.players if p.is_active)

    next_player = self.players[self.current_player_idx]
    to_call = max_bet - next_player.current_bet
    
    if to_call <= 0.01 and all_active_equal:
         return True

    return False


if __name__ == "__main__":
    setattr(PokerEnv, "_check_end_of_betting_round", _check_end_of_betting_round_logic)

    run_simulation()