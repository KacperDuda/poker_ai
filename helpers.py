import settings

#WYGENEROWANED W AI!!!
def conv_observation(obs_array):
    n_cards = settings.N_CARDS # 52
    
    # 1. Pierwszy wektor (Ręka)
    hand_part = obs_array[:n_cards]
    hand_str = "".join(['1' if x > 0.5 else '0' for x in hand_part])
    
    # 2. Drugi wektor (Stół)
    # Sprawdzamy czy tablica jest dość długa, jeśli nie - wypełniamy zerami logicznymi
    if len(obs_array) >= 2 * n_cards:
        board_part = obs_array[n_cards : 2*n_cards]
        board_str = "".join(['1' if x > 0.5 else '0' for x in board_part])
    else:
        board_str = ""
        
    remaining_part = obs_array[2*n_cards:]
    result_string = f"{hand_str}\n{board_str}\n{str(remaining_part)}"
    
    return result_string