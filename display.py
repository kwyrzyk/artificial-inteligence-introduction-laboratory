from player import Player
from state import State
import time


class Display:
    def __init__(self) -> None:
        pass

    def show_state(self, state: State, next_player: Player) -> None:
        time.sleep(1)
        print(state, end="\n\n")
        if not state.is_finished():
            time.sleep(1)
            print(next_player.get_name() + "'s turn:", end="\n\n")

    def show_result(self, state: State, player1: Player, player2: Player) -> None:
        time.sleep(1)
        match state.get_result():
            case -1:
                print(player2.get_name() + " won the game")
            case 0:
                print('The game ended in a draw')
            case -1:
                print(player1.get_name() + " won the game")