from state import State
from minimax import minimax
from enum import Enum
import random


class Player:
    def __init__(self, name: str, pred_depth: int) -> None:
        self._name = name
        self._pred_depth = pred_depth

    def make_move(self, state: State) -> tuple[int, int]:
        successors = state.get_successors_with_moves()
        states_heuristics = {}
        for succ in list(successors.keys()):
            heuristic = minimax(succ, self._pred_depth - 1)
            states_heuristics[succ] = heuristic
        if state.get_next_move() == 1:
            best_heuristic = max(list(states_heuristics.values()))
        else:
            best_heuristic = min(list(states_heuristics.values()))
        best_states = []
        for succ in states_heuristics.keys():
            if states_heuristics[succ] == best_heuristic:
                best_states.append(succ)
        best_state = random.choice(best_states)
        best_move = successors[best_state]
        return best_move

    def get_name(self) -> str:
        return self._name

    def set_name(self, new_name: str) -> None:
        self._name = new_name

    def get_pred_depth(self) -> int:
        return self._pred_depth

    def set_pred_depth(self, new_pred_depth: int) -> None:
        self._pred_depth = new_pred_depth
