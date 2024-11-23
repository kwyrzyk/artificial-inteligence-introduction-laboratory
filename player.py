from state import State
from minimax import minimax
from enum import Enum
import random


class GameStyle(Enum):
    MINIMIZING = -1
    MAXIMIZING = 1


class Player:
    def __init__(self, name:str, pred_depth:int, game_style:GameStyle):
        self.name = name
        self.pred_depth = pred_depth
        self.game_style = game_style
    
    def make_move(self, state:State):
        successors = state.get_successors_with_moves()
        states_heuristics = {}
        for succ in list(successors.keys()):
            heuristic = minimax(succ, self.pred_depth)
            states_heuristics[succ] = heuristic
        if self.game_style == GameStyle.MAXIMIZING:
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