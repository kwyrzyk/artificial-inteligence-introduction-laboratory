from state import State

def minimax(state:State, depth:int=None):
    if state.is_finished() or depth == 0:
        return state.heuristic()
    
    successors = state.get_successors()
    heuristics = [minimax((succ), depth-1) for succ in successors]

    if state.next_move == 1:
        return max(heuristics)
    else:
        return min(heuristics)
     
