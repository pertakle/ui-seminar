from minimax_agent import h
import connectx as cx
import time

class GreedyHeuristicAgent(cx.Agent):
    def __init__(self) -> None:
        self.value_history = []
        self.thinking_times = []

    def reset(self) -> None:
        self.value_history = []
        self.thinking_times = []

    def __repr__(self) -> str:
        return "GreedyHeuristic"

    def move(self, state: cx.ConnectX) -> int:
        start = time.time()
        best_action = None
        best_val = None
        for action in state.possible_actions():
            state.make_move(action)
            val = h(state)
            state.undo_move(0) # last move nepotrebujeme
            if best_val is None or val < best_val:
                best_action = action
                best_val = val
        end = time.time()
        self.value_history.append(-best_val)
        self.thinking_times.append(end - start)
        return best_action