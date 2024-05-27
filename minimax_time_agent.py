import numpy as np
import connectx as cx
import time
from minimax_agent import h


class MinimaxTimeAgent(cx.Agent):
    def __init__(self, time_to_think: float) -> None:
        self.time_to_think = time_to_think
        self.value_history = []
        self.thinking_times = []
        self.min_depth = []
        self.max_depth = []
        self.turn = 0

    def __repr__(self) -> str:
        return f"MinimaxTime(thinking={self.time_to_think})"
    
    def reset(self) -> None:
        self.value_history = []
        self.thinking_times = []
        self.min_depth = []
        self.max_depth = []
        self.turn = 0


    def move(self, state: cx.ConnectX) -> int:
        self.turn += 1
        self.min_d = 10000
        self.max_d = -10000
        start = time.time()
        action, value = self.minimax_time(state, start + self.time_to_think, -2, 2, 0)
        end = time.time()
        self.thinking_times.append(end-start)
        self.value_history.append(state.get_player() * value)
        self.min_depth.append(self.min_d)
        self.max_depth.append(self.max_d)
        print(f"MinimaxTime: {state.get_player()*value:.3f} (d {self.min_d} - {self.max_d})")
        return action
    
    def minimax_time(self, state: cx.ConnectX, deadline: float, alpha: int, beta: int, depth: int) -> tuple[int, float]:
        if state.is_done():
            self.min_d = min(depth, self.min_d)
            self.max_d = max(depth, self.max_d)
            return None, -abs(state.outcome())# * 0.99**depth
        if time.time() >= deadline:
            self.min_d = min(depth, self.min_d)
            self.max_d = max(depth, self.max_d)
            return None, h(state)# * 0.99**depth
        

        best_action = None
        best_action_value = None
        last_move = state.last_move
        possible_actions = state.possible_actions()

        for i, action in enumerate(possible_actions):
            state.make_move(action)

            now = time.time()
            subtree_deadline = now + (deadline - now) / (len(possible_actions) - i)
            
            value = -self.minimax_time(state, subtree_deadline, -beta, -alpha, depth+1)[1]
            state.undo_move(last_move)

            if best_action_value is None or value > best_action_value:
                best_action = action
                best_action_value = value
            alpha = max(alpha, value)
            if alpha >= beta or best_action_value == 1:
                break
        
        return best_action, best_action_value