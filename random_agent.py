import connectx as cx
import numpy as np
from connectx import Config


class RandomAgent(cx.Agent):
    def __init__(self, seed:int=12) -> None:
        self.rng = np.random.RandomState(seed)
        self.seed = seed
        self.value_history = [0]
        self.thinking_times = [0]

    def __repr__(self) -> str:
        return f"Random(seed={self.seed})"
    
    def reset(self) -> None:
        pass

    def move(self, state: cx.ConnectX) -> int:
        return self.rng.choice(state.possible_actions())