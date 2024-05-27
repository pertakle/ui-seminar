import numpy as np
import connectx as cx
import time
from typing import Self, Optional

class Node:
    
    state: cx.ConnectX
    base_strategy: cx.Agent
    maximizing: bool
    action: Optional[int]
    utility: int
    visits: int
    parent: Optional[Self]
    children: list[Self] #TODO: misto listu pouzivat ndarray s max indexem
    unexplored_actions: list[int]
    terminal: bool

    def __init__(self,
                 base_strategy: cx.Agent,
                 state: cx.ConnectX, 
                 maximizing: bool,
                 action: int = None,
                 parent: Self = None) -> None:
        self.state = state
        self.base_strategy = base_strategy
        self.maximizing = maximizing
        self.action = action
        self.utility = 0
        self.visits = 0
        self.parent = parent
        self.children = []
        self.unexplored_actions = state.list_possible_actions()
        self.terminal = len(self.unexplored_actions) == 0

    def __repr__(self) -> str:
        return f"{self.utility}/{self.visits}ch: {[(ch.utility, ch.visits) for ch in self.children]}"
        
    def get_child(self, state: object) -> Optional[Self]:
        for child in self.children:
            if child.state == state:
                child.parent = None
                return child
        return None

    def fully_expanded(self) -> bool:
        return len(self.unexplored_actions) == 0
    
    def uct(self) -> float:
        return (self.utility if not self.maximizing else -self.utility) / self.visits + np.sqrt(2 * np.log(self.parent.visits) / self.visits)

    def sellect(self) -> Self:
        best_uct = 0
        best_child = None
        for child in self.children:
            child_uct = child.uct()
            if best_child is None or child_uct > best_uct:
                best_uct = child_uct
                best_child = child
        return best_child

    def expand(self) -> Self:
        if len(self.unexplored_actions) == 0 or self.state.is_done():
            return self
        
        action = self.unexplored_actions.pop()
        new_state = self.state.get_state()#self.game.clone(self.state)
        #self.game.apply(new_state, action)
        new_state.make_move(action)
        new_node = Node(self.base_strategy, new_state, not self.maximizing, action, self)
        self.children.append(new_node)

        return new_node


    def simulate(self) -> float:
        state = self.state.get_state()#self.game.clone(self.state)
        while not state.is_done():
            action = self.base_strategy.move(state)
            #self.game.apply(state, action)
            state.make_move(action)
        return state.outcome()

    def add_result(self, result: float) -> None:
        node = self
        while node is not None:
            node.utility += result
            node.visits += 1
            node = node.parent



class MCTSAgent(cx.Agent):
    """
    Strategy implementation selecting action
    by Monte Carlo Tree-search with base strategy method.
    """

    def __init__(self, base_strat: cx.Agent, limit: int, seed: Optional[int] = None):
        super().__init__()
        self.seed = seed
        self.base_strat = base_strat
        self.head = None
        self.limit = limit
        self.rng = np.random.RandomState(seed)
        self.value_history = []
        self.thinking_times = []

    def __repr__(self) -> str:
        return f"MCTS(base={self.base_strat}, seed={self.seed})"
    
    def reset(self) -> None:
        self.head = None
        self.value_history = []
        self.thinking_times = []

    def move(self, state: cx.ConnectX) -> int:
        """
        Return best action for given state.
        """
        start = time.time()
        # Your implementation goes here.
        if self.head is not None:
            self.head = self.head.get_child(state)
        if self.head is None:
            self.head = Node(self.base_strat, state, state.get_player() == 1)

        while time.time() - start < 1.9:
        #for simulation in range(self.limit):
            node = self.head
            while node.fully_expanded() and len(node.children) > 0:
                node = node.sellect()

            node = node.expand()
            result  = node.simulate()

            #while node is not None:
            node.add_result(result)
            #    node = node.parent

        best_child = None
        for child in self.head.children:
            if best_child is None or child.visits > best_child.visits:
                best_child = child
        end = time.time()
        self.thinking_times.append(end - start)
        self.value_history.append(best_child.utility / best_child.visits)
        print(f"{str(self)}: {best_child.utility / best_child.visits:.3f} \
({best_child.visits}/{best_child.parent.visits}={best_child.visits/best_child.parent.visits:.3f})")
        #print(f"MCTS: {best_child.visits / self.head.visits:.3f} ({self.head.visits})")
        return best_child.action