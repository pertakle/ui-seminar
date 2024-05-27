import numpy as np
import connectx as cx
import time
from typing import Self, Optional
from mcts_agent import Node
from minimax_agent import h

class NodeNoRollout(Node):
    
    state: cx.ConnectX
    base_strategy: cx.Agent
    maximizing: bool
    action: Optional[int]
    utility: int
    visits: int
    parent: Optional[Self]
    children: list[Self]
    unexplored_actions: list[int]
    terminal: bool

    def __init__(self,
                 state: cx.ConnectX, 
                 maximizing: bool,
                 action: int = None,
                 parent: Self = None) -> None:
        super().__init__(None, state, maximizing, action, parent)
        
    def expand(self) -> Self:
        if len(self.unexplored_actions) == 0 or self.state.is_done():
            return self
        
        action = self.unexplored_actions.pop()
        new_state = self.state.get_state()#self.game.clone(self.state)
        #self.game.apply(new_state, action)
        new_state.make_move(action)
        new_node = NodeNoRollout(new_state, not self.maximizing, action, self)
        self.children.append(new_node)

        return new_node
    
    def simulate(self) -> float:
        return h(self.state)
    



class MCTSNoRolloutAgent(cx.Agent):
    """
    Strategy implementation selecting action
    by Monte Carlo Tree-search with base strategy method.
    """

    def __init__(self, limit: int, seed: Optional[int] = None):
        super().__init__()
        self.seed = seed
        self.head = None
        self.limit = limit
        self.rng = np.random.RandomState(seed)
        self.value_history = []
        self.thinking_times = []

    def __repr__(self) -> str:
        return f"MCTSNoRollout(seed={self.seed})"

    def move(self, state: cx.ConnectX) -> int:
        """
        Return best action for given state.
        """
        start = time.time()
        # Your implementation goes here.
        if self.head is not None:
            self.head = self.head.get_child(state)
        if self.head is None:
            self.head = NodeNoRollout(state, state.get_player() == 1)

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