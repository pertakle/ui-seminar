import numpy as np
# import connectx as cx
from dataclasses import dataclass
import time
# from typing import Self, Optional

@dataclass
class Config:
    columns: int
    rows: int
    win_len: int



class ConnectX:

    ZNAKY = [" ", "X", "O"]

    def __init__(self, config: Config) -> None:
        self.config = config
        self.board = np.zeros((config.rows, config.columns), dtype=int)
        self.top = np.full((config.columns,), config.rows - 1, dtype=int)
        self.player = 1
        self.last_move = None

    def print_board(self) -> None:
        print("\n  ", end="")
        print(*range(self.config.columns), sep="   ")
        cara = ("+---"*self.config.columns) + "+"
        for line in self.board:
            print(cara)
            for znak in line:
                print("| ", self.ZNAKY[znak], " ", sep="", end="")
            print("|")
        print(cara)

    def copy_from(self, game_state) -> None:
        #self.config = game_state.config
        self.board = game_state.board.copy()
        self.top = game_state.top.copy()
        self.player = game_state.player
        self.last_move = game_state.last_move

    def get_state(self):
        new_state = ConnectX(self.config)
        new_state.copy_from(self)
        return new_state

    
    def get_player(self) -> int:
        return self.player

    def make_move(self, column: int) -> None:
        #assert column >= 0
        #assert column < self.config.columns
        #assert self.top[column] >= 0

        self.board[self.top[column], column] = self.player
        self.top[column] -= 1
        self.player = -self.player
        self.last_move = column

    def make_different_move(self, column: int) -> None:
        self.undo_move(0) # prev_last_move nas nezajima, protoze se v make_move hned prepise
        self.make_move(column)

    def undo_move(self, prev_last_move: int) -> None:
        x, y = self._get_last_position()
        self.board[x, y] = 0
        self.top[y] += 1
        self.player = -self.player
        self.last_move = prev_last_move

    def _inside_bounds(self, x: int, y: int) -> bool:
        return 0 <= x and x < self.config.rows and 0 <= y and y < self.config.columns

    def _direction_count(self, x: int, y: int, dx: int, dy: int) -> int:
        checked_player = self.board[x, y]
        count = 0
        for i in range(1, self.config.win_len):
            xx = x + i*dx
            yy = y + i*dy
            if not self._inside_bounds(xx, yy) or self.board[xx, yy] != checked_player:
                break
            count += 1
        return count
    
    def _get_last_position(self) -> tuple[int, int]:
        return self.top[self.last_move] + 1, self.last_move

    def _check_win(self) -> bool:
        if self.last_move is None:
            return False
        x, y = self._get_last_position()
        for dx, dy in [(1, 0), (0, 1), (1, 1), (1, -1)]:
            connected_count = self._direction_count(x, y, dx, dy) + self._direction_count(x, y, -dx, -dy) + 1
            if connected_count >= self.config.win_len:
                return True
        return False

    def _board_full(self) -> bool:
        return np.all(self.top == -1)

    def is_done(self) -> bool:
        return self._board_full() or self._check_win()
    
    def outcome(self) -> int:
        if self._board_full():
            return 0
        x, y = self._get_last_position()
        return self.board[x, y]
    
    def possible_actions(self) -> np.ndarray:
        return np.arange(self.config.columns, dtype=int)[self.top >= 0]
    
    def list_possible_actions(self) -> list[int]:
        return list(self.possible_actions())


class RandomAgent:
    def __init__(self, seed:int=12) -> None:
        self.rng = np.random.RandomState(seed)
        self.seed = seed
        self.value_history = [0]
        self.thinking_times = [0]

    def __repr__(self) -> str:
        return f"Random(seed={self.seed})"
    
    def reset(self) -> None:
        pass

    def move(self, state: ConnectX) -> int:
        return self.rng.choice(state.possible_actions())

class Node:
    
    state: ConnectX
    base_strategy: RandomAgent
    maximizing: bool
    action: int
    utility: int
    visits: int
    parent: None
    children: list
    unexplored_actions: list[int]
    terminal: bool

    def __init__(self,
                 base_strategy: RandomAgent,
                 state: ConnectX, 
                 maximizing: bool,
                 action: int = None,
                 parent = None) -> None:
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
        
    def get_child(self, state: object):
        for child in self.children:
            if child.state == state:
                child.parent = None
                return child
        return None

    def fully_expanded(self) -> bool:
        return len(self.unexplored_actions) == 0
    
    def uct(self) -> float:
        return (self.utility if not self.maximizing else -self.utility) / self.visits + np.sqrt(2 * np.log(self.parent.visits) / self.visits)

    def sellect(self):
        best_uct = 0
        best_child = None
        for child in self.children:
            child_uct = child.uct()
            if best_child is None or child_uct > best_uct:
                best_uct = child_uct
                best_child = child
        return best_child

    def expand(self):
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


class MCTSAgent:
    """
    Strategy implementation selecting action
    by Monte Carlo Tree-search with base strategy method.
    """

    def __init__(self, base_strat, seed = None):
        super().__init__()
        self.seed = seed
        self.base_strat = base_strat
        self.head = None
        self.rng = np.random.RandomState(seed)
        self.value_history = []
        self.thinking_times = []

        self.game_state: ConnectX = None

    def __repr__(self) -> str:
        return f"MCTS(base={self.base_strat}, seed={self.seed})"
    
    def reset(self) -> None:
        self.head = None
        self.value_history = []
        self.thinking_times = []

    def move(self, conf: Config, board: np.ndarray, player: int) -> int:
        """
        Return best action for given state.
        """
        start = time.time()

        if self.game_state is None:
            self.game_state = ConnectX(conf)
            limit = 59.9
        else:
            limit = 1.9

        # update internal state according to opponent's move
        last_move = find_last_move(self.game_state.board, board)
        self.game_state.player = -player
        self.game_state.make_move(last_move)


        if self.head is not None:
            self.head = self.head.get_child(self.game_state)
        if self.head is None:
            self.head = Node(self.base_strat, self.game_state, self.game_state.get_player() == 1)

        while time.time() - start < limit:
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
        if best_child is None:
            action = self.base_strat.move(self.game_state)
        else:
            action = best_child.action
        self.game_state.make_move(action)
        # end = time.time()
        # self.thinking_times.append(end - start)
        # self.value_history.append(best_child.utility / best_child.visits)
#         print(f"{str(self)}: {best_child.utility / best_child.visits:.3f} \
# ({best_child.visits}/{best_child.parent.visits}={best_child.visits/best_child.parent.visits:.3f})")
        return best_child.action
    


def find_last_move(board: np.ndarray, new_board: np.ndarray) -> int:
    return np.argmax(np.any(board != new_board, axis=0))

def empty_game(board: np.ndarray) -> bool:
    return np.all(board == 0)

agent = MCTSAgent(RandomAgent(12), 21)
def move_agent(observation, configuration):
    # Number of Columns on the Board.
    columns = configuration.columns
    # Number of Rows on the Board.
    rows = configuration.rows
    # Number of Checkers "in a row" needed to win.
    inarow = configuration.inarow
    # The current serialized Board (rows x columns).
    board = observation.board
    # Which player the agent is playing as (1 or 2).
    mark = observation.mark

    conf = Config(columns, rows, inarow)
    board = (np.array(board).reshape(rows, columns)*2) - 3
    player = (mark * 2) - 3

    return agent.move(conf, board, player)