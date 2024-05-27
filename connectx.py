import numpy as np
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Self

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

    def copy_from(self, game_state: Self) -> None:
        #self.config = game_state.config
        self.board = game_state.board.copy()
        self.top = game_state.top.copy()
        self.player = game_state.player
        self.last_move = game_state.last_move

    def get_state(self) -> Self:
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



class Agent(ABC):

    @abstractmethod
    def move(self, state: ConnectX) -> int:
        raise NotImplementedError
    
    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError
    
import time
def play(config: Config, agent1: Agent, agent2: Agent, verbose:int=0) -> int:
    game = ConnectX(config)

    # NOTE: agents[0] je jen vylpn pro indexovani 1/-1 a nemel by se nikdy pouzit
    agents = [agent1, agent1, agent2]
    thinking = [None, [], []]

    while not game.is_done():
        if verbose > 0: 
            game.print_board()
        player = game.get_player()
        state = game.get_state()

        start = time.time()
        action = agents[player].move(state)
        end = time.time()
        thinking[player].append(end-start)
        game.make_move(action)
        
        if verbose > 1: 
            print(f"Na tahu: {ConnectX.ZNAKY[player]}")
            print(f"Tazeno: {action}")
        if verbose > 2: 
            input("<Enter>")
    if verbose > 0:
        game.print_board()
    if verbose > 1:
        print(f"Vyhral {ConnectX.ZNAKY[game.outcome()]}")
    if verbose > 2:
        input("<Enter>")
    print("Mean time spent")
    print(f"X: {np.mean(thinking[1]):.3f}")
    print(f"O: {np.mean(thinking[-1]):.3f}")
    return game.outcome()

