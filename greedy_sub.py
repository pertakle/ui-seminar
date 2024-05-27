#from minimax_agent import h
# import connectx as cx
# import time
from dataclasses import dataclass
import numpy as np


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


def h(state: ConnectX) -> float:
    def h_universal(state: ConnectX, player: int, pocatky: list[tuple[int, int]], dx: int, dy: int) -> int:
        board = state.board
        win_len = state.config.win_len

        res = 0
        for x, y in pocatky:
            while state._inside_bounds(x, y):
                mezera_pred = 0
                while state._inside_bounds(x, y) and board[x, y] != player:
                    if board[x, y] == 0:
                        mezera_pred += 1
                    else:
                        mezera_pred = 0
                    x += dx
                    y += dy

                delka = 0
                while state._inside_bounds(x, y) and board[x, y] == player:
                    delka += 1
                    x += dx
                    y += dy

                mezera_za = 0
                while state._inside_bounds(x, y) and board[x, y] != -player:
                    mezera_za += 1
                    x += dx
                    y += dy

                if delka > 0 and mezera_pred + delka + mezera_za >= win_len:
                    if mezera_pred > 0 and mezera_za > 0:
                        res += delka * delka
                    else:
                        res += delka
        return res


    player = state.get_player()
    rows, columns = state.board.shape

    hp = 0
    ho = 0
    for dx, dy, pocatky in [
        (1, 1, [*((0, i) for i in range(columns)), *((i, 0) for i in range(1, rows))]),
        (1,-1, [*((0, i) for i in range(columns)), *((i, columns-1) for i in range(1, rows))]),
        (1, 0, [(0, i) for i in range(columns)]),
        (0, 1, [(i, 0) for i in range(rows)])
    ]:
        hp += h_universal(state, player, pocatky, dx, dy)
        ho += h_universal(state, -player, pocatky, dx, dy)

    return (hp - ho) / (hp + ho + 1)# if hp + ho != 0 else 0k

def greedy_move(state: ConnectX) -> int:
    best_action = None
    best_val = None
    for action in state.possible_actions():
        state.make_move(action)
        val = h(state)
        state.undo_move(0) # last move nepotrebujeme
        if best_val is None or val < best_val:
            best_action = action
            best_val = val
    return best_action

def compute_top(board):
    extended = np.vstack((np.zeros(board.shape[1]), board))
    return np.argmin(extended, axis=0) - 2

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

    game = ConnectX(conf)
    game.board = board
    game.player = player
    game.top = compute_top(board)


    return greedy_move(game)