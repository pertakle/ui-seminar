import numpy as np
import connectx as cx
import time


def h(state: cx.ConnectX) -> float:
    def h_universal(state: cx.ConnectX, player: int, pocatky: list[tuple[int, int]], dx: int, dy: int) -> int:
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
        
    return (hp - ho) / (hp + ho + 1)# if hp + ho != 0 else 0

def h_fast(state: cx.ConnectX, last_x: int, last_y: int, hp: int, ho: int) -> tuple[int, int]:
    ...

#g = cx.ConnectX(cx.Config(8, 7, 4))
#g.board[-1, 1] = 1
#g.board[-1,2] = -1
#g.print_board()
#print("ri, i, j, delka, mezera_pred, mezera_za")
#print(h(g))
#exit()


class MinimaxAgent(cx.Agent):
    def __init__(self, max_depth: int) -> None:
        self.max_depth = max_depth
        self.value_history = []
        self.thinking_times = []
        self.turn = 0

    def __repr__(self) -> str:
        return f"Minimax(depth={self.max_depth})"
    
    def reset(self) -> None:
        self.value_history = []
        self.thinking_times = []
        self.turn = 0

    def move(self, state: cx.ConnectX) -> int:
        self.turn += 1

        start = time.time()
        action, value = self.minimax(state, self.max_depth, -2, 2)
        end = time.time()
        self.thinking_times.append(end-start)
        self.value_history.append(state.get_player() * value)
        print(f"Minimax: {state.get_player()*value:.3f}")
        return action

    def minimax(self, state: cx.ConnectX, max_depth: int, alpha: int, beta: int) -> tuple[int, float]:
        if state.is_done():
            return None, -abs(state.outcome()) #* (0.99 ** (self.max_depth - max_depth))
        if max_depth == 0:
            return None, h(state)
        
        best_action = None
        best_action_value = None
        last_move = state.last_move
        for action in state.possible_actions():
            state.make_move(action)
            value = -self.minimax(state, max_depth-1, -beta, -alpha)[1]
            state.undo_move(last_move)

            if best_action_value is None or value > best_action_value:
                best_action = action
                best_action_value = value
            alpha = max(alpha, value)
            if alpha >= beta or best_action_value == 1:
                break
        
        return best_action, best_action_value
    