import connectx as cx
from random_agent import RandomAgent
from mcts_agent import MCTSAgent
from minimax_agent import MinimaxAgent
from greedy_heuristic_agent import GreedyHeuristicAgent
from minimax_time_agent import MinimaxTimeAgent
from mcts_no_rollout import MCTSNoRolloutAgent
import numpy as np
from matplotlib import pyplot as plt

def print_table(table: list[tuple[str, str]]) -> None:
    def max_table_lengths(table: list[tuple[str, str]]) -> np.ndarray:
        max_lens = np.zeros(2, dtype=int)
        for agent, wins in table:
            max_lens[0] = max(max_lens[0], len(agent))
            max_lens[1] = max(max_lens[1], len(wins))
        return  max_lens
    
    def print_dato(dato: tuple[str, str], max_len: int, pad:str=" ", sep:str="|") -> None:
        print(pad * (1 + max_len - len(dato)), dato, pad+sep, sep="", end="")

    def print_hline(line_char:str="-") -> None:
        print("+", end="")
        for column in range(2):
            print_dato("", max_lens[column], line_char, "+")
        print()

    max_lens = max_table_lengths(table)
    print_hline()
    for i, record in enumerate(table):
        print("|", end="")
        for max_len, dato in zip(max_lens, record):
            print_dato(dato, max_len)
        print()

        if i == 0: # separate header
            print_hline("=")
    print_hline()



def main():
    #agent1 = MCTSAgent(GreedyHeuristicAgent(), 1000, 5)
    # agent1 = MCTSAgent(RandomAgent(3), 1000, 5)
    agent1 = MinimaxTimeAgent(2)
    # agent2 = MinimaxTimeAgent(2)
    agent2 = MinimaxAgent(5)
    #agent2 = MinimaxAgent(5)
    # agent2 = MCTSAgent(GreedyHeuristicAgent(), 4000, 4)
    # agent2 = MCTSNoRolloutAgent(2000, 12)
    #agent1 = RandomAgent(3)
    #agent2 = GreedyHeuristicAgent()
    
    config = cx.Config(8, 7, 4)
    outcomes = np.zeros(3, dtype=int)
    games = 1
    for _ in range(games):
        agent1.reset()
        agent2.reset()
        outcome = cx.play(config, agent1, agent2, 0)
        outcomes[outcome] += 1

        # swap players
        agent1, agent2 = agent2, agent1
        outcomes[[1,2]] = outcomes[[2,1]]
    plt.plot(agent2.min_depth)
    plt.plot(agent2.max_depth)
    plt.legend("Nerovnost prohledávání")
    plt.xlabel("Číslo tahu")
    plt.ylabel("Hloubka")
    plt.legend(["minimální", "maximální"])
    plt.show()

    table = [
        ("Agent name", "Wins"),
        (str(agent1), str(outcomes[1])),
        (str(agent2), str(outcomes[2])),
        ("Draws", str(outcomes[0]))
    ]    
    print_table(table)
    # return

    jmena = str(agent1), str(agent2)
    fig, axs = plt.subplots(1, 2)

    axs[0].plot(agent1.value_history)
    axs[0].plot(agent2.value_history)
    axs[0].legend(jmena)
    axs[0].set_title("Value function")
    axs[0].set_ylabel("Predicted value")
    axs[0].set_xlabel("Turn number")
    axs[0].set_ylim(-1.1, 1.1)

    axs[1].plot(agent1.thinking_times)
    axs[1].plot(agent2.thinking_times)
    axs[1].legend(jmena)
    axs[1].set_title("Thinking time")
    axs[1].set_ylabel("Seconds")
    axs[1].set_xlabel("Turn number")
    axs[1].set_ylim(bottom=-0.1, top=max(max(agent1.thinking_times), max(agent2.thinking_times), 2.1))

    plt.show()



if __name__ == "__main__":
    main()