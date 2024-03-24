import argparse

from combinatorium import ConnectFour, TicTacToe
from combinatorium.agents import (
    RandomAgent,
    MinimaxAgent,
    AlphaBetaAgent,
    NegamaxAgent,
    HumanAgent,
)


AGENTS = {
    "Human": HumanAgent(),
    "Random": RandomAgent(),
    "Minimax": MinimaxAgent(),
    "Negamax": NegamaxAgent(),
    "AlphaBeta": AlphaBetaAgent(),
}

GAMES = {
    "TicTacToe": TicTacToe,
    "ConnectFour": ConnectFour,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Combinatorium", description="A collection of combinatorial games."
    )
    parser.add_argument(
        "-g",
        required=True,
        type=str,
        choices=GAMES.keys(),
        help=f"select a game",
    )
    parser.add_argument(
        "-p1",
        required=True,
        type=str,
        choices=AGENTS.keys(),
        help="select an agent for player one",
    )
    parser.add_argument(
        "-p2",
        required=True,
        type=str,
        choices=AGENTS.keys(),
        help="select an agent for player two",
    )

    args = parser.parse_args()
    game = GAMES[args.g](AGENTS[args.p1], AGENTS[args.p2])
    game.reset()
    game.run()
