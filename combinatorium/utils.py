from combinatorium.base import Agent, Game
from combinatorium import ConnectFour, TicTacToe
from combinatorium.agents import (
    RandomAgent,
    MinimaxAgent,
    AlphaBetaAgent,
    NegamaxAgent,
    HumanAgent,
    MCTSAgent,
)

import yaml


AGENTS = {
    "Human": HumanAgent,
    "Random": RandomAgent,
    "Minimax": MinimaxAgent,
    "Negamax": NegamaxAgent,
    "AlphaBeta": AlphaBetaAgent,
    "MCTS": MCTSAgent,
}

GAMES = {
    "TicTacToe": TicTacToe,
    "ConnectFour": ConnectFour,
}


def load_agent(name: str, kwargs: dict) -> Agent:
    return AGENTS[name](**kwargs)


def load_game(name: str, player_one: str, player_two: str, config_file: str) -> Game:
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)

    p1 = load_agent(player_one, config[name].get(player_one, {}))
    p2 = load_agent(player_two, config[name].get(player_two, {}))

    return GAMES[name](p1, p2)
