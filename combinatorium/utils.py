from combinatorium.interfaces import Agent, Game, Board
from combinatorium.games import ConnectFour, TicTacToe
from combinatorium.agents import (
    RandomAgent,
    MinimaxAgent,
    AlphaBetaAgent,
    NegamaxAgent,
    HumanAgent,
    MCTSAgent,
    AlphaZeroTicTacToe,
    AlphaZeroConnectFour,
)

import yaml


AGENTS = {
    "Human": HumanAgent,
    "Random": RandomAgent,
    "Minimax": MinimaxAgent,
    "Negamax": NegamaxAgent,
    "AlphaBeta": AlphaBetaAgent,
    "MCTS": MCTSAgent,
    "AZ_TicTacToe": AlphaZeroTicTacToe,
    "AZ_ConnectFour": AlphaZeroConnectFour,
}

GAMES = {
    "TicTacToe": TicTacToe,
    "ConnectFour": ConnectFour,
}


def load_agent(name: str, kwargs: dict) -> Agent:
    agent = AGENTS[name](**kwargs)
    if isinstance(agent, AlphaZeroTicTacToe):
        agent.load_model("./combinatorium/pretrained/az_tictactoe_500_100.pt")
    if isinstance(agent, AlphaZeroConnectFour):
        agent.load_model("./combinatorium/pretrained/az_connectfour_1000_200.pt")

    return AGENTS[name](**kwargs)


def load_game(name: str, player_one: str, player_two: str, config_file: str) -> Game:
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)

    p1 = load_agent(player_one, config[name].get(player_one, {}))
    p2 = load_agent(player_two, config[name].get(player_two, {}))

    return GAMES[name](p1, p2)
