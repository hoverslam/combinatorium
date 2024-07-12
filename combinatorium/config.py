from combinatorium.interfaces import Agent
from combinatorium.games import *
from combinatorium.agents import *

import yaml


AGENTS = ["Human", "Random", "Minimax", "Negamax", "AlphaBeta", "MCTS", "AlphaZero"]
GAMES = ["TicTacToe", "ConnectFour"]


class AgentConfig:

    def __init__(self, fname: str) -> None:
        self._config = self._read_config_file(fname)

    def load_agent(self, game: str) -> Agent:
        cls_name = self._config[game]["class_name"]     
        if self._config[game].get("settings") is not None:
            agent = globals()[cls_name](**self._config[game]["settings"])
        else:
            agent = globals()[cls_name]()   
                 
        if self._config[game].get("pretrained") is not None:
            fname = f"./combinatorium/pretrained/{self._config[game]["pretrained"]}"
            agent.load_model(fname)
            
        return agent

    def _read_config_file(self, fname: str) -> dict:
        with open(fname, "r") as file:
            return yaml.safe_load(file)

