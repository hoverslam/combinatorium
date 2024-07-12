from combinatorium.interfaces import Agent
from combinatorium.games import *
from combinatorium.agents import *

import yaml


AGENTS = ["Human", "Random", "Minimax", "Negamax", "AlphaBeta", "MCTS", "AlphaZero"]
GAMES = ["TicTacToe", "ConnectFour"]


class AgentConfig:
    """Class for loading and configuring game agents based on a configuration file."""

    def __init__(self, fname: str) -> None:
        """Initializes the AgentConfig with the given configuration file.

        Args:
            fname (str): The filename of the configuration file.
        """
        self._config = self._read_config_file(fname)

    def load_agent(self, game: str) -> Agent:
        """Loads and returns an agent for the specified game.

        Args:
            game (str): The name of the game for which to load the agent.

        Returns:
            Agent: An instance of the agent class configured for the specified game.
        """
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
        """Reads and parses the configuration file.

        Args:
            fname (str): The filename of the configuration file.

        Returns:
            dict: The parsed configuration as a dictionary.
        """
        with open(fname, "r") as file:
            return yaml.safe_load(file)

