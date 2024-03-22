from combinatorium.base import Agent, Board

import random


class RandomAgent(Agent):
    """An agent that selects actions randomly from the available options.

    This agent implements a simple strategy of choosing a random valid action from the list
    of possible moves on the current board.
    """

    def __init__(self) -> None:
        """Initialize an agent that plays randomly."""
        super().__init__()

    def act(self, board: Board) -> int:
        """Choose a random action given a board.

        Args:
            board (Board): The current state of the game board.

        Return:
            int: A randomly chosen action from the list of possible actions.
        """
        return random.choice(board.possible_actions)
