from combinatorium.games.base import Board

from typing import Protocol


class Agent(Protocol):
    """This class defines the core functionality for an agent that can act within a game
    environment by selecting actions based on the current board state.
    """

    def act(self, board: Board) -> int:
        """Choose an action to take on the given board.

        Args:
            board (Board): The current state of the game board.

        Return:
            int: The action that the agent has chosen to take.
        """
        ...
