from combinatorium.interfaces import Board

from typing import Protocol


class Agent(Protocol):
    """This class defines the core functionality for an agent that can act within a game
    environment by selecting actions based on the current board state.
    """

    def act(self, board: Board, verbose: int = 0) -> int:
        """Choose an action to take on the given board.

        Args:
            board (Board): The current state of the game board.
            verbose (int, optional): Controls the verbosity of the agent's output. Defaults to 0.
                * 0: No additional output.
                * >= 2: Print the selected action.

        Return:
            int: The action that the agent has chosen to take.
        """
        ...
