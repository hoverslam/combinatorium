from combinatorium.games.base import Board

from abc import ABC, abstractmethod


class Agent(ABC):
    """Abstract base class representing a game-playing agent.

    This class defines the core functionality for an agent that can act within a game
    environment by selecting actions based on the current board state.
    """

    @abstractmethod
    def act(self, board: Board) -> int:
        """Choose an action to take on the given board.

        Args:
            board (Board): The current state of the game board.

        Return:
            int: The action that the agent has chosen to take.
        """
        pass
