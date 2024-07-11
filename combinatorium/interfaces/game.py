from combinatorium.interfaces.board import Board

from typing import Protocol


class Game(Protocol):
    """This class provides the core functionality for managing a game's state, including
    the board, history of moves, and game flow.
    """

    @property
    def board(self) -> Board:
        """Return the current state of the game.

        Returns:
            Board: A representation of the current game board.
        """
        ...

    def reset(self) -> None:
        """Reset the game to its initial state."""
        ...

    def run(self, verbose: int = 0) -> None:
        """Start and run the game loop.

        Args:
          verbose (int, optional): Controls the verbosity of the game's output during execution. Defaults to 0.
              * 0: No additional output.
              * >= 1: Print the end result of the game (winner, score, etc.).
              * >= 2: Print the selected actions throughout the game.
        """
        ...
