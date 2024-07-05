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

    def run(self) -> None:
        """Start and run the game loop."""
        ...
