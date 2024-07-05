from __future__ import annotations

from typing import Protocol

import numpy as np


class Board(Protocol):
    """This class provides the core functionality for managing the state of a game board,
    including representing the board, tracking the current player, handling moves, and
    evaluating game outcomes.
    """

    @property
    def state(self) -> np.ndarray:
        """Return the current state of the board.

        Return:
            np.ndarray: A NumPy array representing the board's state.
        """
        ...

    @property
    def player(self) -> int:
        """Return the player whose turn it is.

        Return:
            int: An integer representing the current player (player 1 = 1, player 2 = -1).
        """
        ...

    @property
    def possible_actions(self) -> list[int]:
        """Return a list of valid actions that can be taken on the current board.

        Return:
            list[int]: A list of integers representing valid actions.
        """
        ...

    @property
    def heuristic_value(self) -> float:
        """Return the heuristic value of the current board state.

        This value represents an estimate of the desirability of the current state.

        Return:
            float: The heuristic value of the board state.
        """
        ...

    def move(self, action: int) -> Board:
        """Apply a move to the board and return a new board representing the updated state.

        Args:
            action (int): The action to be taken on the board.

        Return:
            Board: A new board instance representing the state after the move.
        """
        ...

    def evaluate(self) -> tuple[bool, int]:
        """Evaluate the current board state and return the game outcome.

        Return:
            tuple[bool, int]: A tuple containing:
                - bool: True if the game has ended, False otherwise.
                - int: The winning player (1 or -1) if the game has ended, otherwise 0.
        """
        ...
