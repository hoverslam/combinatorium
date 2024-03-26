from __future__ import annotations
from combinatorium.games.base import Board

import numpy as np


class TicTacToeBoard(Board):
    """Tic-Tac-Toe board representation."""

    def __init__(self, size: int) -> None:
        """Initialize a new Tic-Tac-Toe board instance.

        Args:
            size (int): The size of the Tic-Tac-Toe board.
        """
        super().__init__()
        self._size = size
        self._state = np.zeros((size, size), dtype=np.int8)
        self._player = 1

    @property
    def state(self) -> np.ndarray:
        return self._state

    @property
    def player(self) -> int:
        return self._player

    @property
    def possible_actions(self) -> list[int]:
        return np.where(self._state.ravel() == 0)[0].tolist()

    @property
    def heuristic_value(self) -> float:
        return float(self.evaluate()[1])

    def move(self, action: int) -> TicTacToeBoard:
        # Check if the given action is valid
        if action not in self.possible_actions:
            raise ValueError(
                f"'{action}' is not a valid action. Possible actions: {self.possible_actions}."
            )

        # Create a new state updated by the given action
        new_state = self._state.copy()
        new_state.reshape(-1)[action] = self._player

        # Create a board with the new state and next player
        new_board = TicTacToeBoard(self._size)
        new_board._state = new_state
        new_board._player = -self._player

        return new_board

    def evaluate(self) -> tuple[bool, int]:
        # Check rows
        sum_rows = self._state.sum(axis=1)
        if (np.abs(sum_rows) == self._size).any():
            return (True, 1) if np.max(sum_rows) == self._size else (True, -1)

        # Check columns
        sum_cols = self._state.sum(axis=0)
        if (np.abs(sum_cols) == self._size).any():
            return (True, 1) if np.max(sum_cols) == self._size else (True, -1)

        # Check 1st diagonal
        sum_diag1 = np.trace(self._state)
        if (np.abs(sum_diag1) == self._size).any():
            return (True, 1) if np.max(sum_diag1) == self._size else (True, -1)

        # Check 2nd diagonal
        sum_diag2 = np.trace(np.fliplr(self._state))
        if (np.abs(sum_diag2) == self._size).any():
            return (True, 1) if np.max(sum_diag2) == self._size else (True, -1)

        # Check if board has no 0 cells
        if (self._state != 0).all():
            return (True, 0)

        return (False, 0)

    def __str__(self) -> str:
        string = 13 * "-"
        for row in self._state:
            string += "\n" + "| " + " | ".join(self.player_to_string(x) for x in row) + " |" + "\n"
            string += 13 * "-"

        return string

    def player_to_string(self, player: int) -> str:
        """Convert player number to a human-readable string representation.

        Args:
            player (int): The player number (1 for player 1, -1 for player 2, 0 for empty cell).

        Returns:
            str: The corresponding string representation ("X" for player 1, "O" for player 2, "-" for empty cell).
        """
        mapping = {1: "X", -1: "O", 0: "-"}

        return mapping[player]

    def action_to_string(self, action: int) -> str:
        """Convert an action (cell index) to a human-readable coordinate string.

        Args:
            action (int): The action representing a cell on the board (index).

        Returns:
            str: The coordinates of the cell in "row-column" format (1-based indexing).
        """
        row = action // self._size
        col = action % self._size

        return f"{row+1}-{col+1}"
