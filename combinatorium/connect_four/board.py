from __future__ import annotations
from combinatorium.base import Board

import numpy as np
from scipy.signal import convolve


class ConnectFourBoard(Board):
    """Represents a Connect Four game board."""

    def __init__(self) -> None:
        """Initialize a new instance of a Connect Four board with size 6x7."""
        super().__init__()
        self._size = (6, 7)
        self._state = np.zeros(self._size, dtype=np.int8)
        self._player = 1

    @property
    def state(self) -> np.ndarray:
        return self._state

    @property
    def player(self) -> int:
        return self._player

    @property
    def possible_actions(self) -> list[int]:
        return np.where(self._get_free_row_indices() >= 0)[0].tolist()

    @property
    def heuristic_value(self) -> float:
        # TODO: find better heuristic value
        return float(self.evaluate()[1])

    def move(self, action: int) -> ConnectFourBoard:
        # Check if the given action is valid
        if action not in self.possible_actions:
            raise ValueError(
                f"'{action}' is not a valid action. Possible actions: {self.possible_actions}."
            )

        # Create a new state updated by the given action
        new_state = self._state.copy()
        row_idx = self._get_free_row_indices()[action]
        new_state[row_idx, action] = self._player

        # Create a board with the new state and next player
        new_board = ConnectFourBoard()
        new_board._state = new_state
        new_board._player = -self._player

        return new_board

    def evaluate(self) -> tuple[bool, int]:
        # Compute the convolution of the current state with each "winning" kernel
        convolutions = [
            convolve(self._state, np.ones((1, 4), dtype=np.int8)),  # row
            convolve(self._state, np.ones((4, 1), dtype=np.int8)),  # column
            convolve(self._state, np.eye(4, dtype=np.int8)),  # 1st diagonal
            convolve(self._state, np.fliplr(np.eye(4, dtype=np.int8))),  # 2nd diagonal
        ]

        # Check player one
        for conv in convolutions:
            if (conv == 4).any():
                return (True, 1)

        # Check player two
        for conv in convolutions:
            if (conv == -4).any():
                return (True, -1)

        # Check if board has no 0 cells
        if (self._state != 0).all():
            return (True, 0)

        return (False, 0)

    def _get_free_row_indices(self) -> np.ndarray:
        """Return the indices of the bottom empty row in each column.

        This method calculates the number of filled cells in each column and returns an array
        indicating the index of the bottom empty cell (or -1 if the column is full).

        Returns:
            np.ndarray: An array containing the indices of the bottom empty row in each column.
        """
        col_height = np.sum(np.abs(self._state), axis=0)

        return self._size[0] - col_height - 1

    def __str__(self) -> str:
        string = 36 * "-"
        for row in self._state:
            string += "\n" + "| " + " | ".join(self.player_to_string(x) for x in row) + " |" + "\n"
            string += 36 * "-"

        return string

    def player_to_string(self, player: int) -> str:
        """Convert player number to a human-readable string representation.

        Args:
            player (int): The player number (1 for player 1, -1 for player 2, 0 for empty cell).

        Returns:
            str: The corresponding string representation.
        """
        mapping = {1: chr(0x1F534), -1: chr(0x1F7E1), 0: "  "}

        return mapping[player]

    def action_to_string(self, action: int) -> str:
        """Convert an action to a human-readable string.

        Args:
            action (int): The action representing a column on the board.

        Returns:
            str: The column corresponding to the action (1-based indexing).
        """

        return f"Column {action+1}"
