from __future__ import annotations
from combinatorium.games import Board

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
        """Return the heuristic value of the current board state.

        The heuristic function weights the coins based on their position on the board. It is
        simple but fast to compute.
        Found in this paper: Kang, Wang, Hu (2019) Research on Different Heuristics for Minimax
        Algorithm Insight from Connect-4 Game.

        Return:
            float: The heuristic value of the board state.
        """
        # If it is a terminal state, return the result
        finished, result = self.evaluate()
        if finished:
            return result

        # Otherwise use heuristic function
        weights = np.array(
            [
                [3, 4, 5, 7, 5, 4, 3],
                [4, 6, 8, 10, 8, 6, 4],
                [5, 8, 11, 13, 11, 8, 5],
                [5, 8, 11, 13, 11, 8, 5],
                [4, 6, 8, 10, 8, 6, 4],
                [3, 4, 5, 7, 5, 4, 3],
            ]
        )
        weighted_sum = np.sum(self._state * weights)
        normalized_weighted_sum = weighted_sum / np.sum(weights)

        return normalized_weighted_sum.item()

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
        string += "\n" + "  " + "   ".join(f"C{str(i)}" for i in range(self._size[1])) + "\n"

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
            str: The column corresponding to the action (0-based indexing).
        """

        return f"C{action}"
