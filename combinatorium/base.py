from __future__ import annotations
from abc import ABC, abstractmethod

import re

import numpy as np


class Game(ABC):
    """Abstract base class representing a general game.

    This class provides the core functionality for managing a game's state, including
    the board, history of moves, and game flow.
    """

    @property
    @abstractmethod
    def board(self) -> Board:
        """Return the current state of the game.

        Returns:
            Board: A representation of the current game board.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset the game to its initial state."""
        pass

    @abstractmethod
    def run(self) -> None:
        """Start and run the game loop."""
        pass


class Board(ABC):
    """Abstract base class representing a game board.

    This class provides the core functionality for managing the state of a game board,
    including representing the board, tracking the current player, handling moves, and
    evaluating game outcomes.
    """

    @property
    @abstractmethod
    def state(self) -> np.ndarray:
        """Return the current state of the board.

        Return:
            np.ndarray: A NumPy array representing the board's state.
        """
        pass

    @property
    @abstractmethod
    def player(self) -> int:
        """Return the player whose turn it is.

        Return:
            int: An integer representing the current player (player 1 = 1, player 2 = -1).
        """
        pass

    @property
    @abstractmethod
    def possible_actions(self) -> list[int]:
        """Return a list of valid actions that can be taken on the current board.

        Return:
            list[int]: A list of integers representing valid actions.
        """
        pass

    @property
    @abstractmethod
    def heuristic_value(self) -> float:
        """Return the heuristic value of the current board state.

        This value represents an estimate of the desirability of the current state.

        Return:
            float: The heuristic value of the board state.
        """
        pass

    @abstractmethod
    def move(self, action: int) -> Board:
        """Apply a move to the board and return a new board representing the updated state.

        Args:
            action (int): The action to be taken on the board.

        Return:
            Board: A new board instance representing the state after the move.
        """
        pass

    @abstractmethod
    def evaluate(self) -> tuple[bool, int]:
        """Evaluate the current board state and return the game outcome.

        Return:
            tuple[bool, int]: A tuple containing:
                - bool: True if the game has ended, False otherwise.
                - int: The winning player (1 or -1) if the game has ended, otherwise 0.
        """
        pass


class Agent(ABC):
    """Abstract base class representing a game-playing agent.

    This class defines the core functionality for an agent that can act within a game
    environment by selecting actions based on the current board state.
    """

    def __str__(self) -> str:
        class_name = type(self).__name__
        match = re.search(r"(.*)Agent$", class_name)

        return match.group(1) if match else class_name

    @abstractmethod
    def act(self, board: Board) -> int:
        """Choose an action to take on the given board.

        Args:
            board (Board): The current state of the game board.

        Return:
            int: The action that the agent has chosen to take.
        """
        pass
