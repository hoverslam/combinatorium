from combinatorium.interfaces import Board

import time
import random


class MinimaxAgent:
    """General purpose minimax agent.

    This agent implements the Minimax algorithm to select the best action based on maximizing its
    own score and minimizing the opponent's score, considering a specified search depth.

    See: https://en.wikipedia.org/wiki/Minimax
    """

    def __init__(self, depth: int = 5) -> None:
        """Initialize a new MinimaxAgent instance.

        Args:
            depth (int, optional): The maximum depth of the search tree explored by the Minimax algorithm.
                Defaults to 5.
        """
        super().__init__()
        self._depth = depth

    def act(self, board: Board) -> int:
        """Choose the best action for the current board state using Minimax.

        Args:
            board (Board): The current state of the game board.

        Returns:
            int: The action chosen by the agent.
        """
        start_runtime = time.time()

        # Since we want the best action for a given game state, we don't make an initial call to the
        # root. Instead, we call all childs (i.e. possible actions) and figure out the optimal
        # action from the values each child returns.
        values = {}
        for action in board.possible_actions:
            child = board.move(action)
            values[action] = board.player * self._minimax(
                board=child,
                depth=self._depth - 1,
                maximizing_player=not (True if board.player == 1 else False),
            )

        # Break ties randomly
        max_values = max(values.values())
        max_keys = [key for key, value in values.items() if value == max_values]
        action = max_keys[0] if (len(max_keys) == 1) else random.choice(max_keys)

        runtime = time.time() - start_runtime
        print(f"# Selected action: {action} ({runtime=:.3f}s)\n")

        return action

    def _minimax(self, board: Board, depth, maximizing_player: int) -> float:
        """Perform the Minimax recursive search to find the best action.

        Args:
            board (Board): The current board state.
            depth (int): The current depth in the search tree.
            maximizing_player (int): Flag indicating whether the current player is trying to
                maximize (True) or minimize (False) the score.

        Returns:
            float: The score for the best state found at the current depth.
        """
        if depth == 0 or board.evaluate()[0]:
            return board.heuristic_value

        if maximizing_player:
            value = float("-inf")
            for action in board.possible_actions:
                child = board.move(action)
                value = max(value, self._minimax(child, depth - 1, False))
            return value
        else:
            value = float("inf")
            for action in board.possible_actions:
                child = board.move(action)
                value = min(value, self._minimax(child, depth - 1, True))
            return value

    def __str__(self) -> str:
        return f"Minimax, depth={self._depth}"


class AlphaBetaAgent:
    """An agent that uses the alpha-beta pruning algorithm to make decisions.

    Alpha-beta pruning is a search algorithm optimization technique used in minimax decision trees. It
    eliminates portions of the search tree that cannot possibly influence the final decision, leading
    to significant efficiency gains.

    See: https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning
    """

    def __init__(self, depth: int = 5) -> None:
        """Initialize a new alpha-beta pruning agent.

        Args:
            depth (int, optional): The maximum depth of the search tree explored. Defaults to 5.
        """
        super().__init__()
        self._depth = depth

    def act(self, board: Board) -> int:
        """Choose the best action for the current board state using alpha-beta pruning.

        Args:
            board (Board): The current state of the game board.

        Returns:
            int: The action chosen by the agent.
        """
        start_runtime = time.time()

        # Since we want the best action for a given game state, we don't make an initial call to the
        # root. Instead, we call all childs (i.e. possible actions) and figure out the optimal
        # action from the values each child returns.
        values = {}
        for action in board.possible_actions:
            child = board.move(action)
            values[action] = board.player * self._alpha_beta(
                board=child,
                depth=self._depth - 1,
                alpha=float("-inf"),
                beta=float("inf"),
                maximizing_player=not (True if board.player == 1 else False),
            )

        # Break ties randomly
        max_values = max(values.values())
        max_keys = [key for key, value in values.items() if value == max_values]
        action = max_keys[0] if (len(max_keys) == 1) else random.choice(max_keys)

        runtime = time.time() - start_runtime
        print(f"# Selected action: {action} ({runtime=:.3f}s)\n")

        return action

    def _alpha_beta(
        self, board: Board, depth, alpha: float, beta: float, maximizing_player: int
    ) -> float:
        """Perform the Minimax recursive search with alpha-beta pruning to find the best action.

        Args:
            board (Board): The current board state.
            depth (int): The current depth in the search tree.
            alpha (float): The current minimum score guaranteed to the maximizing player.
            beta (float): The current maximum score guaranteed to the minimizing player.
            maximizing_player (int): Flag indicating whether the current player is trying to
                maximize (True) or minimize (False) the score.

        Returns:
            float: The score for the best state found at the current depth.
        """
        if depth == 0 or board.evaluate()[0]:
            return board.heuristic_value

        if maximizing_player:
            value = float("-inf")
            for action in board.possible_actions:
                child = board.move(action)
                value = max(value, self._alpha_beta(child, depth - 1, alpha, beta, False))
                alpha = max(alpha, value)
                if value > beta:  # beta cutoff
                    break
            return value
        else:
            value = float("inf")
            for action in board.possible_actions:
                child = board.move(action)
                value = min(value, self._alpha_beta(child, depth - 1, alpha, beta, True))
                beta = min(beta, value)
                if value < alpha:  # alpha cutoff
                    break
            return value

    def __str__(self) -> str:
        return f"AlphaBeta, depth={self._depth}"


class NegamaxAgent:
    """An agent that uses the negamax search algorithm to find the best action.

    It is a variant form of minimax search that relies on the zero-sum property of a two-player game.

    See: https://en.wikipedia.org/wiki/Negamax
    """

    def __init__(self, depth: int = 5) -> None:
        """Initialize a new negamax agent.

        Args:
            depth (int, optional): The maximum depth of the search tree explored. Defaults to 5.
        """
        super().__init__()
        self._depth = depth

    def act(self, board: Board) -> int:
        """Choose the best action for the current board.

        Args:
            board (Board): The current state of the game board.

        Returns:
            int: The action chosen by the agent.
        """
        start_runtime = time.time()

        # Since we want the best action for a given game state, we don't make an initial call to the
        # root. Instead, we call all childs (i.e. possible actions) and figure out the optimal
        # action from the values each child returns.
        values = {}
        for action in board.possible_actions:
            child = board.move(action)
            values[action] = -self._negamax(child, self._depth - 1, -board.player)

        # Break ties randomly
        max_values = max(values.values())
        max_keys = [key for key, value in values.items() if value == max_values]
        action = max_keys[0] if (len(max_keys) == 1) else random.choice(max_keys)

        runtime = time.time() - start_runtime
        print(f"# Selected action: {action} ({runtime=:.3f}s)\n")

        return action

    def _negamax(self, board: Board, depth, color: int) -> float:
        """Performs a negamax search on the given board state.

        Args:
            board (Board): The current board state.
            depth (int): The current depth in the search tree.
            color (int): The player's color (+1 = player 1 or -1: player 2).

        Returns:
            float: The score for the best state found at the current depth.
        """
        if depth == 0 or board.evaluate()[0]:
            return color * board.heuristic_value

        value = float("-inf")
        for action in board.possible_actions:
            child = board.move(action)
            value = max(value, -self._negamax(child, depth - 1, -color))
        return value

    def __str__(self) -> str:
        return f"Negamax, depth={self._depth}"
