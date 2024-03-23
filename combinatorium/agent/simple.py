from combinatorium.base import Agent, Board


class MinimaxAgent(Agent):
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
        maximizing_player = True if board.player == 1 else False
        _, action = self._minimax(board, self._depth, maximizing_player)

        return action

    def _minimax(self, board: Board, depth, maximizing_player: int) -> tuple[float, int]:
        """Perform the Minimax recursive search to find the best action.

        Args:
            board (Board): The current board state.
            depth (int): The current depth in the search tree.
            maximizing_player (int): Flag indicating whether the current player is trying to
                maximize (True) or minimize (False) the score.

        Returns:
            tuple[float, int]: A tuple containing:
                - float: The score for the best state found at the current depth.
                - int: The action that leads to the best state.
        """
        if depth == 0 or board.evaluate()[0]:
            return board.heuristic_value, -1  # Terminal state has no valid action

        if maximizing_player:
            value = float("-inf")
            best_action = -1  # Initialize with invalid action
            for action in board.possible_actions:
                child = board.move(action)
                child_value, _ = self._minimax(child, depth - 1, False)
                value = max(value, child_value)
                if value == child_value:  # If child value is optimal, it has the optimal action.
                    best_action = action
            return value, best_action
        else:
            value = float("inf")
            best_action = -1
            for action in board.possible_actions:
                child = board.move(action)
                child_value, _ = self._minimax(child, depth - 1, True)
                value = min(value, child_value)
                if value == child_value:
                    best_action = action
            return value, best_action


class AlphaBetaAgent(Agent):
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
        _, action = self._alpha_beta(
            board=board,
            depth=self._depth,
            alpha=float("-inf"),
            beta=float("inf"),
            maximizing_player=(True if board.player == 1 else False),
        )

        return action

    def _alpha_beta(
        self, board: Board, depth, alpha: float, beta: float, maximizing_player: int
    ) -> tuple[float, int]:
        """Perform the Minimax recursive search with alpha-beta pruning to find the best action.

        Args:
            board (Board): The current board state.
            depth (int): The current depth in the search tree.
            alpha (float): The current minimum score guaranteed to the maximizing player.
            beta (float): The current maximum score guaranteed to the minimizing player.
            maximizing_player (int): Flag indicating whether the current player is trying to
                maximize (True) or minimize (False) the score.

        Returns:
            tuple[float, int]: A tuple containing:
                - float: The score for the best state found at the current depth.
                - int: The action that leads to the best state.
        """
        if depth == 0 or board.evaluate()[0]:
            return board.heuristic_value, -1  # Terminal state has no valid action

        if maximizing_player:
            value = float("-inf")
            best_action = -1  # Initialize with invalid action
            for action in board.possible_actions:
                child = board.move(action)
                child_value, _ = self._alpha_beta(child, depth - 1, alpha, beta, False)
                value = max(value, child_value)
                alpha = max(alpha, value)
                if value > beta:  # cutoff
                    break
                if value == child_value:  # If child value is optimal, it has the optimal action
                    best_action = action
            return value, best_action
        else:
            value = float("inf")
            best_action = -1
            for action in board.possible_actions:
                child = board.move(action)
                child_value, _ = self._alpha_beta(child, depth - 1, alpha, beta, True)
                value = min(value, child_value)
                beta = min(beta, value)
                if value < alpha:
                    break
                if value == child_value:
                    best_action = action
            return value, best_action
