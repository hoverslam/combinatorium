from __future__ import annotations
from combinatorium.games import Board
from combinatorium.agents import Agent

import time
import math
import random
import numpy as np


class MCTSAgent(Agent):
    """An agent that uses Monte Carlo tree search (MCTS) to make decisions in games.

    This agent implements the MCTS algorithm to select the best action based on simulating random
    games while balancing exploration and exploitation of different strategies.

    See: https://en.wikipedia.org/wiki/Monte_Carlo_tree_search
    """

    def __init__(self, search_time: int = 60) -> None:
        """Initialize a MCTS agent.

        Args:
            search_time (int, optional): The amount of time in seconds to spend searching for the
                best move. Defaults to 60.
        """
        super().__init__()
        self._search_time = search_time

    def act(self, board: Board) -> int:
        start_runtime = time.time()

        if len(board.possible_actions) == 1:
            action = board.possible_actions[0]
        else:
            root = MCTSNode(board, None)
            end_search_time = time.time() + self._search_time

            # MCTS loop
            while time.time() < end_search_time:
                current = root
                nodes = [current]

                self._selection(nodes)
                self._expansion(nodes)
                result = self._rollout(nodes)
                self._backpropagation(nodes, result)

            action = self._find_best_action(root)

        runtime = time.time() - start_runtime
        print(f"# Selected action: {action} ({runtime=:.3f}s)\n")

        return action

    def _selection(self, nodes: list[MCTSNode]) -> None:
        """Select the most promising node in the tree using the UCT score.

        Args:
            nodes (list[MCTSNode]): A list of nodes representing the current search path.
        """
        current = nodes[-1]
        while not current.is_leaf():
            scores = np.array([child.score for child in current._children])
            best_idx = np.random.choice(np.flatnonzero(scores == np.max(scores)))
            current = current._children[best_idx]
            nodes.append(current)

    def _expansion(self, nodes: list[MCTSNode]) -> None:
        """Expand the tree by adding a new child node for each possible action from the current state.

        Args:
            nodes (list[MCTSNode]): A list of nodes representing the current search path.
        """
        current = nodes[-1]
        board = current._board
        finished, _ = board.evaluate()
        if current._visits == 0 and not finished:
            current.expand()
            current = random.choice(current._children)
            nodes.append(current)

    def _rollout(self, nodes: list[MCTSNode]) -> int:
        """Simulate a random playout from the current state to estimate the win rate.

        Args:
            nodes (list[MCTSNode]): A list of nodes representing the current search path.

        Returns:
            int: The outcome of the simulated playout (1 = player 1, -1 = player 2 or 0 = draw).
        """
        current = nodes[-1]
        board = current._board
        finished, result = board.evaluate()
        while not finished:
            action = random.choice(board.possible_actions)
            board = board.move(action)
            finished, result = board.evaluate()

        return result

    def _backpropagation(self, nodes: list[MCTSNode], result: int) -> None:
        """Update the win and visit counts of all nodes in the search path based on the simulation result.

        Args:
            nodes (list[MCTSNode]): A list of nodes representing the current search path.
            result (int): The outcome of the simulated playout (1 = player 1, -1 = player 2 or 0 = draw).
        """
        for node in reversed(nodes):
            node.update(result)

    def _find_best_action(self, root: MCTSNode) -> int:
        visits = np.array([child._visits for child in root._children])
        best_idx = np.random.choice(np.flatnonzero(visits == np.max(visits)))
        action = root._actions[best_idx]

        return action

    def __str__(self) -> str:
        return f"MCTS, search_time={self._search_time}s"


class MCTSNode:
    """A class representing a node in the MCTS tree."""

    def __init__(self, board: Board, parent: MCTSNode | None) -> None:
        """Initialize a node in the MCTS tree.

        Args:
            board (Board): The game board state represented by this node.
            parent (MCTSNode | None): The parent node of this node in the tree.
        """
        self._board = board
        self._parent = parent
        self._wins = 0.0
        self._visits = 0
        self._c = math.sqrt(2)
        self._children = []
        self._actions = []

    @property
    def score(self) -> float:
        """Calculate the 'Upper Confidence Bound 1 applied to trees' (UCT) score of this node.

        Returns:
            float: The UCT score used for node selection in MCTS.
        """
        if self._parent is None:
            return 0.0

        if self._visits == 0:
            return float("inf")

        score = self._wins / self._visits + self._c * math.sqrt(
            math.log(self._parent._visits) / self._visits
        )

        return score

    @property
    def win_rate(self) -> float:
        """Calculate the win rate of this node, which is the number of wins divided by the number of visits.

        Returns:
            float: The win rate of this node.
        """
        if self._visits == 0:
            return 0.0
        else:
            return self._wins / self._visits

    def is_leaf(self) -> bool:
        """Check if this node is a leaf node (i.e. has no children).

        Returns:
            bool: True if the node is a leaf, False otherwise.
        """
        return True if (len(self._children) == 0) else False

    def update(self, result: int) -> None:
        """Update the win and visit counts of this node based on the simulation result.

        Args:
            result (int): The outcome of the simulated playout (1 = player 1, -1 = player 2 or 0 = draw).
        """
        # Since the action that leads to this node is performed by the other player, the result must
        # also be seen from the perspective of the other player.
        self._visits += 1
        self._wins += 0.5 if (result == 0) else int(self._board.player != result)

    def expand(self) -> None:
        """Expand this node by adding child nodes for each possible action from the current state."""
        for action in self._board.possible_actions:
            new_board = self._board.move(action)
            child = MCTSNode(new_board, self)
            self._children.append(child)
            self._actions.append(action)
