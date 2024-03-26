from __future__ import annotations
from combinatorium.base import Agent, Board

import time
import math
import random
import numpy as np


class MCTSAgent(Agent):

    def __init__(self, search_time: int = 60) -> None:
        super().__init__()
        self._search_time = search_time

    def act(self, board: Board) -> int:
        root = MCTSNode(board, None)
        end_time = time.time() + self._search_time

        # MCTS loop
        while time.time() < end_time:
            current = root
            nodes = [current]

            # Selection
            self._selection(nodes)

            # Expansion
            self._expansion(nodes)

            # Rollout
            result = self._rollout(nodes)

            # Backpropagation
            self._backpropagation(nodes, result)

        # Select optimal action
        win_rates = np.array([child.win_rate for child in root._children])
        highest_idx = np.random.choice(np.flatnonzero(win_rates == np.min(win_rates)))

        return root._actions[highest_idx]

    def _selection(self, nodes: list[MCTSNode]) -> None:
        current = nodes[-1]
        while not current.is_leaf():
            uct_scores = np.array([child.uct_score for child in current._children])
            highest_idx = np.random.choice(np.flatnonzero(uct_scores == np.max(uct_scores)))
            current = current._children[highest_idx]
            nodes.append(current)

    def _expansion(self, nodes: list[MCTSNode]) -> None:
        current = nodes[-1]
        board = current._board
        finished, _ = board.evaluate()
        if current._visits == 0 and not finished:
            current.expand()
            current = random.choice(current._children)
            nodes.append(current)

    def _rollout(self, nodes: list[MCTSNode]) -> int:
        current = nodes[-1]
        board = current._board
        finished, result = board.evaluate()
        while not finished:
            action = random.choice(board.possible_actions)
            board = board.move(action)
            finished, result = board.evaluate()

        return result

    def _backpropagation(self, nodes: list[MCTSNode], result: int) -> None:
        for node in reversed(nodes):
            node.update(result)

    def __str__(self) -> str:
        return f"MCTS, search_time={self._search_time}s"


class MCTSNode:

    def __init__(self, board: Board, parent: MCTSNode | None) -> None:
        self._board = board
        self._parent = parent
        self._wins = 0.0
        self._visits = 0
        self._c = math.sqrt(2)
        self._children = []
        self._actions = []

    @property
    def uct_score(self) -> float:
        if self._parent is None:
            return 0.0

        if self._visits == 0:
            return float("inf")

        # Upper Confidence Bound 1 applied to trees
        score = self._wins / self._visits + self._c * math.sqrt(
            math.log(self._parent._visits) / self._visits
        )

        return score

    @property
    def win_rate(self) -> float:
        if self._visits == 0:
            return 0.0
        else:
            return self._wins / self._visits

    def is_leaf(self) -> bool:
        return True if (len(self._children) == 0) else False

    def update(self, result: int) -> None:
        self._visits += 1
        self._wins += 0.5 if (result == 0) else int(self._board.player == result)

    def expand(self) -> None:
        for action in self._board.possible_actions:
            new_board = self._board.move(action)
            child = MCTSNode(new_board, self)
            self._children.append(child)
            self._actions.append(action)
