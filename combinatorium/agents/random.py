from combinatorium.interfaces import Board

import time
import random


class RandomAgent:
    """An agent that selects actions randomly from the available options.

    This agent implements a simple strategy of choosing a random valid action from the list
    of possible moves on the current board.
    """

    def __init__(self) -> None:
        """Initialize an agent that plays randomly."""
        super().__init__()

    def act(self, board: Board, verbose: int = 0) -> int:
        start_runtime = time.time()
        action = random.choice(board.possible_actions)
        runtime = time.time() - start_runtime

        if verbose >= 2:
            print(f"# Selected action: {action} ({runtime=:.3f}s)\n")

        return action

    def __str__(self) -> str:
        return "Random"
