from combinatorium.interfaces import Board

import time


class HumanAgent:
    """An agent that represents a human player in the game.

    It prompts the user for an action to take and checks for the validity of the input.
    """

    def __init__(self) -> None:
        """Initialize a new human player."""
        super().__init__()

    def act(self, board: Board, verbose: int = 0) -> int:
        start_runtime = time.time()

        action = -1
        while action not in board.possible_actions:
            try:
                action_str = input("Select an action and press Enter: ")
                action = int(action_str)
            except ValueError:
                print(f"Input '{action_str}' is not a valid action!")
                continue

            if action not in board.possible_actions:
                print(f"Input '{action}' is not a valid action!")

        runtime = time.time() - start_runtime
        if verbose >= 2:
            print(f"# Selected action: {action} ({runtime=:.3f}s)\n")

        return action

    def __str__(self) -> str:
        return "Human"
