from combinatorium.games import Board
from combinatorium.agents import Agent

import time


class HumanAgent(Agent):
    """An agent that represents a human player in the game.

    It prompts the user for an action to take and checks for the validity of the input.
    """

    def __init__(self) -> None:
        """Initialize a new human player."""
        super().__init__()

    def act(self, board: Board) -> int:
        """Prompts the user for a valid action and returns the integer action value.

        Args:
            board (Board): The current state of the game board.

        Returns:
            int: The integer value representing the chosen action by the human player.
        """
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
        print(f"# Selected action: {action} ({runtime=:.3f}s)\n")

        return action

    def __str__(self) -> str:
        return "Human"
