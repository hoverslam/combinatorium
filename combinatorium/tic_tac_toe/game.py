from combinatorium.base import Game, Agent
from combinatorium.tic_tac_toe.board import TicTacToeBoard

import time


class TicTacToe(Game):
    """Tic-tac-toe game implementation."""

    def __init__(self, player_one: Agent, player_two: Agent, board_size: int = 3) -> None:
        """Initialize a new Tic-tac-toe game instance.

        Args:
            player_one (Agent): The agent representing player one.
            player_two (Agent): The agent representing player two.
            board_size (int, optional): The size of the Tic-tac-toe board. Defaults to 3.
        """
        super().__init__()
        self._board_size = board_size
        self._player_one = player_one
        self._player_two = player_two

    @property
    def board(self) -> TicTacToeBoard:
        return self._board

    def reset(self) -> None:
        self._board = TicTacToeBoard(self._board_size)
        self._players = {1: self._player_one, -1: self._player_two}
        self._round = 1

    def run(self) -> None:
        finished, result = self._board.evaluate()

        while not finished:
            print(self, end="\n")
            start_time = time.time()
            action = self._players[self._board.player].act(self._board)
            end_time = time.time()
            new_board = self._board.move(action)
            finished, result = new_board.evaluate()
            print(f"# Selected action: {action} (runtime={(end_time - start_time):.3f}s)\n")

            self._board = new_board
            self._round += 1

        print(27 * "=")
        print(f"Game is finished! Winner: {self._board.player_to_string(result)}")
        print(self._board)

    def __str__(self) -> str:
        player_symbol = self._board.player_to_string(self._board.player)
        player_type = str(self._players[self._board.player])
        string = f"# {self._round}: Player {player_symbol} ({player_type})" + "\n"
        string += str(self._board)
        for a in self._board.possible_actions:
            string += "\n" + f"{a} -> {self._board.action_to_string(a)}"

        return string
