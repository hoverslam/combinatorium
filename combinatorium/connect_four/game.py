from combinatorium.base import Game, Agent
from combinatorium.connect_four.board import ConnectFourBoard

import time


class ConnectFour(Game):
    """This class represents a Connect Four game between two players."""

    def __init__(self, player_one: Agent, player_two: Agent) -> None:
        """Initialize a new Connect Four game.

        Args:
            player_one (Agent): The agent representing player one.
            player_two (Agent): The agent representing player two.
        """
        super().__init__()
        self._player_one = player_one
        self._player_two = player_two

    @property
    def board(self) -> ConnectFourBoard:
        return self._board

    def reset(self) -> None:
        self._board = ConnectFourBoard()
        self._players = {1: self._player_one, -1: self._player_two}
        self._round = 1

    def run(self) -> None:
        finished, _ = self._board.evaluate()

        while not finished:
            print(self, end="\n")
            start_time = time.time()
            action = self._players[self._board.player].act(self._board)
            end_time = time.time()
            new_board = self._board.move(action)
            finished, _ = new_board.evaluate()      
            print(f"# Selected action: {action} (runtime={(end_time - start_time):.3f}s)\n")

            self._board = new_board
            self._round += 1
        
        self._show_final_results()        
        
    def _show_final_results(self) -> None:
        """Print the game's outcome to the console.
        """
        finished, result = self._board.evaluate()
        if finished:
            print(f"{10 * "="} Game finished after {self._round - 1} rounds {10 * "="}")
            print(f"Winner: {self._board.player_to_string(result)}")
            print(self._board)        

    def __str__(self) -> str:
        player_symbol = self._board.player_to_string(self._board.player)
        player_type = str(self._players[self._board.player])
        string = f"# {self._round}: Player {player_symbol} ({player_type})" + "\n"
        string += str(self._board)
        for a in self._board.possible_actions:
            string += "\n" + f"{a} -> {self._board.action_to_string(a)}"

        return string
