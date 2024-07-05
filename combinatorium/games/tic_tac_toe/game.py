from combinatorium.interfaces import Agent, Board
from combinatorium.games.tic_tac_toe.board import TicTacToeBoard


class TicTacToe:
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
    def board(self) -> Board:
        return self._board

    def reset(self) -> None:
        self._board = TicTacToeBoard(self._board_size)
        self._players = {1: self._player_one, -1: self._player_two}
        self._round = 1

    def run(self) -> None:
        finished, _ = self._board.evaluate()

        while not finished:
            print(self, end="\n")
            action = self._players[self._board.player].act(self._board)
            new_board = self._board.move(action)
            finished, _ = new_board.evaluate()      

            self._board = new_board
            self._round += 1
        
        self._show_final_result()        
        
    def _show_final_result(self) -> None:
        """Print the game's outcome to the console.
        """
        finished, result = self._board.evaluate()
        if finished:
            print(f"{10 * "="} Game finished after {self._round - 1} rounds {10 * "="}")
            print(self._board)  
            if result == 0:
                print("Result: Draw")
            else:
                player_symbol = self._board.player_to_string(result)
                player_type = str(self._players[result])
                print(f"Result: {player_symbol} ({player_type})")        

    def __str__(self) -> str:
        player_symbol = self._board.player_to_string(self._board.player)
        player_type = str(self._players[self._board.player])
        string = f"# {self._round}: Player {player_symbol} ({player_type})" + "\n"
        string += str(self._board)
        for a in self._board.possible_actions:
            string += "\n" + f"{a} -> {self._board.action_to_string(a)}"

        return string
