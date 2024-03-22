from combinatorium.base import Game, Agent
from combinatorium.tic_tac_toe.board import TicTacToeBoard


class TicTacToe(Game):
    """Tic-Tac-Toe game implementation."""

    def __init__(self, player_one: Agent, player_two: Agent, board_size: int = 3) -> None:
        """Initialize a new Tic-Tac-Toe game instance.

        Args:
            player_one (Agent): The agent representing player one.
            player_two (Agent): The agent representing player two.
            board_size (int, optional): The size of the Tic-Tac-Toe board. Defaults to 3.
        """
        super().__init__()
        self._board_size = board_size
        self._players = {1: player_one, -1: player_two}

    @property
    def board(self) -> TicTacToeBoard:
        return self._board

    def reset(self) -> None:
        self._board = TicTacToeBoard(self._board_size)
        self._round = 1

    def run(self) -> None:
        finished, result = self._board.evaluate()

        while not finished:
            cur_player = self._board.player

            action = self._players[cur_player].act(self._board)
            new_board = self._board.move(action)
            finished, result = new_board.evaluate()

            print(self, end="\n\n")
            self._board = new_board
            self._round += 1

        print(f"Game is finished! Winner: {self._board.player_to_string(result)}")
        print(self._board)

    def __str__(self) -> str:
        string = f"# {self._round}: Player {self._board.player_to_string(self._board.player)}" + "\n"
        string += self._board.__str__()
        for a in self._board.possible_actions:
            string += "\n" + f"{a} -> {self._board.action_to_string(a)}"

        return string
