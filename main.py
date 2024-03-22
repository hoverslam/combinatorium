from combinatorium.tic_tac_toe import TicTacToe
from combinatorium.agent import RandomAgent


g = TicTacToe(player_one=RandomAgent(), player_two=RandomAgent())
g.reset()
g.run()
