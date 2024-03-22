from combinatorium.tic_tac_toe import TicTacToe
from combinatorium.agent import RandomAgent, MinimaxAgent


g = TicTacToe(player_one=RandomAgent(), player_two=MinimaxAgent(10))
g.reset()
g.run()
