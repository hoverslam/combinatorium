from combinatorium.tic_tac_toe import TicTacToe
from combinatorium.agent import RandomAgent, MinimaxAgent, AlphaBetaAgent

p1 = AlphaBetaAgent(10)
p2 = MinimaxAgent(10)
g = TicTacToe(p1, p2)
g.reset()
g.run()
