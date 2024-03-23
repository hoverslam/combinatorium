from combinatorium import ConnectFour, TicTacToe
from combinatorium.agent import RandomAgent, MinimaxAgent, AlphaBetaAgent

p1 = RandomAgent()
p2 = RandomAgent()
g = ConnectFour(p1, p2)
g.reset()
g.run()
