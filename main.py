from combinatorium import ConnectFour, TicTacToe
from combinatorium.agents import RandomAgent, MinimaxAgent, AlphaBetaAgent

p1 = AlphaBetaAgent(5)
p2 = RandomAgent()
g = ConnectFour(p1, p2)
g.reset()
g.run()
