from combinatorium.base import Agent, Board


class MCTSAgent(Agent):

    def act(self, board: Board) -> int:
        raise NotImplementedError
