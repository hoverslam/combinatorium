from combinatorium.config import AgentConfig, AGENTS, GAMES
from combinatorium.games import *

import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Combinatorium", description="A collection of combinatorial games.")
    parser.add_argument(
        "-g",
        required=True,
        type=str,
        choices=GAMES,
        help=f"select a game",
    )
    parser.add_argument(
        "-p1",
        required=True,
        type=str,
        choices=AGENTS,
        help="select an agent for player one",
    )
    parser.add_argument(
        "-p2",
        required=True,
        type=str,
        choices=AGENTS,
        help="select an agent for player two",
    )

    args = parser.parse_args()
    player_1 = AgentConfig(f"./combinatorium/configs/{args.p1}.yaml").load_agent(args.g)
    player_2 = AgentConfig(f"./combinatorium/configs/{args.p2}.yaml").load_agent(args.g)
    game = globals()[args.g](player_1, player_2)
    game.reset()
    game.run(verbose=2)
