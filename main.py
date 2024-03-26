import argparse

from combinatorium.utils import GAMES, AGENTS, load_game


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Combinatorium", description="A collection of combinatorial games."
    )
    parser.add_argument(
        "-g",
        required=True,
        type=str,
        choices=GAMES.keys(),
        help=f"select a game",
    )
    parser.add_argument(
        "-p1",
        required=True,
        type=str,
        choices=AGENTS.keys(),
        help="select an agent for player one",
    )
    parser.add_argument(
        "-p2",
        required=True,
        type=str,
        choices=AGENTS.keys(),
        help="select an agent for player two",
    )

    args = parser.parse_args()
    game = load_game(args.g, args.p1, args.p2, "./combinatorium/config/agents.yaml")
    game.reset()
    game.run()
