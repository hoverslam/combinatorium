from __future__ import annotations

from combinatorium.interfaces import Board
from combinatorium.models import AlphaZeroResNet
from combinatorium.games.tic_tac_toe.board import TicTacToeBoard
from combinatorium.games.connect_four.board import ConnectFourBoard

import time
import multiprocessing as mp
from typing import Callable
from collections import deque
from abc import ABC, abstractmethod
from copy import deepcopy

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import Dataset, DataLoader


class AlphaZeroNode:
    """Represents a node in the AlphaZero MCTS tree."""

    def __init__(self, board: Board, parent: AlphaZeroNode | None) -> None:
        """Initializes an AlphaZeroNode.

        Args:
            board (Board): The game board at this node.
            parent (AlphaZeroNode | None): The parent node of this node.
        """
        self._board = board
        self._parent = parent
        self._children = {}  # dictionary with <action, child> as key-value pairs

    @property
    def children(self) -> list:
        """Returns the child nodes.

        Raises:
            ValueError: If the node is a leaf.

        Returns:
            list: List of child nodes.
        """
        if self.is_leaf():
            raise ValueError("This unexpanded node has no child nodes.")

        return list(self._children.values())

    @property
    def actions(self) -> list:
        """Returns the possible actions from this node.

        Raises:
            ValueError: If the node is a leaf.

        Returns:
            list: List of possible actions.
        """
        if self.is_leaf():
            raise ValueError("This unexpanded node has no actions because it has no child nodes.")

        return list(self._children.keys())

    @property
    def visits(self) -> torch.Tensor:
        """Returns the visit statistics of this node.

        Raises:
            ValueError: If the node is a leaf.

        Returns:
            torch.Tensor: Tensor of visit counts.
        """
        if self.is_leaf():
            raise ValueError("This unexpanded node has no visit statistics because it has no child nodes.")

        return self._visits

    def compute_scores(self, exploration_rate: float) -> torch.Tensor:
        """Computes the scores for each child node.

        Args:
            exploration_rate (float): The exploration rate for the PUCT algorithm.

        Raises:
            ValueError: If the node is a leaf.

        Returns:
            torch.Tensor: Tensor of computed scores.
        """
        if self.is_leaf():
            raise ValueError("This unexpanded node cannot compute scores because it has no child nodes.")

        # The PUCT algorithm from the paper with a small amount added to the total visit count. This has the effect that
        # with 0 visits the score equals the prior probabilities
        puct_scores = (
            exploration_rate * self._prior_probs * ((torch.sqrt(self._visits.sum()) + 1e-10) / (1 + self._visits))
        )

        return self._mean_action_values + puct_scores

    def is_leaf(self) -> bool:
        """Checks if the node is a leaf.

        Returns:
            bool: True if the node is a leaf, otherwise False.
        """
        return len(self._children) == 0

    def is_terminal(self) -> bool:
        """Checks if the node is a terminal node.

        Returns:
            bool: True if the node is terminal, otherwise False.
        """
        return self._board.evaluate()[0]

    def expand(self, model: nn.Module, board_encoding_fn: Callable[[Board], torch.Tensor]) -> tuple[float, int]:
        """Expands the node by generating child nodes.

        Args:
            model (nn.Module): The neural network model for evaluation.
            board_encoding_fn (Callable[[Board], torch.Tensor]): Function to encode the board state.

        Returns:
            tuple[float, int]: The value and player of the node.
        """
        legal_actions = self._board.possible_actions
        self._children = {action: AlphaZeroNode(self._board.move(action), self) for action in legal_actions}

        # Evaluate node
        model.eval()
        with torch.no_grad():
            state = board_encoding_fn(self._board).to("cuda" if torch.cuda.is_available() else "cpu")
            action_logits, value = model(state.unsqueeze(0))
            action_probs = F.softmax(action_logits, dim=0).squeeze(0)

        # If the current node is a terminal state we know the true value (outcome of the game)
        if self.is_terminal():
            value = self._board.evaluate()[1] * self._board.player
            value = torch.tensor(value, dtype=torch.float).unsqueeze(0)

        # Remove invalid actions and renormalize action probabilities
        normalized_action_probs = torch.zeros_like(action_probs)
        normalized_action_probs[legal_actions] = action_probs[legal_actions]
        normalized_action_probs = normalized_action_probs / normalized_action_probs.sum()

        # Create "edges" for all legal actions as explained in the methods sections of
        # Silver et al. (2017) Mastering the game of Go without human knowledge
        self._visits = torch.zeros(len(legal_actions), dtype=torch.int)  # N(s,a)
        self._total_action_values = torch.zeros(len(legal_actions), dtype=torch.float)  # W(s,a)
        self._mean_action_values = torch.zeros(len(legal_actions), dtype=torch.float)  # Q(s,a)
        self._prior_probs = normalized_action_probs[legal_actions]  # P(s,a)

        return value.item(), self._board.player

    def update(self, value: float, action: int, to_play: int) -> None:
        """Updates the node statistics.

        Args:
            value (float): The value to update.
            action (int): The action taken.
            to_play (int): The player to update.
        """
        self._visits[action] += 1
        self._total_action_values[action] += value if self._board.player == to_play else -1 * value
        self._mean_action_values = self._total_action_values / (self._visits + 1e-10)

    def add_noise(self, epsilon: float, alpha: float) -> None:
        """Adds Dirichlet noise to the node.

        Args:
            epsilon (float): The epsilon value for noise.
            alpha (float): The alpha value for noise.
        """
        noise = torch.distributions.Dirichlet(torch.full(self._prior_probs.shape, alpha)).sample()
        self._prior_probs = (1 - epsilon) * self._prior_probs + epsilon * noise
        self._prior_probs = self._prior_probs / self._prior_probs.sum()

    def __str__(self) -> str:
        if self.is_leaf():
            string = f"Player: {self._board.player}\n" f"{self._board}\n" f"Unexpanded node with no children."
        else:
            string = (
                f"Player: {self._board.player}\n"
                f"{self._board}\n"
                f"V(s,a)={self._visits.tolist()}\n"
                f"W(s,a)={[round(w, 4) for w in self._total_action_values.tolist()]}\n"
                f"Q(s,a)={[round(q, 4) for q in self._mean_action_values.tolist()]}\n"
                f"P(s,a)={[round(p, 4) for p in self._prior_probs.tolist()]}"
            )

        return string


class AlphaZeroMCTS:
    """Implements the AlphaZero Monte Carlo Tree Search (MCTS) algorithm."""

    def __init__(
        self,
        model: nn.Module,
        board: Board,
        board_encoding_fn: Callable[[Board], torch.Tensor],
        num_actions: int,
    ) -> None:
        """Initializes the AlphaZeroMCTS.

        Args:
            model (nn.Module): The neural network model.
            board (Board): The initial game board.
            board_encoding_fn (Callable[[Board], torch.Tensor]): Function to encode the board state.
            num_actions (int): Number of possible actions.
        """
        self._model = model
        self._board_encoding_fn = board_encoding_fn
        self._num_actions = num_actions
        self._root = AlphaZeroNode(board, parent=None)

    def run(
        self,
        search_time: int | None,
        num_simulations: int | None,
        exploration_rate: float,
        temperature: float,
        noise: tuple[float, float] | None = None,
    ) -> torch.Tensor:
        """Runs the MCTS algorithm.

        Args:
            search_time (int | None): The time to run the search.
            num_simulations (int | None): The number of simulations to run.
            exploration_rate (float): The exploration rate for the PUCT algorithm.
            temperature (float): The temperature for exploration.
            noise (tuple[float, float] | None): Optional tuple representing epsilon and alpha
                parameters for Dirichlet noise.

        Raises:
            ValueError: If neither search_time nor num_simulations is provided.

        Returns:
            torch.Tensor: Search probabilities for the root node.
        """
        if (search_time is None) and (num_simulations is None):
            raise ValueError(
                "Either 'search_time' or 'num_simulations' must be provided. Please specify at least one of these arguments."
            )

        # Expand the root node immediately
        if self._root.is_leaf():
            self._root.expand(self._model, self._board_encoding_fn)

        # Add Dirichlet noise to the root node
        if noise is not None:
            epsilon, alpha = noise
            self._root.add_noise(epsilon, alpha)

        simulation_cnt = 0
        start_time = time.time()
        while True:
            current_node = self._root
            search_path = []  # [(node, action), ...]

            # 1. Select node with the highest score until leaf nod is reached
            while not current_node.is_leaf():
                scores = current_node.compute_scores(exploration_rate)
                action = int(scores.argmax().item())
                search_path.append((current_node, action))
                current_node = current_node.children[action]

            # 2. Expand and evaluate leaf node
            value, to_play = current_node.expand(self._model, self._board_encoding_fn)

            # 3. Backpropagate value of the leaf node along the search path
            for node, action in search_path:
                node.update(value, action, to_play)

            # Check for both abort condition. If a single one is met, stop the search
            simulation_cnt += 1
            if (search_time is not None) and ((time.time() - start_time) >= search_time):
                break
            if (num_simulations is not None) and (simulation_cnt >= num_simulations):
                break

        # Return search probabilities of the root node
        return self._compute_search_probs(self._root, temperature)

    def select_subtree_from_action(self, action: int) -> AlphaZeroMCTS:
        """Selects a subtree based on an action.

        Args:
            action (int): The action to select the subtree.

        Returns:
            AlphaZeroMCTS: The selected subtree.
        """
        new_mcts = deepcopy(self)
        new_mcts._root = self._root._children[action]

        return new_mcts

    def _compute_search_probs(self, root: AlphaZeroNode, temperature: float) -> torch.Tensor:
        """Computes the search probabilities.

        Args:
            root (AlphaZeroNode): The root node.
            temperature (float): The temperature for exploration.

        Returns:
            torch.Tensor: Search probabilities for the root node.
        """
        if temperature < 0.1:
            # If temperature is smaller than 0.1 we give the most visited action a probability of 1.
            search_probs = torch.zeros_like(root.visits, dtype=torch.float)
            search_probs[torch.argmax(root.visits)] = 1
        else:
            search_probs = torch.pow(root.visits, 1 / temperature) / torch.pow(root.visits.sum(), 1 / temperature)
            search_probs = search_probs / search_probs.sum()

        # "Add" back invalid actions with probability of zero and renormalize to match neural network output
        normalized_search_probs = torch.zeros(self._num_actions)
        normalized_search_probs[root.actions] = search_probs
        normalized_search_probs = normalized_search_probs / normalized_search_probs.sum()

        return normalized_search_probs


class AlphaZeroReplayBuffer(Dataset):
    """A replay buffer for storing game data for AlphaZero."""

    def __init__(self, size: int) -> None:
        """Initializes the replay buffer.

        Args:
            size (int): The maximum size of the buffer.
        """
        super().__init__()
        self._size = size
        self._states = deque(maxlen=size)
        self._search_probs = deque(maxlen=size)
        self._results = deque(maxlen=size)

    def add(self, history: list, result: int) -> None:
        """Adds game data to the buffer.

        Args:
            history (list): The history of states, search probabilities, and players.
            result (int): The result of the game.
        """
        for state, search_probs, player in history:
            self._states.append(state)
            self._search_probs.append(search_probs)
            self._results.append(result * player)

    def __len__(self) -> int:
        return len(self._states)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state = self._states[idx]
        search_probs = self._search_probs[idx]
        result = torch.tensor(self._results[idx], dtype=torch.float)

        return state, search_probs, result.unsqueeze(0)


class AlphaZeroAgent(ABC):
    """Abstract base class for an AlphaZero agent. Each game requires a separate instance of an agent derived from this class."""

    def __init__(self, search_time: int, exploration_rate: float) -> None:
        """Initializes the AlphaZero agent.

        Args:
            search_time (int): The search time for the MCTS.
            exploration_rate (float): The exploration rate for the PUCT algorithm.
        """
        super().__init__()
        self._search_time = search_time
        self._exploration_rate = exploration_rate

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model, self._board = self._initialize_model()
        self._model.to(self._device)
        self._num_actions = len(self._board.possible_actions)

    @abstractmethod
    def _initialize_model(self) -> tuple[nn.Module, Board]:
        """Initializes the model and board.

        Returns:
            tuple[nn.Module, Board]: The model and board.
        """
        pass

    @abstractmethod
    def _encode_board(self, board: Board) -> torch.Tensor:
        """Encodes the board state.

        Args:
            board (Board): The game board.

        Returns:
            torch.Tensor: The encoded board state.
        """
        pass

    def act(self, board: Board, verbose: int = 0) -> int:
        start_runtime = time.time()

        if len(board.possible_actions) == 1:
            action = board.possible_actions[0]
        else:
            mcts = AlphaZeroMCTS(self._model, board, self._encode_board, self._num_actions)
            search_probs = mcts.run(
                search_time=self._search_time,
                num_simulations=None,
                exploration_rate=self._exploration_rate,
                temperature=0.0,
            )
            action = int(torch.multinomial(search_probs, 1).item())

        runtime = time.time() - start_runtime
        if verbose >= 2:
            print(f"# Selected action: {action} ({runtime=:.3f}s)\n")

        return action

    def train(
        self,
        num_iterations: int,
        num_games: int,
        num_simulations: int,
        buffer_size: int,
        temperature: tuple[float, int, float],
        noise: tuple[float, float] | None,
        learning_rate: tuple[float, list[int], float],
        weight_decay: float,
        batch_size: int,
    ) -> None:
        """Trains the agent.

        Args:
            num_iterations (int): Number of training iterations.
            num_games (int): Number of games per iteration.
            num_simulations (int): Number of simulations per game.
            buffer_size (int): Size of the replay buffer.
            temperature (tuple[float, int, float] | None): Tuple representing initial temperature,
                moves before reducing, and final temperature value for exploration.
            noise (tuple[float, float] | None): Optional tuple representing epsilon and alpha
                parameters for Dirichlet noise.
            learning_rate (tuple[float, list[int], float]): Tuple representing initial learning rate,
                milestones for learning rate decay, and decay factor (gamma) for the optimizer.
            weight_decay (float): Weight decay for the optimizer.
            batch_size (int): Batch size for training.
        """
        lr, milestones, gamma = learning_rate
        replay_buffer = AlphaZeroReplayBuffer(buffer_size)
        optimizer = torch.optim.Adam(self._model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = MultiStepLR(optimizer, milestones, gamma)

        for i in range(num_iterations):
            model_state_dict = self._model.state_dict()
            with mp.Pool() as pool:
                results = []
                for _ in range(num_games):
                    results.append(
                        pool.apply_async(self._self_play, (model_state_dict, num_simulations, temperature, noise))
                    )

                for r in results:
                    history, result = r.get()
                    replay_buffer.add(history, result)

            value_loss, policy_loss = self._retrain_model(replay_buffer, optimizer, scheduler, batch_size)

            last_lr = scheduler.get_last_lr()
            scheduler.step()

            num_digits = len(str(abs(num_iterations)))
            print(f"{i+1:{num_digits}}/{num_iterations}: {value_loss=:.4f}, {policy_loss=:.4f}, lr={last_lr[0]}")

    def save_model(self, fname: str) -> None:
        """Saves the model to a file.

        Args:
            fname (str): The filename to save the model.
        """
        torch.save(self._model.state_dict(), fname)

    def load_model(self, fname: str) -> None:
        """Loads the model from a file.

        Args:
            fname (str): The filename to load the model.
        """
        self._model.load_state_dict(torch.load(fname))

    def _self_play(
        self,
        model_state_dict: dict,
        num_simulations: int,
        temperature: tuple[float, int, float] | None,
        noise: tuple[float, float] | None,
    ) -> tuple[list, int]:
        """Performs self-play to generate training data.

        Args:
            model_state_dict (dict): The state dictionary of the model.
            num_simulations (int): Number of simulations per move.
            temperature (tuple[float, int, float] | None): Tuple representing initial temperature,
                moves before reducing, and final temperature value for exploration.
            noise (tuple[float, float] | None): Optional tuple representing epsilon and alpha
                parameters for Dirichlet noise.

        Returns:
            tuple[list, int]: The history of the game and the result.
        """
        model, board = self._initialize_model()
        model.load_state_dict(model_state_dict)
        mcts = AlphaZeroMCTS(model, board, self._encode_board, self._num_actions)

        round = 0
        history = []  # (state, search_probs, current_player)
        while True:
            # Set temperature
            if temperature is None:
                current_temp = 1.0
            else:
                if round < temperature[1]:
                    current_temp = temperature[0]
                else:
                    current_temp = temperature[2]

            # Run a MCTS from the current board.
            search_probs = mcts.run(
                search_time=None,
                num_simulations=num_simulations,
                exploration_rate=self._exploration_rate,
                temperature=current_temp,
                noise=noise,
            )

            # Add the state, search probabilities, and current player to the history.
            state = self._encode_board(board)
            history.append((state, search_probs, board.player))

            # Make a move based on the search probabilities from the MCTS.
            action = int(torch.multinomial(search_probs, 1).item())
            board = board.move(action)
            finished, result = board.evaluate()
            mcts = mcts.select_subtree_from_action(action)

            if finished:
                # Since we don't need search probabilities after the finally state we can append it with fake ones.
                state = self._encode_board(board)
                search_probs = torch.full((self._num_actions,), 1 / self._num_actions, dtype=torch.float)
                history.append((state, search_probs, board.player))
                break

            round += 1

        return history, result

    def _retrain_model(
        self, replay_buffer: AlphaZeroReplayBuffer, optimizer: Optimizer, scheduler: MultiStepLR, batch_size: int
    ) -> tuple[float, float]:
        """Retrains the model using the replay buffer.

        Args:
            replay_buffer (AlphaZeroReplayBuffer): The replay buffer.
            optimizer (Optimizer): The optimizer.
            scheduler (MultiStepLR): The learning rate scheduler.
            batch_size (int): The batch size for training.
        """
        total_value_loss = 0.0
        total_policy_loss = 0.0

        self._model.train()
        loader = DataLoader(replay_buffer, batch_size, shuffle=True)
        for state, search_probs, result in loader:
            state = state.to(self._device)
            search_probs = search_probs.to(self._device)
            result = result.to(self._device)

            action_logits, value = self._model(state)
            value_loss = F.mse_loss(value, result)
            policy_loss = F.cross_entropy(action_logits, search_probs)
            loss = value_loss + policy_loss

            total_value_loss += value_loss.item()
            total_policy_loss += policy_loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        mean_value_loss = value_loss / len(loader)
        mean_policy_loss = float(policy_loss / len(loader))

        return float(mean_value_loss), float(mean_policy_loss)

    def __str__(self) -> str:
        return f"AlphaZero, search_time={self._search_time}s"


class AlphaZeroTicTacToeAgent(AlphaZeroAgent):
    """AlphaZero agent implementation for the Tic-Tac-Toe game."""

    def __init__(self, search_time: int = 1, exploration_rate: float = 1.0) -> None:
        super().__init__(search_time, exploration_rate)

    def _initialize_model(self) -> tuple[nn.Module, Board]:
        model = AlphaZeroResNet(input_dim=(3, 3), input_channels=3, num_filters=32, num_blocks=3, num_actions=9)
        board = TicTacToeBoard(3)

        return model, board

    def _encode_board(self, board: Board) -> torch.Tensor:
        state = torch.from_numpy(board.state).type(torch.float)
        encoded_state = torch.stack(
            [
                (state == 1),
                (state == -1),
                torch.ones_like(state) if board.player == 1 else torch.zeros_like(state),
            ],
            dim=0,
        )

        return encoded_state


class AlphaZeroConnectFourAgent(AlphaZeroAgent):
    """AlphaZero agent implementation for the Connect Four game."""

    def __init__(self, search_time: int = 3, exploration_rate: float = 1.0) -> None:
        super().__init__(search_time, exploration_rate)

    def _initialize_model(self) -> tuple[nn.Module, Board]:
        model = AlphaZeroResNet(input_dim=(6, 7), input_channels=3, num_filters=64, num_blocks=5, num_actions=7)
        board = ConnectFourBoard()

        return model, board

    def _encode_board(self, board: Board) -> torch.Tensor:
        state = torch.from_numpy(board.state).type(torch.float)
        encoded_state = torch.stack(
            [
                (state == 1),
                (state == -1),
                torch.ones_like(state) if board.player == 1 else torch.zeros_like(state),
            ],
            dim=0,
        )

        return encoded_state
