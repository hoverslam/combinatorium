from __future__ import annotations

from combinatorium.interfaces import Board
from combinatorium.models import TicTacToeFCNN
from combinatorium.games.tic_tac_toe.board import TicTacToeBoard

from typing import Callable
from collections import deque
from abc import ABC, abstractmethod
from copy import deepcopy
import multiprocessing as mp

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader


class AlphaZeroNode:

    def __init__(self, board: Board, parent: AlphaZeroNode | None) -> None:
        self._board = board
        self._parent = parent
        self._children = {}  # dictionary with <action, child> as key-value pairs

    @property
    def children(self) -> list:
        if self.is_leaf():
            raise ValueError("This unexpanded node has no child nodes.")

        return list(self._children.values())

    @property
    def actions(self) -> list:
        if self.is_leaf():
            raise ValueError("This unexpanded node has no actions because it has no child nodes.")

        return list(self._children.keys())

    @property
    def visits(self) -> torch.Tensor:
        if self.is_leaf():
            raise ValueError("This unexpanded node has no visit statistics because it has no child nodes.")

        return self._visits

    def compute_scores(self, exploration_rate: float) -> torch.Tensor:
        if self.is_leaf():
            raise ValueError("This unexpanded node cannot compute scores because it has no child nodes.")

        # The PUCT algorithm from the paper with a small amount added to the total visit count. This has the effect that
        # with 0 visits the score equals the prior probabilities
        puct_scores = (
            exploration_rate * self._prior_probs * ((torch.sqrt(self._visits.sum()) + 1e-10) / (1 + self._visits))
        )

        return self._mean_action_values + puct_scores

    def is_leaf(self) -> bool:
        return len(self._children) == 0

    def expand(self, model: nn.Module, board_encoding_fn: Callable[[Board], torch.Tensor]) -> tuple[float, int]:
        legal_actions = self._board.possible_actions
        self._children = {action: AlphaZeroNode(self._board.move(action), self) for action in legal_actions}

        # Evaluate node
        model.eval()
        with torch.no_grad():
            device = "cuda" if torch.cuda.is_available() else "cpu"
            state = board_encoding_fn(self._board).to(device)
            action_logits, value = model(state)
            action_probs = F.softmax(action_logits, dim=0)

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
        self._visits[action] += 1
        self._total_action_values[action] += value if self._board.player == to_play else -1 * value
        self._mean_action_values = self._total_action_values / (self._visits + 1e-10)

    def add_noise(self, epsilon: float, alpha: float) -> None:
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

    def __init__(
        self,
        model: nn.Module,
        board: Board,
        board_encoding_fn: Callable[[Board], torch.Tensor],
        num_actions: int,
    ) -> None:
        self._model = model
        self._board_encoding_fn = board_encoding_fn
        self._num_actions = num_actions
        self._root = AlphaZeroNode(board, parent=None)

    def run(
        self,
        num_simulations: int,
        exploration_rate: float,
        temperature: float,
        noise: tuple[float, float] | None = None,
    ) -> torch.Tensor:
        # Expand the root node immediately
        if self._root.is_leaf():
            self._root.expand(self._model, self._board_encoding_fn)

        # Add Dirichlet noise to the root node
        if noise is not None:
            epsilon, alpha = noise
            self._root.add_noise(epsilon, alpha)

        for _ in range(num_simulations):
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

        # Return search probabilities of the root node
        return self._compute_search_probs(self._root, temperature)

    def select_subtree_from_action(self, action: int) -> AlphaZeroMCTS:
        new_mcts = deepcopy(self)
        new_mcts._root = self._root._children[action]

        return new_mcts

    def _compute_search_probs(self, root: AlphaZeroNode, temperature: float) -> torch.Tensor:
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

    def __init__(self, size: int) -> None:
        super().__init__()
        self._size = size
        self._states = deque(maxlen=size)
        self._search_probs = deque(maxlen=size)
        self._results = deque(maxlen=size)

    def add(self, history: list, result: int) -> None:
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


class AlphaZero(ABC):

    def __init__(self, num_simulations: int, exploration_rate: float) -> None:
        super().__init__()
        self._num_simulations = num_simulations
        self._exploration_rate = exploration_rate  # c_puct

        self._device = self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model, self._num_actions = self._initialize_model()
        self._model.to(self._device)

    @abstractmethod
    def _initialize_model(self) -> tuple[nn.Module, int]:
        pass

    @abstractmethod
    def _encode_board(self, board: Board) -> torch.Tensor:
        pass

    def act(self, board: Board) -> int:
        mcts = AlphaZeroMCTS(self._model, board, self._encode_board, self._num_actions)
        search_probs = mcts.run(self._num_simulations, self._exploration_rate, temperature=0.0)
        action = torch.multinomial(search_probs, 1)

        return int(action.item())

    def train(
        self,
        num_iterations: int,
        num_games: int,
        buffer_size: int,
        temperature: float,
        noise: tuple[float, float] | None,
        learning_rate: float,
        weight_decay: float,
        batch_size: int,
    ) -> None:
        replay_buffer = AlphaZeroReplayBuffer(buffer_size)
        optimizer = torch.optim.Adam(self._model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        num_digits = len(str(abs(num_iterations)))

        for i in range(num_iterations):
            results = []
            with mp.Pool() as pool:
                for _ in range(num_games):
                    results.append(pool.apply_async(self._self_play, (temperature, noise)))

                for r in results:
                    history, result = r.get()
                    replay_buffer.add(history, result)

            value_loss, policy_loss = self._retrain_model(replay_buffer, optimizer, batch_size)
            print(f"{i+1:{num_digits}}/{num_iterations}: {value_loss=:.6f}, {policy_loss=:.6f}")

    def save_model(self, fname: str) -> None:
        torch.save(self._model.state_dict(), fname)

    def load_model(self, fname: str) -> None:
        self._model.load_state_dict(torch.load(fname))

    def _self_play(
        self,
        temperature: float,
        noise: tuple[float, float] | None,
    ) -> tuple[list, int]:
        board = TicTacToeBoard(3)
        mcts = AlphaZeroMCTS(self._model, board, self._encode_board, self._num_actions)
        finished, result = board.evaluate()
        history = []  # (state, search_probs, current_player)

        while True:
            # Run a MCTS from the current board.
            search_probs = mcts.run(
                self._num_simulations,
                self._exploration_rate,
                temperature,
                noise,
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

        return history, result

    def _retrain_model(
        self, replay_buffer: AlphaZeroReplayBuffer, optimizer: Optimizer, batch_size: int
    ) -> tuple[float, float]:
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


class AlphaZeroTicTacToe(AlphaZero):

    def __init__(self, num_simulations: int = 100, exploration_rate: float = 1.0) -> None:
        super().__init__(num_simulations, exploration_rate)

    def _initialize_model(self) -> tuple[nn.Module, int]:
        return TicTacToeFCNN(), 9

    def _encode_board(self, board: Board) -> torch.Tensor:
        state = torch.from_numpy(board.state).type(torch.float)
        encoded_state = torch.concat(
            [
                (state == 1).ravel(),
                (state == -1).ravel(),
                torch.ones(1) if board.player == 1 else torch.zeros(1),
            ]
        )

        return encoded_state
