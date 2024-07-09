import torch
from torch import nn


class TicTacToeFCNN(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self._input_dim = 19  # AlphaZeroTicTacToe()._encode_board() => tensor of shape (19,)
        self._num_actions = 9

        self._backbone = nn.Sequential(
            nn.Linear(self._input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        self._policy_head = nn.Sequential(
            nn.Linear(32, self._num_actions),
        )

        self._value_head = nn.Sequential(
            nn.Linear(32, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self._backbone(x)
        action_logits = self._policy_head(x)
        value = self._value_head(x)

        return action_logits, value
