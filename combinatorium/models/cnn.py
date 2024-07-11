import torch
from torch import nn


class AlphaZeroResNet(nn.Module):

    def __init__(
        self, input_dim: tuple[int, int], input_channels: int, num_filters: int, num_blocks: int, num_actions: int
    ) -> None:
        super().__init__()

        self._conv_block = nn.Sequential(
            nn.Conv2d(input_channels, num_filters, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(inplace=True),
        )

        self._residual_tower = nn.Sequential(*[ResidualBlock(num_filters) for _ in range(num_blocks)])

        self._policy_head = nn.Sequential(
            nn.Conv2d(num_filters, 2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(input_dim[0] * input_dim[1] * 2, num_actions),
        )

        self._value_head = nn.Sequential(
            nn.Conv2d(num_filters, 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(input_dim[0] * input_dim[1] * 1, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self._conv_block(x)
        x = self._residual_tower(x)

        action_logits = self._policy_head(x)
        value = self._value_head(x)

        return action_logits, value


class ResidualBlock(nn.Module):

    def __init__(self, num_filters: int) -> None:
        super().__init__()

        self._conv_block = nn.Sequential(
            nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_filters),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self._conv_block(x)
        out += identity  # skip connection
        out = nn.functional.relu(out)

        return out
