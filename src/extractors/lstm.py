import torch as th
import torch.nn as nn
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class LSTMExtractor(BaseFeaturesExtractor):
    """
    LSTM features extractor for Stable-Baselines3.

    Args:
        observation_space: gymnasium Box describing the observation (shape=(input_dim,))
        hidden_dim: LSTM hidden size
        layer_dim: number of LSTM layers
        features_dim: number of output features (if None, defaults to hidden_dim)

    The forward returns a tensor of shape (batch_size, features_dim).
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        hidden_dim,
        layer_dim,
    ):
        input_dim = int(observation_space.shape[0])
        self.hidden_dim = int(hidden_dim)
        self.layer_dim = int(layer_dim)

        super().__init__(observation_space, 1)

        self.lstm = nn.LSTM(input_dim, self.hidden_dim, self.layer_dim, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, 1)

    def forward(self, x, h0=None, c0=None):
        assert isinstance(x, th.Tensor), f"Expected torch.Tensor, got {type(x)}"
        if x.dim() == 2:
            x = x.unsqueeze(1)

        batch_size = x.size(0)
        if h0 is None or c0 is None:
            h0 = th.zeros(self.layer_dim, batch_size, self.hidden_dim, device=x.device, dtype=x.dtype)
            c0 = th.zeros(self.layer_dim, batch_size, self.hidden_dim, device=x.device, dtype=x.dtype)

        out, (hn, cn) = self.lstm(x, (h0, c0))

        last = out[:, -1, :]
        features = self.fc(last)
        return features


policy_kwargs = dict(
    features_extractor_class=LSTMExtractor,
    features_extractor_kwargs=dict(features_dim=128),
)