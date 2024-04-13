from pathlib import Path
from typing import Optional, Protocol, Union

import torch
import torch.nn as nn


class SerializableModule(Protocol):

    def save(self, fpath: Union[str, Path]) -> None:
        ...

    @classmethod
    def from_pretrained(
            cls, fpath: Union[str, Path],
            map_location: Optional[torch.device]) -> "SerializableModule":
        ...


class Generator(nn.Module):
    """
    Generator with input shape: (batch_size, window_size, latent_space_dim)
    output shape: (batch_size, window_size, output_dim)
    The output_dim should be the same as the input_dim of the Discriminator, aka the number of features.
    """

    def __init__(self,
                 window_size: int,
                 latent_space_dim: int,
                 hidden_units: int,
                 output_dim: int,
                 n_lstm_layers: int = 2) -> None:
        super().__init__()
        self.window_size = window_size
        self.latent_space_dim = latent_space_dim
        self.hidden_units = hidden_units
        self.n_lstm_layers = n_lstm_layers
        self.output_dim = output_dim

        self.lstm = nn.LSTM(input_size=self.latent_space_dim,
                            hidden_size=self.hidden_units,
                            num_layers=self.n_lstm_layers,
                            batch_first=True,
                            dropout=.1)
        # Add batch normalization layer
        self.batch_norm = nn.BatchNorm1d(self.hidden_units)

        self.linear = nn.Linear(in_features=self.hidden_units,
                                out_features=self.output_dim)
        # Initialize weights
        nn.init.trunc_normal_(self.linear.bias)
        nn.init.trunc_normal_(self.linear.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rnn_output, _ = self.lstm(x)
        rnn_output_2d = rnn_output.reshape(-1, self.hidden_units)
        rnn_output_2d = self.batch_norm(
            rnn_output_2d)  # Apply batch normalization
        output_2d = torch.tanh(self.linear(rnn_output_2d))
        output_3d = output_2d.view(-1, self.window_size, self.output_dim)
        return output_3d

    def save(self, fpath: Union[Path, str]) -> None:
        chkp = {
            "config": {
                "window_size": self.window_size,
                "latent_space_dim": self.latent_space_dim,
                "hidden_units": self.hidden_units,
                "n_lstm_layers": self.n_lstm_layers,
                "output_dim": self.output_dim
            },
            "weights": self.state_dict(),
        }
        torch.save(chkp, fpath)

    @classmethod
    def from_pretrained(
            cls,
            fpath: Union[Path, str],
            map_location: Optional[torch.device] = None) -> "Generator":
        chkp = torch.load(fpath, map_location=map_location)
        model = cls(**chkp.pop("config"))
        model.load_state_dict(chkp.pop("weights"))
        model.eval()
        return model


class Discriminator(nn.Module):
    """
    Discriminator with input shape: (batch_size, window_size, input_dim)
    output shape: (batch_size, window_size, 1)
    The input_dim should be the same as the output_dim of the Generator, aka the number of features.
    """

    def __init__(self,
                 input_dim: int,
                 hidden_units: int,
                 n_lstm_layers: int = 2,
                 add_batch_mean: bool = False) -> None:
        super().__init__()
        self.add_batch_mean = add_batch_mean
        self.hidden_units = hidden_units
        self.input_dim = input_dim
        self.n_lstm_layers = n_lstm_layers

        extra_features = self.hidden_units if self.add_batch_mean else 0
        drop_out = 0.1 if n_lstm_layers > 1 else 0
        self.lstm = nn.LSTM(input_size=self.input_dim,
                            hidden_size=self.hidden_units + extra_features,
                            num_layers=self.n_lstm_layers,
                            batch_first=True,
                            dropout=drop_out)

        self.linear = nn.Linear(in_features=self.hidden_units + extra_features,
                                out_features=1)
        nn.init.trunc_normal_(self.linear.bias)
        nn.init.trunc_normal_(self.linear.weight)

        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.add_batch_mean:
            bs = x.size(0)
            batch_mean = x.mean(0, keepdim=True).repeat(bs, 1, 1)
            x = torch.cat([x, batch_mean], dim=-1)

        rnn_output, _ = self.lstm(x)
        logits = self.linear(rnn_output)
        output = self.activation(logits)
        return output, logits

    def save(self, fpath: Union[Path, str]) -> None:
        chkp = {
            "config": {
                "add_batch_mean": self.add_batch_mean,
                "hidden_units": self.hidden_units,
                "input_dim": self.input_dim,
                "n_lstm_layers": self.n_lstm_layers
            },
            "weights": self.state_dict(),
        }
        torch.save(chkp, fpath)

    @classmethod
    def from_pretrained(
            cls,
            fpath: Union[Path, str],
            map_location: Optional[torch.device] = None) -> "Discriminator":
        chkp = torch.load(fpath, map_location=map_location)
        model = cls(**chkp.pop("config"))
        model.load_state_dict(chkp.pop("weights"))
        model.eval()
        return model
