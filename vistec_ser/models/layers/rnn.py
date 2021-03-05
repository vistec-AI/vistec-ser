import torch
import torch.nn as nn
from torch.nn import functional as F


class PyramidLSTM(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            bidirectional: bool = True,
            dropout: float = 0.):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=input_dim * 2,
            hidden_size=hidden_dim,
            bidirectional=bidirectional,
            dropout=dropout,
            batch_first=True
        )

    def forward(self, x: torch.Tensor):
        batch_size, time_dim, feat_dim = x.shape
        x = x.contiguous().view(batch_size, int(time_dim / 2), feat_dim * 2)  # stack features
        x, (h, c) = self.rnn(x)
        return x, (h, c)


class AttentionLSTM(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            bidirectional: bool = True,
            dropout: float = 0.):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            bidirectional=bidirectional,
            dropout=dropout,
            batch_first=True
        )
        if bidirectional:
            hidden_dim = hidden_dim * 2
        self.logits = nn.Linear(hidden_dim, output_dim)

    def dot_attention(self, keys: torch.Tensor, query: torch.Tensor):
        """
        alpha = softmax(qK) * V
        where
            - q (query) is LSTM last hidden states
            - K (keys) = V (values) are LSTM sequence
        """
        batch_size = query.shape[1]
        query = query.reshape(batch_size, -1)
        attn_weights = torch.bmm(keys, query.unsqueeze(2)).squeeze(2)
        attn_weights = F.softmax(attn_weights, dim=1)
        output = torch.bmm(keys.transpose(1, 2), attn_weights.unsqueeze(2)).squeeze(2)
        return output

    def forward(self, x):
        x, (h, c) = self.rnn(x)
        x = self.dot_attention(x, h)
        x = self.logits(x)
        return x
