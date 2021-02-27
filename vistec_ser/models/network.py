from collections import OrderedDict

import torch.nn as nn

from .base_model import BaseSliceModel, BaseModel
from .layers.rnn import AttentionLSTM


class CNN1DLSTM(BaseModel):
    def __init__(self, hparams):
        super().__init__(hparams)
        in_channel = hparams.get("in_channel", 40)
        sequence_length = hparams.get("sequence_length", 300)
        n_channels = hparams.get("n_channels", [64, 64, 128, 128])
        kernel_size = hparams.get("kernel_size", [5, 3, 3, 3])
        pool_size = hparams.get("pool_size", [4, 2, 2, 2])
        lstm_unit = hparams.get("lstm_unit", 128)
        n_classes = hparams.get("n_classes", 4)

        self.in_channel = in_channel
        self.sequence_length = sequence_length
        assert len(n_channels) == len(kernel_size) == len(pool_size), "Size of `n_channels`, `kernel_size`, and " \
                                                                      "`pool_size` must equal "

        # configure cnn parameters
        in_channels = [in_channel] + n_channels[:-1]
        out_channels = n_channels
        seq_lens = []
        for p in pool_size:
            seq_lens.append(sequence_length)
            sequence_length = sequence_length // p
        assert len(in_channels) == len(out_channels) == len(seq_lens)

        self.cnn_layers = nn.Sequential(OrderedDict([
            (f"conv{i}", nn.Sequential(
                nn.Conv1d(in_channels=ic, out_channels=oc, kernel_size=k, padding=(k - 1) // 2),
                nn.LeakyReLU(),
                nn.LayerNorm(normalized_shape=[oc, seq]),
                nn.MaxPool1d(kernel_size=p)
            )) for i, (ic, oc, k, seq, p) in enumerate(zip(in_channels, out_channels, kernel_size, seq_lens, pool_size))
        ]))
        self.lstm = nn.LSTM(input_size=out_channels[-1], hidden_size=lstm_unit, bidirectional=True, batch_first=True)
        self.logits = nn.Linear(lstm_unit * 2, n_classes)

    def forward(self, x):
        for cnn in self.cnn_layers:
            x = cnn(x)
        x = x.transpose(1, 2)
        _, (x, _) = self.lstm(x)
        x = x.transpose(0, 1)
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])
        x = self.logits(x)
        return x


class CNN1DLSTMSlice(BaseSliceModel):
    def __init__(self, hparams):
        super().__init__(hparams)
        in_channel = hparams.get("in_channel", 40)
        sequence_length = hparams.get("sequence_length", 300)
        n_channels = hparams.get("n_channels", [64, 64, 128, 128])
        kernel_size = hparams.get("kernel_size", [5, 3, 3, 3])
        pool_size = hparams.get("pool_size", [4, 2, 2, 2])
        lstm_unit = hparams.get("lstm_unit", 128)
        n_classes = hparams.get("n_classes", 4)

        self.in_channel = in_channel
        self.sequence_length = sequence_length
        assert len(n_channels) == len(kernel_size) == len(pool_size), "Size of `n_channels`, `kernel_size`, and " \
                                                                      "`pool_size` must equal "

        # configure cnn parameters
        in_channels = [in_channel] + n_channels[:-1]
        out_channels = n_channels
        seq_lens = []
        for p in pool_size:
            seq_lens.append(sequence_length)
            sequence_length = sequence_length // p
        assert len(in_channels) == len(out_channels) == len(seq_lens)

        self.cnn_layers = nn.Sequential(OrderedDict([
            (f"conv{i}", nn.Sequential(
                nn.Conv1d(in_channels=ic, out_channels=oc, kernel_size=k, padding=(k - 1) // 2),
                nn.LeakyReLU(),
                nn.LayerNorm(normalized_shape=[oc, seq]),
                nn.MaxPool1d(kernel_size=p)
            )) for i, (ic, oc, k, seq, p) in enumerate(zip(in_channels, out_channels, kernel_size, seq_lens, pool_size))
        ]))
        self.lstm = nn.LSTM(input_size=out_channels[-1], hidden_size=lstm_unit, bidirectional=True, batch_first=True)
        self.logits = nn.Linear(lstm_unit * 2, n_classes)

    def forward(self, x):
        for cnn in self.cnn_layers:
            x = cnn(x)
        x = x.transpose(1, 2)
        _, (x, _) = self.lstm(x)
        x = x.transpose(0, 1)
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])
        x = self.logits(x)
        return x


class CNN1DLSTMAttentionSlice(BaseSliceModel):
    def __init__(self, hparams):
        super().__init__(hparams)
        in_channel = hparams.get("in_channel", 40)
        sequence_length = hparams.get("sequence_length", 300)
        n_channels = hparams.get("n_channels", [64, 64, 128, 128])
        kernel_size = hparams.get("kernel_size", [5, 3, 3, 3])
        pool_size = hparams.get("pool_size", [4, 2, 2, 2])
        lstm_unit = hparams.get("lstm_unit", 128)
        n_classes = hparams.get("n_classes", 4)

        self.in_channel = in_channel
        self.sequence_length = sequence_length
        assert len(n_channels) == len(kernel_size) == len(pool_size), "Size of `n_channels`, `kernel_size`, and " \
                                                                      "`pool_size` must equal "

        # configure cnn parameters
        in_channels = [in_channel] + n_channels[:-1]
        out_channels = n_channels
        seq_lens = []
        for p in pool_size:
            seq_lens.append(sequence_length)
            sequence_length = sequence_length // p
        assert len(in_channels) == len(out_channels) == len(seq_lens)

        self.cnn_layers = nn.Sequential(OrderedDict([
            (f"conv{i}", nn.Sequential(
                nn.Conv1d(in_channels=ic, out_channels=oc, kernel_size=k, padding=(k - 1) // 2),
                nn.LeakyReLU(),
                nn.LayerNorm(normalized_shape=[oc, seq]),
                nn.MaxPool1d(kernel_size=p)
            )) for i, (ic, oc, k, seq, p) in enumerate(zip(in_channels, out_channels, kernel_size, seq_lens, pool_size))
        ]))
        self.lstm = AttentionLSTM(
            input_dim=out_channels[-1], hidden_dim=lstm_unit, bidirectional=True, output_dim=n_classes)

    def forward(self, x):
        for cnn in self.cnn_layers:
            x = cnn(x)
        x = x.transpose(1, 2)
        x = self.lstm(x)
        return x
