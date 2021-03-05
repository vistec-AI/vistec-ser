from typing import Union

import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset
from torchaudio.compliance import kaldi

from .features.padding import pad_dup


class SERDataset(Dataset):
    """Speech Emotion Recognition Dataset
    Computing features on the fly with truncated batch
    """

    def __init__(
            self,
            csv_file: Union[str, pd.DataFrame],
            max_len: int,
            sampling_rate: int = 16000,
            emotions=None,
            transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            max_len (int): Maximum length (in second) of audio file. Others will be truncated
            sampling_rate (int): target sampling rate of audio. If this isn't match with file's sampling rate,
                result will be resampled into target sampling rate
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        assert isinstance(max_len, int)
        assert isinstance(sampling_rate, int)
        if emotions is None:
            self.emotions = ["neutral", "anger", "happiness", "sadness"]
        else:
            self.emotions = emotions
        if isinstance(csv_file, str):
            self.audios = pd.read_csv(csv_file)
        else:
            self.audios = csv_file
        self.audios = self.audios[self.audios["EMOTION"].str.lower().isin(self.emotions)]
        self.n_classes = len(self.emotions)
        self.sampling_rate = sampling_rate
        self.max_len = max_len
        self.transform = transform

    def __len__(self):
        return len(self.audios)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # df contains 2 columns, [PATH, EMOTION]
        audio_path = self.audios.iloc[idx, 0]
        emotion = self.audios.iloc[idx, 1]
        audio, sample_rate = torchaudio.backend.sox_backend.load(audio_path)

        # initial preprocess
        # convert to mono, resample, truncate
        audio = torch.unsqueeze(audio.mean(dim=0), dim=0)  # convert to mono
        if sample_rate != self.sampling_rate:
            audio = kaldi.resample_waveform(audio, orig_freq=sample_rate, new_freq=self.sampling_rate)

        # emotion to label
        emotion = self.emotions.index(emotion.lower().strip())

        # extract features
        sample = {'feature': audio, 'emotion': emotion}
        if self.transform:
            sample = self.transform(sample)
        feature = sample["feature"]

        # truncated
        sequence_length = self.max_len * 100  # 100 frame = 1 sec
        if feature.shape[-1] > sequence_length:
            feature = feature[:, :sequence_length]
        else:
            feature = pad_dup(feature, sequence_length)
        sample = {"feature": feature, "emotion": emotion}

        return sample
