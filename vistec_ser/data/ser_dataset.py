import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset
from torchaudio.compliance import kaldi


class SERDataset(Dataset):
    """Speech Emotion Recognition Dataset
    Computing features on the fly with truncated batch
    """

    def __init__(
            self,
            csv_file: str,
            max_len: int,
            sampling_rate: int = 16000,
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
        assert isinstance(csv_file, str)
        assert isinstance(max_len, int)
        assert isinstance(sampling_rate, int)
        self.audios = pd.read_csv(csv_file)
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
        audio = audio[:self.sampling_rate * self.max_len]

        sample = {'feature': audio, 'emotion': emotion}
        if self.transform:
            sample = self.transform(sample)

        return sample
