from glob import glob
from typing import List
import os

import pandas as pd
import wget


class EmoDB(object):
    def __init__(self,
                 sampling_rate: int = 16000,
                 num_mel_bins: int = 40,
                 frame_length: int = 50,
                 frame_shift: int = 10,
                 max_len: int = 3,
                 center_feats: bool = True,
                 scale_feats: bool = True,
                 emotions: List[str] = None,
                 download_dir: str = None,
                 experiment_dir: str = None):
        if emotions is None:
            emotions = ["neutral", "anger", "happiness", "sadness"]
        # config download dir
        if download_dir is None:
            self.download_root = f"{os.path.expanduser('~')}/vistec-ser_tmpfiles/vistec"
        else:
            self.download_root = f"{download_dir}/vistec-ser_tmpfiles/emodb"
        if not os.path.exists(self.download_root):
            os.makedirs(self.download_root)

        # config experiment dir
        if experiment_dir is None:
            self.experiment_root = f"{os.path.expanduser('~')}/vistec-ser_tmpfiles/exp_emodb"
        else:
            self.experiment_root = f"{experiment_dir}"
        self.experiment_dir = f"{self.experiment_root}"
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)

        self.download_url = "http://www.emodb.bilderbar.info/download/download.zip"
        self.emotion_mappings = {'N': 'neutral', 'W': 'anger', 'F': 'happiness', 'T': 'sadness'}
        self.label_path = f"{self.download_root}/labels.csv"
        self.test_speaker = ["09", "15"]
        self.val_speaker = ["12", "10"]

        self.max_len = max_len
        self.sampling_rate = sampling_rate
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        self.num_mel_bins = num_mel_bins
        self.center_feats = center_feats
        self.scale_feats = scale_feats
        self.sec_to_frame = 10 * self.frame_shift

        self.emotions = emotions
        self.n_classes = len(self.emotions)

    def download(self):
        # download
        if not os.path.exists(f"{self.download_root}/download.zip"):
            print(">downloading dataset...")
            wget.download(url=self.download_url, out=f"{self.download_root}/download.zip", bar=wget.bar_adaptive)
        # unzip
        if not os.path.exists(f"{self.download_root}/emo-db"):
            print(">unzipping data...")
            os.system(f"unzip -q {self.download_root}/download.zip -d {self.download_root}/emo-db")
        if not os.path.exists(f"{self.label_path}"):
            print(">preparing labels...")
            labels = ["PATH, EMOTION\n"]
            for wav in glob(f"{self.download_root}/emo-db/*/*.wav"):
                key = wav.split('.')[0][-2]
                if key not in self.emotion_mappings.keys():
                    continue
                emotion = self.emotion_mappings[key]
                wav = os.path.abspath(wav)
                labels.append(f"{wav},{emotion}\n")
            open(self.label_path, "w").writelines(labels)

    def prepare_labels(self):
        self.download()
        assert os.path.exists(self.label_path)
        labels = pd.read_csv(self.label_path)

