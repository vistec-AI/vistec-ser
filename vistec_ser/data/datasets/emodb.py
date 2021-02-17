from glob import glob
import os

import pandas as pd


class EmoDB(object):
    def __init__(self, agreement_threshold: float = 0.7):
        self.agreement_threshold = agreement_threshold
        self.download_root = f"{os.path.expanduser('~')}/vistec-ser_tmpfiles/emodb"
        if not os.path.exists(self.download_root):
            os.system(f"mkdir -p {self.download_root}")
        self.download_url = "http://www.emodb.bilderbar.info/download/download.zip"
        self.emotion_mappings = {'N': 'neutral', 'W': 'anger', 'F':'happiness', 'T': 'sadness'}
        self.label_path = f"{self.download_root}/labels.csv"

    def download(self):
        # download
        os.system(f"wget {self.download_url} -q --show-progress -O {self.download_root}/download.zip")
        # unzip
        os.system(f"unzip -q {self.download_root}/download.zip -d {self.download_root}/emo-db")
        labels = ["PATH, EMOTION\n"]
        for wav in glob(f"{self.download_root}/emo-db/*/*.wav"):
            key = wav.split('.')[0][-2]
            if key not in self.emotion_mappings.keys():
                continue
            emotion = self.emotion_mappings[key]
            wav = os.path.abspath(wav)
            labels.append(f"{wav},{emotion}\n")
        open(self.label_path, "w").writelines(labels)

    def get_labels(self):
        if not os.path.exists(self.label_path):
            self.download()
            assert os.path.exists(self.label_path)
        return pd.read_csv(self.label_path)
