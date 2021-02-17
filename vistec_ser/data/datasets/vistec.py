from glob import glob
from typing import Dict, List, Union
import json
import os

import gdown
import numpy as np
import pandas as pd

emo2idx = {emo: i for i, emo in enumerate(['Neutral', 'Angry', 'Happy', 'Sad', 'Frustrated'])}
idx2emo = {v: k for k, v in emo2idx.items()}
correctemo = {
    'Neutral': 'Neutral',
    'Angry': 'Anger',
    'Happy': 'Happiness',
    'Sad': 'Sadness',
    'Frustrated': 'Frustration'
}


class VISTEC(object):
    def __init__(
            self,
            agreement_threshold: float = 0.7,
            mic_type: str = 'con',
            download_dir: str = None):
        self.agreement_threshold = agreement_threshold
        self.mic_type = mic_type
        if download_dir is None:
            self.download_root = f"{os.path.expanduser('~')}/vistec-ser_tmpfiles/vistec"
        else:
            self.download_root = f"{download_dir}/vistec-ser_tmpfiles/vistec"
        if not os.path.exists(self.download_root):
            os.makedirs(self.download_root)
        self.download_ids = {
            "studio1-10": "1M69xuXhPE6YRFWatm0D4MiDi1blLNy0P",
            "studio11-20": "1MqEestPscu2ao_jKUdM9HLva4DZFxCXe",
            "studio21-30": "1lHhMEDs4YhnsGdKYBKbvidhFert_XF74",
            "studio31-40": "1-AOy30Lm0yEnK_Q44QrSsgQN-XBKmfoW",
            "studio41-50": "16iRYWn614AQjZoWlW9-Vc9f6TW_1Z4Ii",
            "studio51-60": "1YX3Xus9hJEfbhww1mHOG_osLJho9yFBf",
            "zoom1-10": "1-2QGXwfsDFfEqDl4KQ5jLPtDmbzuSc7z",
            "zoom11-20": "17DXFur1ZAA7IAkX4-xa0OHyDTRa_KmZP",
            "labels": "1Ym3Go5mN_5jCmvV7H3bpNnctk_tRqhNb",
        }
        self.github_url = {
            k: f"https://github.com/vistec-AI/dataset-releases/releases/download/v0.1/{k}.zip"
            for k in self.download_ids.keys()
            if k != "labels"
        }

    def get_audio_path(self, audio_name: str) -> str:
        if not isinstance(audio_name, str):
            raise TypeError(f"audio name must be string but got {type(audio_name)}")
        studio_type = audio_name[0]
        studio_num = audio_name.split('_')[0][1:]
        if studio_type == "s":
            directory = f"studio{studio_num}"
        elif studio_type == "z":
            directory = f"zoom{studio_num}"
        else:
            raise NameError(f"Error reading file name {audio_name}")
        audio_path = f"{self.download_root}/{directory}/con/{audio_name}".replace(".wav", ".flac")
        if studio_type == "s":
            audio_path = audio_path.replace("con", self.mic_type)
        elif studio_type == "z":
            audio_path = audio_path.replace("con", "mic")
        else:
            raise NameError(f"Error reading file name {audio_name}")
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"{audio_path} not found")
        return audio_path

    def get_labels(self):
        # format
        print("+-----------------------------------+")
        print("| Formatting labels...              |")
        print("+-----------------------------------+\n")
        labels = []
        for json_path in sorted(glob(f"{self.download_root}/*.json")):
            print(f">formatting {json_path} ...")
            data = read_json(json_path)
            agreements = get_agreements(data)
            json_df = pd.DataFrame([
                (f"{self.get_audio_path(k)}", correctemo[idx2emo[v]])
                for k, v in {k: convert_to_hardlabel(v, thresh=self.agreement_threshold)
                             for k, v in agreements.items()}.items()
                if v != -1
            ], columns=['PATH', 'EMOTION'])
            labels.append(json_df)
        return pd.concat(labels)

    def download(self):
        # download
        print("+-----------------------------------+")
        print("| Downloading dataset...            |")
        print("+-----------------------------------+\n")
        for f, gid in self.download_ids.items():
            if not os.path.exists(f"{self.download_root}/{f}.zip"):
                print(f">downloading {f}.zip ...")
                out_name = os.path.join(self.download_root, f"{f}.zip")
                try:
                    gdown.download(f"https://drive.google.com/uc?id={gid}", output=out_name, quiet=False)
                except ConnectionError:
                    if f in self.github_url.keys():
                        os.system(f"wget {self.github_url[f]} -O {out_name}")
            else:
                print(f"{f}.zip existed, skipping...")
        print("Finished Downloading Dataset\n")

        # extract
        print("+-----------------------------------+")
        print("| Extracting dataset...             |")
        print("+-----------------------------------+\n")
        for f in glob(f"{self.download_root}/*.zip"):
            print(f">unzipping {f}...")
            assert os.path.exists(f)
            os.system(f"unzip -q {f} -d {self.download_root}")


# read JSON
def read_json(json_path: str) -> Dict[str, dict]:
    """Read label JSON"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    assert len(set([len(x) for x in data.values()])) == 1, 'Length of the json object values not equal to 1'
    return {k: v[0] for k, v in data.items()}


# labels manipulation
def convert_to_softlabel(evals: List[str]) -> List[float]:
    """Converts a list of emotion into distribution"""
    softlabel = [0 for _ in range(len(emo2idx.keys()))]
    for emo in evals:
        if emo.lower() in ['none', 'other']:
            continue
        softlabel[emo2idx[emo]] += 1
    if np.sum(softlabel) != 0:
        softlabel = np.array(softlabel) / np.sum(softlabel)
    return softlabel


def get_score_from_emo_list(emo_list: List[List[str]]) -> np.ndarray:
    """Aggregate a list of evaluations (which are a list of emotions) into distribution"""
    return np.mean([convert_to_softlabel(evals) for evals in emo_list], axis=0)


def get_agreements(data: Dict[str, dict]) -> Dict[str, np.array]:
    """Get agreement distribution from provided label"""
    softlabel = {k: get_score_from_emo_list(v['annotated']) for k, v in data.items()}
    return softlabel


def convert_to_hardlabel(agreement_dist: List[float], thresh: float = 0.7) -> Union[np.ndarray, int]:
    """convert list"""
    if max(agreement_dist) < thresh:
        return -1
    if pd.Series(agreement_dist).value_counts().sort_index(ascending=False).iloc[0] > 1:
        return -1
    else:
        return np.argmax(agreement_dist)
