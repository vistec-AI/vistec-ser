from glob import glob
from typing import Dict, List, Union
import json
import os
import subprocess

import numpy as np
import pandas as pd

import vistec_ser


emo2idx = {emo: i for i, emo in enumerate(['Neutral', 'Angry', 'Happy', 'Sad', 'Frustrated'])}
idx2emo = {v: k for k, v in emo2idx.items()}
correctemo = {
    'Neutral': 'Neutral',
    'Angry': 'Anger',
    'Happy': 'Happiness',
    'Sad': 'Sadness',
    'Frustrated': 'Frustration'
}


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
        if emo.lower() in ['none', 'other']: continue
        softlabel[emo2idx[emo]] += 1
    if np.sum(softlabel) != 0: softlabel = np.array(softlabel) / np.sum(softlabel)
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
    if max(agreement_dist) < thresh: return -1
    if pd.Series(agreement_dist).value_counts().sort_index(ascending=False).iloc[0] > 1:
        return -1
    else:
        return np.argmax(agreement_dist)


def get_labels(agreements: Dict[str, np.array], thresh: float = 0.7) -> pd.DataFrame:
    return pd.DataFrame([(k.replace('.wav', ''), correctemo[idx2emo[v]]) for k, v in
                         {k: convert_to_hardlabel(v, thresh=thresh) for k, v in agreements.items()}.items() if v != -1],
                        columns=['audio_name', 'emotion'])


def generate_csv(json_path, csv_path, agreement_threshold):
    data = read_json(json_path)
    agreements = get_agreements(data)
    labels = get_labels(agreements, thresh=agreement_threshold)
    if os.path.exists(csv_path):
        print('WARNING: CSV files already exists. Replacing...')
    labels.to_csv(csv_path, index=False)


class VISTEC:
    def __init__(self, agreement_threshold: float = 0.7):
        self.download_script = "download_vistec.sh"
        self.agreement_threshold = agreement_threshold

    def format_label(self):
        print("+-----------------------------------+")
        print("| Formatting labels...              |")
        print("+-----------------------------------+\n")
        for json_path in glob("labels/*.json"):
            print(f">formatting {json_path} ...")
            csv_path = os.path.basename(json_path).replace('json', 'csv')
            generate_csv(
                json_path=json_path,
                csv_path=csv_path,
                agreement_threshold=self.agreement_threshold)
        os.system("cat *.csv | sort -u > labels.csv")
        os.system("mv studio*.csv zoom*.csv labels")

    def download(self):
        module_root = vistec_ser.__path__[0]
        executable = os.path.join(module_root, f"data/datasets/{self.download_script}")
        assert os.path.exists(executable)
        subprocess.call(['sh', executable])
