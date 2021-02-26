from glob import glob
from typing import Dict, List, Union
import json
import os
import zipfile

from torch.utils.data import DataLoader
from torchvision.transforms.transforms import Compose
import gdown
import numpy as np
import pandas as pd
import pytorch_lightning as pl

from ..ser_dataset import SERDataset
from ..ser_slice_dataset import SERSliceDataset, SERSliceTestDataset
from ..features.transform import FilterBank, SpecAugment

emo2idx = {emo: i for i, emo in enumerate(['Neutral', 'Angry', 'Happy', 'Sad', 'Frustrated'])}
idx2emo = {v: k for k, v in emo2idx.items()}
correctemo = {
    'Neutral': 'Neutral',
    'Angry': 'Anger',
    'Happy': 'Happiness',
    'Sad': 'Sadness',
    'Frustrated': 'Frustration'
}


class AISSERDataModule(pl.LightningDataModule):
    def __init__(
            self,
            test_fold: int,
            agreement_threshold: float = 0.7,
            slice_dataset: bool = True,
            sampling_rate: int = 16000,
            num_mel_bins: int = 40,
            frame_length: int = 50,  # in ms
            frame_shift: int = 10,  # in ms
            center_feats: bool = True,
            scale_feats: bool = True,
            mic_type: str = 'con',
            download_dir: str = None,
            experiment_dir: str = None,
            include_zoom: bool = True,
            max_len: int = 3,
            batch_size: int = 64,
            emotions: List[str] = None,
            num_workers: int = 0,
            *args,
            **kwargs):
        super().__init__(*args, **kwargs)
        if emotions is None:
            emotions = ["neutral", "anger", "happiness", "sadness"]
        # loading dataset config
        self.agreement_threshold = agreement_threshold
        self.mic_type = mic_type
        self.test_fold = test_fold

        # dataset type
        self.slice_dataset = slice_dataset

        # dataset config
        self.max_len = max_len
        self.batch_size = batch_size
        self.sampling_rate = sampling_rate
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        self.sec_to_frame = 10 * self.frame_shift
        self.num_mel_bins = num_mel_bins
        self.num_workers = num_workers

        # normalizing sample
        self.center_feats = center_feats
        self.scale_feats = scale_feats

        # config n_classes, avail emotion
        self.emotions = emotions
        self.n_classes = len(self.emotions)

        # config download dir
        if download_dir is None:
            self.download_root = f"{os.path.expanduser('~')}/vistec-ser_tmpfiles/vistec"
        else:
            self.download_root = f"{download_dir}/vistec-ser_tmpfiles/vistec"
        if not os.path.exists(self.download_root):
            os.makedirs(self.download_root)

        # config experiment dir
        if experiment_dir is None:
            self.experiment_root = f"{os.path.expanduser('~')}/vistec-ser_tmpfiles/exp_vistec"
        else:
            self.experiment_root = f"{experiment_dir}"
        self.experiment_dir = f"{self.experiment_root}/fold{self.test_fold}"
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)

        # define download URL
        self.download_ids = {
            "studio1-10": "1M69xuXhPE6YRFWatm0D4MiDi1blLNy0P",
            "studio11-20": "1MqEestPscu2ao_jKUdM9HLva4DZFxCXe",
            "studio21-30": "1lHhMEDs4YhnsGdKYBKbvidhFert_XF74",
            "studio31-40": "1-AOy30Lm0yEnK_Q44QrSsgQN-XBKmfoW",
            "studio41-50": "16iRYWn614AQjZoWlW9-Vc9f6TW_1Z4Ii",
            "studio51-60": "1YX3Xus9hJEfbhww1mHOG_osLJho9yFBf",
            "zoom1-10": "1-2QGXwfsDFfEqDl4KQ5jLPtDmbzuSc7z",
            "zoom11-20": "17DXFur1ZAA7IAkX4-xa0OHyDTRa_KmZP",
        }
        self.labels_id = "1-1UML7pI0A-NwzlYZMthO8lu5onsu2_y"
        self.github_url = {
            k: f"https://github.com/vistec-AI/dataset-releases/releases/download/v0.1/{k}.zip"
            for k in self.download_ids.keys()
            if k != "labels"
        }

        # define fold split
        self.fold_config = {
            0: [f"studio{s:03d}" for s in range(1, 11)],
            1: [f"studio{s:03d}" for s in range(11, 21)],
            2: [f"studio{s:03d}" for s in range(21, 31)],
            3: [f"studio{s:03d}" for s in range(31, 41)],
            4: [f"studio{s:03d}" for s in range(41, 51)],
            5: [f"studio{s:03d}" for s in range(51, 61)],
            # 6: [f"studio{s:03d}" for s in range(61, 71)],
            # 7: [f"studio{s:03d}" for s in range(71, 81)],
        }
        if include_zoom:
            self.fold_config[8] = [f"zoom{s:03d}" for s in range(1, 11)]
            self.fold_config[9] = [f"zoom{s:03d}" for s in range(11, 21)]
        assert self.test_fold in self.fold_config.keys()
        self.studio_list = []
        for studios in self.fold_config.values():
            for s in studios:
                self.studio_list.append(s)

        self.train = None
        self.val = None
        self.test = None

    def set_fold(self, fold):
        self.test_fold = fold
        self.experiment_dir = f"{self.experiment_root}/fold{self.test_fold}"
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)

    def prepare_data(self):
        """Run once as a preparation: download dataset, generate csv labels"""
        self._download()
        self._prepare_labels()

    def setup(self, *args, **kwargs):
        self.train = pd.concat([pd.read_csv(f"{self.download_root}/fold{i}.csv")
                                for i in self.fold_config.keys() if i != self.test_fold])
        test_split = pd.read_csv(f"{self.download_root}/fold{self.test_fold}.csv")
        test_studio = self.fold_config[self.test_fold]
        val_studio = test_studio[:len(test_studio) // 2]
        test_studio = test_studio[len(test_studio) // 2:]
        self.val = test_split[test_split["PATH"].apply(lambda x: x.split("/")[-3]).isin(val_studio)]
        self.test = test_split[test_split["PATH"].apply(lambda x: x.split("/")[-3]).isin(test_studio)]

    def train_dataloader(self) -> DataLoader:
        if self.slice_dataset:
            transform = Compose([FilterBank(
                frame_length=self.frame_length,
                frame_shift=self.frame_shift,
                num_mel_bins=self.num_mel_bins)])
            train_vistec = SERSliceDataset(
                csv_file=self.train,
                sampling_rate=self.sampling_rate,
                max_len=self.max_len,
                center_feats=self.center_feats,
                scale_feats=self.scale_feats,
                transform=transform)
        else:
            transform = Compose([
                FilterBank(
                    frame_length=self.frame_length,
                    frame_shift=self.frame_shift,
                    num_mel_bins=self.num_mel_bins),
                SpecAugment(
                    freq_mask_param=10,
                    time_mask_param=20,
                    n_freq_mask=1,
                    n_time_mask=2)])
            train_vistec = SERDataset(
                csv_file=self.train, sampling_rate=self.sampling_rate, max_len=self.max_len, transform=transform)
        return DataLoader(train_vistec, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        transform = Compose([FilterBank(
                    frame_length=self.frame_length,
                    frame_shift=self.frame_shift,
                    num_mel_bins=self.num_mel_bins)])
        if self.slice_dataset:
            val_vistec = SERSliceTestDataset(
                csv_file=self.val,
                max_len=self.max_len,
                center_feats=self.center_feats,
                scale_feats=self.scale_feats,
                transform=transform)
        else:
            val_vistec = SERDataset(csv_file=self.val, max_len=self.max_len, transform=transform)
        return DataLoader(val_vistec, batch_size=1, num_workers=self.num_workers)

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        transform = Compose([FilterBank(
                    frame_length=self.frame_length,
                    frame_shift=self.frame_shift,
                    num_mel_bins=self.num_mel_bins)])
        if self.slice_dataset:
            test_vistec = SERSliceTestDataset(
                csv_file=self.test,
                max_len=self.max_len,
                center_feats=self.center_feats,
                scale_feats=self.scale_feats,
                transform=transform
            )
        else:
            test_vistec = SERDataset(csv_file=self.test, max_len=self.max_len, transform=transform)
        return DataLoader(test_vistec, batch_size=1, num_workers=self.num_workers)

    def _get_audio_path(self, audio_name: str) -> str:
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

    def _prepare_labels(self):
        # format
        if not os.path.exists(f"{self.download_root}/labels.csv"):
            print("\n+-----------------------------------+")
            print("| Formatting labels...              |")
            print("+-----------------------------------+")

            json_path = f"{self.download_root}/labels.json"
            if not os.path.exists(json_path):
                raise FileNotFoundError(f"labels.json not found at {self.download_root}")

            print(f">formatting {json_path} ...")
            data = read_json(json_path)
            # Filter studio that doesn't appear in download_dir
            avail_studio = []
            for std in sorted(glob(f"{self.download_root}/*/")):
                std = std[:-1].split("/")[-1]
                std = std[0] + std[-3:]
                avail_studio.append(std)
            data = {k: v for k, v in data.items() if k.split("_")[0] in avail_studio}
            agreements = get_agreements(data)
            labels = pd.DataFrame([
                (f"{self._get_audio_path(k)}", correctemo[idx2emo[v]])
                for k, v in {k: convert_to_hardlabel(v, thresh=self.agreement_threshold)
                             for k, v in agreements.items()}.items()
                if v != -1
            ], columns=['PATH', 'EMOTION'])

            labels.to_csv(f"{self.download_root}/labels.csv", index=False)
        else:
            labels = pd.read_csv(f"{self.download_root}/labels.csv")
        if all(os.path.exists(f"{self.download_root}/fold{fold}.csv") for fold in self.fold_config.keys()):
            pass
        else:
            print("\n+-----------------------------------+")
            print("| Separating folds...               |")
            print("+-----------------------------------+")
            studio = labels["PATH"].apply(lambda x: x.split("/")[-3])
            for n_fold, studio_num in self.fold_config.items():
                print(f"\t>separating fold {n_fold}...")
                fold = labels[studio.isin(studio_num)]  # .drop("Unnamed: 0", axis=1)
                fold.to_csv(f"{self.download_root}/fold{n_fold}.csv", index=False)
            print(">finished separating folds")

    def _get_labels(self, mode="fold"):
        if not os.path.exists(f"{self.download_root}/labels.csv") or \
                not all(os.path.exists(f"{self.download_root}/fold{fold}.csv") for fold in self.fold_config.keys()):
            self._prepare_labels()
        if mode.lower().strip() == "fold":
            return {i: pd.read_csv(f"{self.download_root}/fold{i}.csv") for i in self.fold_config.keys()}
        elif mode.lower().strip() == "full":
            return pd.read_csv(f"{self.download_root}/labels.csv")
        else:
            raise KeyError(f"Invalid mode for getting labels. Expecting `fold|full` but got {mode}")

    def _download(self):
        # download dataset
        if not all(os.path.exists(f"{self.download_root}/{studio}.zip") for studio in self.download_ids.keys()):
            print("+-----------------------------------+")
            print("| Downloading dataset...            |")
            print("+-----------------------------------+\n")
        for f, gid in self.download_ids.items():
            if not os.path.exists(f"{self.download_root}/{f}.zip"):
                print(f">downloading {f}.zip ...")
                out_name = os.path.join(self.download_root, f"{f}.zip")
                if f != "studio51-60":
                    gdown.download(f"https://drive.google.com/uc?id={gid}", output=out_name, quiet=False)
                else:
                    if f in self.github_url.keys():
                        os.system(f"wget {self.github_url[f]} -O {out_name}")
            else:
                pass
        # download labels
        if not os.path.exists(f"{self.download_root}/labels.json"):
            gdown.download(
                f"https://drive.google.com/uc?id={self.labels_id}",
                output=f"{self.download_root}/labels.json", quiet=False)
        print("Finished Downloading Dataset\n")

        # extract
        stop_check = False
        for studios in self.fold_config.values():
            if stop_check:
                break
            for studio in studios:
                if not os.path.exists(f"{self.download_root}/{studio}"):
                    print("+-----------------------------------+")
                    print("| Extracting dataset...             |")
                    print("+-----------------------------------+\n")
                    stop_check = True
                    break
        for f in sorted(glob(f"{self.download_root}/*.zip")):
            if "studio" in f:
                start, end = os.path.basename(f).split(".")[0].replace("studio", "").split("-")
                directories = [f"{self.download_root}/studio{i:03d}" for i in range(int(start), int(end)+1)]
                if all(os.path.exists(d) for d in directories):
                    continue
            elif "zoom" in f:
                start, end = os.path.basename(f).split(".")[0].replace("zoom", "").split("-")
                directories = [f"{self.download_root}/zoom{i:03d}" for i in range(int(start), int(end) + 1)]
                if all(os.path.exists(d) for d in directories):
                    continue
            elif "labels" in f:
                json_files = [f"{z.replace('.zip', '_label.json')}"
                              for z in sorted(glob(f"{self.download_root}/*.zip")) if "labels" not in z]
                if all(os.path.exists(d) for d in json_files):
                    continue
            else:
                raise NameError(f"Error on name {f}")
            print(f">unzipping {f}...")
            assert os.path.exists(f)
            with zipfile.ZipFile(f) as zip_ref:
                zip_ref.extractall(self.download_root)


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
