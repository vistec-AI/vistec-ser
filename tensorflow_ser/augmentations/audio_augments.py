import nlpaug.augmenter.audio as naa
import glob
import librosa
import os


class SignalCropping(naa.CropAug):
    def __init__(self,
                 zone=(0.2, 0.8),
                 coverage=0.1,
                 crop_range=(0.2, 0.8),
                 crop_factor=2):
        super().__init__(
            sampling_rate=None,
            zone=zone,
            coverage=coverage,
            crop_range=crop_range,
            crop_factor=crop_factor,
            duration=None)


class SignalLoudness(naa.LoudnessAug):
    def __init__(self,
                 zone=(0.2, 0.8),
                 coverage=1.,
                 factor=(0.5, 2)):
        super().__init__(
            zone=zone,
            coverage=coverage,
            factor=factor)


class SignalMask(naa.MaskAug):
    def __init__(self,
                 zone=(0.2, 0.8),
                 coverage=1.,
                 mask_range=(0.2, 0.8),
                 mask_factor=2,
                 mask_with_noise=True):
        super().__init__(
            sampling_rate=None,
            zone=zone,
            coverage=coverage,
            duration=None,
            mask_range=mask_range,
            mask_factor=mask_factor,
            mask_with_noise=mask_with_noise)


class SignalNoise(naa.NoiseAug):
    def __init__(self,
                 sample_rate=16000,
                 zone=(0.2, 0.8),
                 coverage=1.,
                 color="random",
                 noises: str = None):
        if noises is not None:
            noises = glob.glob(os.path.join(noises, "**", "*.wav"), recursive=True)
            noises = [librosa.load(n, sr=sample_rate)[0] for n in noises]
        super().__init__(
            zone=zone,
            coverage=coverage,
            color=color,
            noises=noises)


class SignalShift(naa.ShiftAug):
    def __init__(self,
                 sample_rate=16000,
                 duration=3,
                 direction="random"):
        super().__init__(
            sample_rate,
            duration=duration,
            direction=direction)


class SignalSpeed(naa.SpeedAug):
    def __init__(self,
                 zone=(0.2, 0.8),
                 coverage=1.,
                 factor=(0.5, 2)):
        super().__init__(
            zone=zone,
            coverage=coverage,
            factor=factor)


class SignalVtlp(naa.VtlpAug):
    def __init__(self,
                 sample_rate=16000,
                 zone=(0.2, 0.8),
                 coverage=0.1,
                 fhi=4800,
                 factor=(0.9, 1.1)):
        super().__init__(
            sample_rate,
            zone=zone,
            coverage=coverage,
            fhi=fhi,
            factor=factor)
