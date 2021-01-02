import tensorflow_io as tfio


# --------------------- FREQ MASKING


class FreqMasking:
    def __init__(self,
                 num_masks: int = 1,
                 mask_factor: float = 27.):
        self.mask_factor = mask_factor
        self.num_masks = num_masks

    def augment(self, data):
        for _ in range(self.num_masks):
            data = tfio.experimental.audio.freq_mask(data, param=self.mask_factor)
        return data


# --------------------- TIME MASKING


class TimeMasking:
    def __init__(self,
                 num_masks: int = 1,
                 mask_factor: float = 40.):
        self.mask_factor = mask_factor
        self.num_masks = num_masks

    def augment(self, data):
        for _ in range(self.num_masks):
            data = tfio.experimental.audio.time_mask(data, param=self.mask_factor)
        return data
