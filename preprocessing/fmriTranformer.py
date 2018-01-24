import numpy as np


class FmriTransformer:
    def __init__(self, num_slices=30, frame_creation_time=540):
        """
        Args:
            num_slices: number of vertical slices
            frame_creation_time: time between two frames
        """
        assert frame_creation_time % num_slices == 0

        self.num_slices = num_slices
        self.frame_creation_time = frame_creation_time
        self.slice_creation_time = frame_creation_time // num_slices

    def get_fmri(self, time, fmri_tensor):
        """
        Args:
            time: time of sample
            fmri_tensor: fmri_tensor: matrix [z, y, x, num] where num is time dimention

        Returns: smooted fmri for time

        """
        time_shift = np.array([self.slice_creation_time * i for i in range(self.num_slices)])
        time_shift = time - time_shift[np.newaxis, np.newaxis, ...]
        frame_time = time // self.frame_creation_time
        koeff = (time_shift % self.frame_creation_time) / self.frame_creation_time

        assert frame_time < fmri_tensor.shape[-1]

        return fmri_tensor[..., frame_time - 1] * koeff + fmri_tensor[..., frame_time] * (1 - koeff)