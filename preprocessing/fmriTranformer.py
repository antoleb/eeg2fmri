import numpy as np


class FmriTransformer:
    def __init__(self, num_slices=30, frame_creation_time=540, fmri_scale=4095**-1):
        """
        Args:
            num_slices: number of vertical slices
            frame_creation_time: time between two frames
        """
        assert frame_creation_time % num_slices == 0

        self.num_slices = num_slices
        self.fmri_scale = fmri_scale
        self.frame_creation_time = frame_creation_time
        self.slice_creation_time = frame_creation_time // num_slices

    def get_fmri(self, time, fmri_tensor):
        """
        Args:
            time: time of sample
            fmri_tensor: fmri_tensor: matrix [z, y, x, num] where num is time dimention

        Returns: smooted fmri for time

        """
        result = np.zeros_like(fmri_tensor[..., 0])
        for i in range(self.num_slices):
            frame_time = time // self.frame_creation_time
            k = (time % self.frame_creation_time) / self.frame_creation_time
            result[..., i] = fmri_tensor[..., i, frame_time - 1] * k + fmri_tensor[..., i, frame_time] * (1 - k)
            time += self.slice_creation_time
        return result * self.fmri_scale
