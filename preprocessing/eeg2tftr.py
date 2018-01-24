from scipy import signal
import numpy as np


class Eeg2Tftr:
    def __init__(self, nperseg=63, padded=False, scale=1e+5):
        """
        Args:
            nperseg:  Length of each segment
            padded: Specifies whether the input signal is zero-padded at the end
            scale: Scale constant for time-freq matrix multiplication
        """

        self.nperseg = nperseg
        self.padded = padded
        self.scale = scale

    def transform(self, sig):
        """
        Args:
            sig: signal to transform

        Returns: transfored to time-freq matrix signal

        """
        f, t, ft = signal.stft(sig, padded=self.padded, nperseg=self.nperseg)
        ft = np.log1p(np.abs(ft))
        ft *= self.scale
        return ft

