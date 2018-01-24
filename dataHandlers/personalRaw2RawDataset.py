import os

import numpy as np
import nibabel as nib
import mne
import torch

from scipy import signal
import math


class PersonalRaw2RawDataset:
    def __init__(self, root_dir, man, test_part=0.2, val_part=0.1, random_seed=42, segment_length=1000, NFFT=65,
                 tfr_len=64, stft_shift=1):
        self.NFFT = NFFT
        self.segment_length = segment_length
        self.tfr_len = tfr_len
        self.random_state = np.random.RandomState(random_seed)
        self.test_part = test_part
        self.val_part = val_part
        self.root_dir = root_dir
        self._man = man
        self.max_time = 300 * 30 * 72 // 4
        self.stft_shift = stft_shift

        self.eeg_dir = os.path.join(root_dir, 'EEG')
        self.fmri_dir = os.path.join(root_dir, 'fMRI')

        self.eeg_data = self.read_vhdr(os.path.join(self.eeg_dir, man))
        self.fmri_data = self.read_img(os.path.join(self.fmri_dir, man))

    def read_vhdr(self, path):
        path = os.path.join(path, 'export')
        files = os.listdir(path)
        for file in files:
            if file[-5:] == '.vhdr':
                return mne.io.read_raw_brainvision(os.path.join(path, file)).get_data()

    def read_img(self, path):
        files = os.listdir(path)
        for file in files:
            if file[-12:] == 'cross.nii.gz':
                return nib.load(os.path.join(path, file)).get_data()

    def get_tfr(self, sig):
        f, t, ft = signal.stft(sig - np.median(sig), nfft=self.NFFT, nperseg=self.NFFT, return_onesided=True)
        ft = np.log1p(np.abs(ft))
        return ft[-self.tfr_len:]

    def get_brain(self, fmri, time):
        time_shift = np.array([18 * i for i in range(30)])
        frame_time = time // 540
        time_shift = time - time_shift[np.newaxis, np.newaxis, ...]
        koeff = (time_shift % 540) / 540
        return fmri[..., frame_time - 1] * koeff + fmri[..., frame_time] * (1 - koeff)

    def get_sample(self, interval):
        full_eeg = self.eeg_data
        full_fmri = self.fmri_data
        start = self.random_state.randint(min(540, interval[0]),
                                          min(interval[1], min(full_eeg.shape[-1], 300 * 30 * 72 // 4))\
                                          - self.segment_length)

        eeg = full_eeg[..., start:start + self.segment_length]
        eeg = np.concatenate([self.get_tfr(eeg[i])[np.newaxis, ...] for i in range(eeg.shape[0])], axis=0)
        fmri = self.get_brain(full_fmri, start+self.segment_length)

        fmri = np.rollaxis(fmri, 2)
        return eeg[:, self.stft_shift:] / 1e-5, fmri / 4095

    def get_batch(self, n, interval):
        eeg_batch = []
        fmri_batch = []
        for i in range(n):
            eeg, fmri = self.get_sample(interval)
            eeg_batch.append(eeg[np.newaxis, ...])
            fmri_batch.append(fmri[np.newaxis, ...])

        eeg_batch = np.concatenate(eeg_batch, axis=0)
        fmri_batch = np.concatenate(fmri_batch, axis=0)

        eeg_batch = torch.FloatTensor(eeg_batch)
        fmri_batch = torch.FloatTensor(fmri_batch)

        eeg_batch = torch.autograd.Variable(eeg_batch)
        fmri_batch = torch.autograd.Variable(fmri_batch)
        return eeg_batch, fmri_batch

    def get_train_batch(self, n):
        lower = 0
        upper = (math.floor(300 * (1-self.val_part-self.test_part)) * 30 * 72) // 4
        return self.get_batch(n, [lower, upper])

    def get_val_batch(self, n):
        lower = (math.ceil(300 * (1 - self.val_part - self.test_part) + 1e-6) * 30 * 72) // 4
        upper = (math.floor(300 * (1 - self.test_part)) * 30 * 72) // 4
        return self.get_batch(n, [lower, upper])

    def get_test_batch(self, n):
        lower = (math.ceil(300 * (1 - self.test_part)) * 30 * 72+1e-6) // 4
        upper = (300 * 30 * 72) // 4
        return self.get_batch(n, [lower, upper])





