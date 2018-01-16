import os

import numpy as np
import nibabel as nib
import mne
import torch

from scipy import signal

class Raw2RawDataset:
    def __init__(self, root_dir, num_test_people, random_seed=42, segment_length=10000, NFFT=300, tfr_len=64):
        self.NFFT = NFFT
        self.segment_length = segment_length
        self.tfr_len = tfr_len
        self.random_state = np.random.RandomState(random_seed)
        self.num_test_people = num_test_people
        self.root_dir = root_dir

        self.eeg_dir = os.path.join(root_dir, 'EEG')
        self.fmri_dir = os.path.join(root_dir, 'fMRI')

        self.all_people = np.array(os.listdir(self.fmri_dir))
        self.test_people = self.random_state.choice(self.all_people, size=self.num_test_people, replace=False)
        self.train_people = self.all_people[~np.isin(self.all_people, self.test_people)]

        self.eeg_data = {man: self.read_vhdr(os.path.join(self.eeg_dir, man))
                         for man in self.all_people}
        self.fmri_data = {man: self.read_img(os.path.join(self.fmri_dir, man))
                          for man in self.all_people}

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
        return ft[:self.tfr_len]

    def get_brain(self, fmri, time):
        time_shift = np.array([72*i for i in range(30)])
        frame_time = time // 2160
        time_shift = time - time_shift[np.newaxis, np.newaxis, ...]
        koeff = (time_shift%2160) / 2160
        return fmri[..., frame_time - 1] * koeff + fmri[..., frame_time] * (1 - koeff)

    def get_sample(self, people):
        man = self.random_state.choice(people)
        full_eeg = self.eeg_data[man]
        full_fmri = self.fmri_data[man]
        start = self.random_state.randint(2160, full_eeg.shape[-1]-self.segment_length)

        eeg = full_eeg[..., start:start+self.segment_length]
        eeg = np.concatenate([self.get_tfr(eeg[i])[np.newaxis, ...] for i in range(eeg.shape[0])], axis=0)
        fmri = self.get_brain(full_fmri, start)

        fmri = np.rollaxis(fmri, 2)
        return eeg[...,4:]/0.015 , fmri / 4095

    def get_batch(self, n, people):
        eeg_batch = []
        fmri_batch = []
        for i in range(n):
            eeg, fmri = self.get_sample(people)
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
        return self.get_batch(n, self.train_people)

    def get_test_batch(self, n):
        return self.get_batch(n, self.test_people)





