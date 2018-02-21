from preprocessing.eeg2tftr import Eeg2Tftr
from preprocessing.fmriTranformer import FmriTransformer

import numpy as np
import nibabel as nib

import mne
import torch
import os


class Sampler:
    def __init__(self, root_dir, random_seed=42, segment_length=1024, eeg_nperseg=63, eeg_padded=False, eeg_scale=1e+5,
                 fmri_scale=4095**-1,num_slices=30, frame_creation_time=540, step=20):
        self.eeg_nperseg = eeg_nperseg
        self.eeg_padded = eeg_padded
        self.eeg_scale = eeg_scale
        self.segment_length = segment_length

        self.num_slices = num_slices
        self.frame_creation_time = frame_creation_time
        self.fmri_scale = fmri_scale

        self.step = step
        self.root_dir = root_dir

        self.random_state = np.random.RandomState(random_seed)
        self.eeg_dir = os.path.join(root_dir, 'EEG')
        self.fmri_dir = os.path.join(root_dir, 'fMRI')

        self.all_people = np.array(os.listdir(self.fmri_dir))
        self.eeg_data = {man: self.read_vhdr(os.path.join(self.eeg_dir, man))
                         for man in self.all_people}
        self.fmri_data = {man: self.read_img(os.path.join(self.fmri_dir, man))
                          for man in self.all_people}



    @staticmethod
    def read_vhdr(path):
        def read_data(path):
            eeg = mne.io.read_raw_brainvision(path, event_id={"Scan Start": 1})
            data = np.delete(eeg.get_data(), [len(eeg.ch_names) - 1], 0)
            return data[:, mne.find_events(eeg)[0][0]:]

        path = os.path.join(path, 'export')
        files = os.listdir(path)
        for file in files:
            if file[-5:] == '.vhdr':
                return read_data(os.path.join(path, file))[:,::-1]

    @staticmethod
    def read_img(path):
        files = os.listdir(path)
        for file in files:
            if file[-12:] == 'cross.nii.gz':
                return nib.load(os.path.join(path, file)).get_data()

    def create_one_man_dataset(self, man, dataset_path):
        """
        Creates dataset for one man
        Args:
            man: string of man index
            dataset_path: path to create dataset

        Returns: None

        """
        assert not os.path.exists(dataset_path)
        os.makedirs(dataset_path)
        eeg = self.eeg_data[man]
        fmri = self.fmri_data[man]

        eegHandler = Eeg2Tftr(nperseg=self.eeg_nperseg, padded=self.eeg_padded, scale=self.eeg_scale)
        fmriHandler = FmriTransformer(num_slices=self.num_slices, frame_creation_time=self.frame_creation_time,
                                      fmri_scale=self.fmri_scale)

        start = 0
        end = start + self.segment_length
        while end < eeg.shape[1] and end <= self.frame_creation_time * (fmri.shape[-1] - 1):
            signal = eeg[..., start:end]

            x = eegHandler.transform(signal)
            y = fmriHandler.get_fmri(end, fmri)
            y = np.rollaxis(y, 2)

            x_path = os.path.join(dataset_path, 'x_{}.npy'.format(end))
            y_path = os.path.join(dataset_path, 'y_{}.npy'.format(end))

            np.save(x_path, x)
            np.save(y_path, y)

            start += self.step
            end += self.step

    def create_dataset(self, dataset_path):
        """
        Creates dataset with one man dataset subfolders
        Args:
            dataset_path: path to create dataset

        Returns: None

        """
        assert not os.path.exists(dataset_path)
        os.makedirs(dataset_path)

        for man in self.all_people:
            self.create_one_man_dataset(man, os.path.join(dataset_path, man))




