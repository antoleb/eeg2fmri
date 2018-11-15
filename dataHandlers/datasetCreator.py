import mne
import os

import numpy as np
import nibabel as nib

from preprocessing.eeg2tftr import Eeg2Tftr
from preprocessing.fmriTranformer import FmriTransformer
from dataHandlers import settings




class Sampler:
    def __init__(self, root_dir, random_seed=42, segment_length=1024, eeg_nperseg=63, eeg_padded=False, eeg_scale=1e+5,
                 fmri_scale=4095**-1, num_slices=settings.num_slices, frame_creation_time=settings.frame_creation_time,
                 step=20):
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

        self.patient_list = np.array(os.listdir(self.fmri_dir))
        self.eeg_data = {man: self.read_vhdr(os.path.join(self.eeg_dir, man))
                         for man in self.patient_list}
        self.fmri_data = {man: self.read_img(os.path.join(self.fmri_dir, man))
                          for man in self.patient_list}



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
                return read_data(os.path.join(path, file))

    @staticmethod
    def read_img(path):
        files = os.listdir(path)
        for file in files:
            if file[-12:] == 'cross.nii.gz':
                return nib.load(os.path.join(path, file)).get_data()

    def create_one_man_dataset(self, patient, start_time, end_time):
        """
        Samples data from start_time to end_time for man
        Args:
            patient: string of man index
            start_time: lower bound of data creation time
            end_time: upper bound of data creation time

        Returns: (X, Y)

        """
        eeg = self.eeg_data[patient]
        fmri = self.fmri_data[patient]

        eegHandler = Eeg2Tftr(nperseg=self.eeg_nperseg, padded=self.eeg_padded, scale=self.eeg_scale)
        fmriHandler = FmriTransformer(num_slices=self.num_slices, frame_creation_time=self.frame_creation_time,
                                      fmri_scale=self.fmri_scale)

        start = start_time
        end = start + self.segment_length
        x_list = []
        y_list = []
        while end < eeg.shape[1] and end <= self.frame_creation_time * (fmri.shape[-1] - 1) and end < end_time:
            signal = eeg[..., start:end]

            x = eegHandler.transform(signal).reshape(-1)
            y = fmriHandler.get_fmri(end, fmri)
            x_list.append(x)
            y_list.append(y)

            start += self.step
            end += self.step

        x_list = np.array(x_list)
        y_list = np.array(y_list)

        return x_list, y_list


