from preprocessing.eeg2tftr import Eeg2Tftr
from preprocessing.fmriTranformer import FmriTransformer

import numpy as np
import nibabel as nib

import mne
import torch
import os


class FullStackBatcher:
    def __init__(self, root_dir, man, min_time, max_time, random_seed=42, segment_length=1024, eeg_nperseg=63, eeg_padded=False, eeg_scale=1e+5, fmri_scale=4095**-1,num_slices=30, frame_creation_time=540, f=1):
        self.min_time = max(min_time, segment_length)
        self.max_time = max_time
        self.eeg_nperseg = eeg_nperseg
        self.eeg_padded = eeg_padded
        self.eeg_scale = eeg_scale
        self.segment_length = segment_length

        self.num_slices = num_slices
        self.frame_creation_time = frame_creation_time
        self.fmri_scale = fmri_scale

        self.root_dir = root_dir

        self.random_state = np.random.RandomState(random_seed)
        self.eeg_dir = os.path.join(root_dir, 'EEG')
        self.fmri_dir = os.path.join(root_dir, 'fMRI')

        self.eeg = self.read_vhdr(os.path.join(self.eeg_dir, man))
        self.fmri = self.read_img(os.path.join(self.fmri_dir, man))
        
        self.eegHandler = Eeg2Tftr(nperseg=self.eeg_nperseg, padded=self.eeg_padded, scale=self.eeg_scale, f=f)
        self.fmriHandler = FmriTransformer(num_slices=self.num_slices, frame_creation_time=self.frame_creation_time,
                                      fmri_scale=self.fmri_scale)



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
    
    def get_sample(self):
        end = self.random_state.randint(self.min_time, self.max_time)
        start = end - self.segment_length
        signal = self.eeg[..., start:end]
        x = self.eegHandler.transform(signal)
        y = self.fmriHandler.get_fmri(end, self.fmri)
        y = np.rollaxis(y, 2)
        return x, y
        
        

    def get_batch(self, batch_size):
        """
        Args:
            batch_size: batch size

        Returns:X, y

        """
        x_batch, y_batch = [], []
        
        for _ in range(batch_size):
            x, y = self.get_sample()
            x_batch.append(x)
            y_batch.append(y)
 
        
        x_batch = np.array(x_batch)
        y_batch = np.array(y_batch)
        
        x_batch = torch.FloatTensor(x_batch)
        y_batch = torch.FloatTensor(y_batch)

        x_batch = torch.autograd.Variable(x_batch)
        y_batch = torch.autograd.Variable(y_batch)

        return x_batch, y_batch


