import os
import torch
import numpy as np


class TorchBatcher:
    def __init__(self, directory, min_time, max_time):
        """
        Expects dataset created by Sampler
        Args:
            directory: directory with samples
            min_time: lower bound for sampling time
            max_time: upper bound for sampling time
        """
        self.directory = directory
        self.possible_samples = []
        file_names = os.listdir(directory)

        for file_name in file_names:
            if file_name[0] == 'x':
                continue
            num = int(file_name[2:-4])
            if min_time <= num <= max_time:
                self.possible_samples.append(num)
        
    def __len__(self):
        return len(self.possible_samples)
    
    def __getitem__(self, indx):
        time = self.possible_samples[indx]
        x = np.load(os.path.join(self.directory, 'x_{}.npy'.format(time)))
        y = np.load(os.path.join(self.directory, 'y_{}.npy'.format(time)))
        return x, y
