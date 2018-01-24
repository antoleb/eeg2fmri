import os
import torch
import numpy as np


class DiskBatcher:
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

    def get_batch(self, batch_size):
        """
        Args:
            batch_size: batch size

        Returns:X, y

        """
        times = np.random.choice(self.possible_samples, size=batch_size)
        x_batch = np.array([np.load(os.path.join(self.directory, 'x_{}.npy'.format(time)))
                            for time in times])
        y_batch = np.array([np.load(os.path.join(self.directory, 'y_{}.npy'.format(time)))
                            for time in times])

        x_batch = torch.FloatTensor(x_batch)
        y_batch = torch.FloatTensor(y_batch)

        x_batch = torch.autograd.Variable(x_batch)
        y_batch = torch.autograd.Variable(y_batch)

        return x_batch, y_batch

