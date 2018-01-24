import os
import torch
import numpy as np


class RamBatcher:
    def __init__(self, directory, min_time, max_time):
        """
        Expects dataset created by Sampler
        Args:
            directory: directory with samples
            min_time: lower bound for sampling time
            max_time: upper bound for sampling time
        """
        self.directory = directory
        file_names = os.listdir(directory)
        self.x_list = []
        self.y_list = []

        for file_name in file_names:
            if file_name[0] == 'x':
                continue
            num = int(file_name[2:-4])
            if min_time <= num <= max_time:
                self.x_list.append(np.load(os.path.join(self.directory, 'x_{}.npy'.format(num))))
                self.y_list.append(np.load(os.path.join(self.directory, 'y_{}.npy'.format(num))))

        self.x_list = np.array(self.x_list)
        self.y_list = np.array(self.y_list)
        self.possible_range = list(range(len(self.x_list)))

    def get_batch(self, batch_size):
        """
        Args:
            batch_size: batch size

        Returns:X, y

        """
        indexes = np.random.choice(self.possible_range, size=batch_size)
        x_batch = self.x_list[indexes]
        y_batch = self.y_list[indexes]

        x_batch = torch.FloatTensor(x_batch)
        y_batch = torch.FloatTensor(y_batch)

        x_batch = torch.autograd.Variable(x_batch)
        y_batch = torch.autograd.Variable(y_batch)

        return x_batch, y_batch

