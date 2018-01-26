from nets.low_net import Net

import torch
import os
import numpy as np


class BaseTester:
    def __init__(self, report_directory, fmri_tensor, eeg_tensor, eeg_transformer, first_test_frame, last_test_frame,
                 first_train_frame, last_train_frame,
                 fmri_multiplicator=100/4095, num_slices=30, frame_creation_time=540, segment_length=1024):
        self.report_directory = report_directory
        self.fmri_tensor = fmri_tensor
        self.eeg_tensor = eeg_tensor
        self.eeg_transformer = eeg_transformer
        self.fmri_multiplicator = fmri_multiplicator
        self.num_slices = num_slices
        self.frame_creation_time = frame_creation_time
        self.segment_length = segment_length
        self.slice_creation_time = frame_creation_time // num_slices

        self.first_test_frame = first_test_frame
        self.last_test_frame = last_test_frame
        self.first_train_frame = first_train_frame
        self.last_train_frame = last_train_frame

        self.mean_brain = self.fmri_tensor[..., first_train_frame:last_train_frame+1].mean(-1) * self.fmri_multiplicator
        self.mean_brain = np.rollaxis(self.mean_brain, 2)

        self.net = torch.load(os.path.join(report_directory, 'net.pt')).cuda()

    @staticmethod
    def loss(a, b):
        return np.sum((a - b) ** 2)

    def test(self):
        time = self.first_test_frame * self.frame_creation_time
        net_losses = []
        base_losses = []
        while time < (self.last_test_frame + 1) * self.frame_creation_time:
            eeg = self.eeg_tensor[:,time-self.segment_length:time]
            eeg = np.array([self.eeg_transformer.transform(eeg)])
            eeg = torch.FloatTensor(eeg)
            eeg = torch.autograd.Variable(eeg)
            eeg = eeg.cuda()

            output = self.net(eeg).data.numpy()

            frame_index = time // self.frame_creation_time
            slice_index = (time % self.frame_creation_time) // self.slice_creation_time

            output_slice = output[0, slice_index]
            ground_truth_slice = self.fmri_tensor[..., slice_index, frame_index] * self.fmri_multiplicator
            mean_slice = self.mean_brain[slice_index]

            net_losses.append(self.loss(output_slice, ground_truth_slice))
            base_losses.append(self.loss(mean_slice, ground_truth_slice))

            time += self.slice_creation_time

        net_losses = np.array(net_losses)
        base_losses = np.array(base_losses)

        np.save(os.path.join(self.report_directory, 'base_losses.npy'), base_losses)
        np.save(os.path.join(self.report_directory, 'net_losses.npy'), net_losses)





