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

        self.mean_test_brain = self.fmri_tensor[..., first_test_frame:last_test_frame].mean(-1) * self.fmri_multiplicator
        self.mean_test_brain = np.rollaxis(self.mean_test_brain, 2)

        self.net = torch.load(os.path.join(report_directory, 'net.pt')).cuda().eval()

    @staticmethod
    def loss(a, b):
        return (a - b) ** 2

    def test(self):
        time = self.first_test_frame * self.frame_creation_time
        net_losses = []
        base_losses = []
        base_test_mean_losses = []
        grad_statistic = []
        frame_index_list = []
        slice_index_list = []
        net_predictions = []
        mean_predictions = []
        gt_predictions = []
        residual_loss_list = []
        loss = torch.nn.MSELoss()
        while time < self.last_test_frame * self.frame_creation_time:
            eeg = self.eeg_tensor[:,time-self.segment_length:time]
            eeg = np.array([self.eeg_transformer.transform(eeg)])
            eeg = torch.FloatTensor(eeg)
            eeg = torch.autograd.Variable(eeg.cuda(), requires_grad=True)

            output = self.net(eeg)

            frame_index = time // self.frame_creation_time
            slice_index = (time % self.frame_creation_time) // self.slice_creation_time

            gt = torch.autograd.Variable(torch.FloatTensor(self.fmri_tensor[..., frame_index]).cuda()) * self.fmri_multiplicator

            output_cpu = output.cpu().data.numpy()

            output_slice = output_cpu[0, slice_index]
            ground_truth_slice = self.fmri_tensor[..., slice_index, frame_index] * self.fmri_multiplicator
            mean_slice = self.mean_brain[slice_index]
            test_mean_slice = self.mean_test_brain[slice_index]

            net_predictions.append(output_slice)
            mean_predictions.append(mean_slice)
            gt_predictions.append(ground_truth_slice)

            net_losses.append(self.loss(output_slice, ground_truth_slice))
            base_losses.append(self.loss(mean_slice, ground_truth_slice))
            base_test_mean_losses.append(self.loss(test_mean_slice, ground_truth_slice))
            residual_loss_list.append(np.sqrt((ground_truth_slice - output_slice)/(ground_truth_slice * ground_truth_slice)))

            frame_index_list.append(frame_index)
            slice_index_list.append(slice_index)

            l = loss(output[0, slice_index], gt[..., slice_index])
            l.backward()

            grad_statistic.append(eeg.grad.cpu().data.numpy())

            time += self.slice_creation_time

        net_losses = np.array(net_losses)
        base_losses = np.array(base_losses)
        grad_statistic = np.array(grad_statistic)
        frame_index_list = np.array(frame_index_list)
        slice_index_list = np.array(slice_index_list)
        residual_loss_list = np.array(residual_loss_list)

        np.save(os.path.join(self.report_directory, 'base_losses.npy'), base_losses)
        np.save(os.path.join(self.report_directory, 'net_losses.npy'), net_losses)
        np.save(os.path.join(self.report_directory, 'base_test_mean_losses.npy'), base_test_mean_losses)
        np.save(os.path.join(self.report_directory, 'grad_statistic.npy'), grad_statistic)
        np.save(os.path.join(self.report_directory, 'frame_index_list.npy'), frame_index_list)
        np.save(os.path.join(self.report_directory, 'slice_index_list.npy'), slice_index_list)
        np.save(os.path.join(self.report_directory, 'net_predictions.npy'), net_predictions)
        np.save(os.path.join(self.report_directory, 'mean_predictions.npy'), mean_predictions)
        np.save(os.path.join(self.report_directory, 'gt_predictions.npy'), gt_predictions)
        np.save(os.path.join(self.report_directory, 'residual_loss_list.npy'), residual_loss_list)




