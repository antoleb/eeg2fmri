from samplers.ram_batcher import RamBatcher as Batcher
from nets.low_net import Net

import torch
import os
import numpy as np

class BaseTrainer:
    def __init__(self, data_dir, num_train_frames, num_val_frames, save_dir):
        assert not os.path.exists(save_dir)
        os.makedirs(save_dir)

        self.save_dir = save_dir
        self.train_batcher = Batcher(data_dir, 0, 540 * num_train_frames)
        self.val_batcher = Batcher(data_dir, 540 * (num_train_frames + 1), 540 * (num_train_frames + num_val_frames))
        self.net = Net().cuda()
        self.loss = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-4)

    def save(self, history, val_history):
        history = np.array(history)
        val_history = np.array(val_history)
        np.save(os.path.join(self.save_dir, 'train_history.npy'), history)
        np.save(os.path.join(self.save_dir, 'val_history.npy'), val_history)
        torch.save(self.net, os.path.join(self.save_dir, 'net.pt'))

    def train(self, num_iters, history_step, batch_size=128):
        iteration = 0
        history = []
        val_history = []

        for _ in range(num_iters):
            iteration += 1
            X, y = self.train_batcher.get_batch(batch_size)
            X = X.cuda()
            y = y.cuda()
            res = self.net(X)
            self.optimizer.zero_grad()
            l = self.loss(res, y)
            l.backward()
            self.optimizer.step()

            if iteration % history_step == 0:
                history.append(float(l.data))
                e, f = self.val_batcher.get_batch(batch_size)
                e = e.cuda()
                f = f.cuda()
                res = self.net(e)
                l = self.loss(res, f)
                val_history.append(l.data)

        self.save(history, val_history)
