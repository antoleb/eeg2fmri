from .base_trainer import BaseTrainer

import os
import numpy as np


class ModelPerPersonTrainer:
    def __init__(self, dataset_path, save_path, num_train_frames, num_val_frames,
                 num_iters=5000, history_step=50, batch_size=128, fmri_mult=100,
                 max_batch_size=128, start_batch_size=1, batch_size_mul=2, batch_size_iters=100):
        assert not os.path.exists(save_path)
        os.makedirs(save_path)
        self.save_path = save_path
        self.dataset_path = dataset_path
        self.num_train_frames = num_train_frames
        self.num_val_frames = num_val_frames
        self.num_iters = num_iters
        self.history_step = history_step
        self.batch_size = batch_size
        self.fmri_mult = fmri_mult
        self.all_people = np.array(os.listdir(self.dataset_path))
        self.max_batch_size = max_batch_size
        self.start_batch_size = start_batch_size
        self.batch_size_mul = batch_size_mul
        self.batch_size_iters = batch_size_iters

    def train(self):
        for man in self.all_people:
            trainer = BaseTrainer(data_dir=os.path.join(self.dataset_path, man),
                                  num_train_frames=self.num_train_frames,
                                  num_val_frames=self.num_val_frames,
                                  save_dir=os.path.join(self.save_path, man),
                                  fmri_mult=self.fmri_mult,

                                  )

            trainer.train(self.num_iters,
                          self.history_step,
                          max_batch_size=self.max_batch_size,
                          start_batch_size=self.start_batch_size,
                          batch_size_mul=self.batch_size_mul,
                          batch_size_iters=self.batch_size_iters)

            print('done for man {}'.format(man))