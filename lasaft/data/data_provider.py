import time
from warnings import warn

from torch.utils.data import DataLoader
import torch

from lasaft.data.bbcso_wrapper import BBCSODataset


class DataProvider(object):

    def __init__(self, musdb_root,
                 batch_size, num_workers, pin_memory, n_fft, hop_length, num_frame,
                 multi_source_training):
        self.musdb_root = musdb_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.num_frame = num_frame
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.multi_source_training = multi_source_training

    def get_training_dataset_and_loader(self):
        if self.multi_source_training:
            training_set = MusdbTrainSetMultiSource(self.musdb_root, self.n_fft, self.hop_length, self.num_frame)
            if self.batch_size % 4 != 0:
                warn('batch_size % 4 should be zero. automatically adjusted')
                time.sleep(5)

        else:
            training_set = MusdbTrainSet(self.musdb_root, self.n_fft, self.hop_length, self.num_frame)

        batch_size = self.batch_size//4 if self.multi_source_training else self.batch_size
        loader = DataLoader(training_set, shuffle=True, batch_size=batch_size,
                            num_workers=self.num_workers,
                            pin_memory=self.pin_memory)

        return training_set, loader

    def get_validation_dataset_and_loader(self):
        validation_set = MusdbValidSetWithGT(self.musdb_root, self.n_fft, self.hop_length, self.num_frame)

        loader = DataLoader(validation_set, shuffle=False, batch_size=self.batch_size,
                            num_workers=self.num_workers,
                            pin_memory=self.pin_memory)

        return validation_set, loader

    def get_test_dataset_and_loader(self):
        test_set = MusdbTestSetWithGT(self.musdb_root, self.n_fft, self.hop_length, self.num_frame)

        loader = DataLoader(test_set, shuffle=False, batch_size=self.batch_size,
                            num_workers=self.num_workers,
                            pin_memory=self.pin_memory)

        return test_set, loader

class BBCSOProvider(object):
    def __init__(self, json_path,
                 batch_size, num_workers, pin_memory, n_fft, hop_length, num_frame,
                 multi_source_training):
        self.json_path = json_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.num_frame = num_frame
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.multi_source_training = multi_source_training
        self.segment = hop_length*(num_frame - 1)

        self.total_set = BBCSODataset(
            json_path,
            self.segment,
            2,
            44100,
        )
        self.train_set, self.val_set = torch.utils.data.random_split(self.total_set, [int(len(self.total_set)*0.9), len(self.total_set) - int(len(self.total_set)*0.9)])
    


    def get_training_dataset_and_loader(self): #need to combine training and val dataloader generation
        
        #batch_size = self.batch_size//4 if self.multi_source_training else self.batch_size
        loader = DataLoader(self.train_set, shuffle=True, batch_size=self.batch_size,
                            num_workers=self.num_workers,
                            pin_memory=self.pin_memory)

        return self.train_set, loader

    def get_validation_dataset_and_loader(self):
        loader = DataLoader(self.val_set, shuffle=False, batch_size=self.batch_size,
                            num_workers=self.num_workers,
                            pin_memory=self.pin_memory)

        return self.val_set, loader

    def get_test_dataset_and_loader(self):
        
        loader = DataLoader(self.total_set, shuffle=False, batch_size=self.batch_size,
                            num_workers=self.num_workers,
                            pin_memory=self.pin_memory)

        return self.total_set, loader
