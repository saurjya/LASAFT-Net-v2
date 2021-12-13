import random
from abc import ABCMeta
from pathlib import Path
from torch.utils.data import Dataset
from lasaft.utils.fourier import get_trim_length

import numpy as np
import soundfile as sf
import torch
import json
import torchaudio

from hydra.utils import to_absolute_path


def check_bbcso_valid(bbcso_train):
    if len(bbcso_train) > 0:
        pass
    else:
        raise Exception('Check bbcso json, something is wrong')

class BBCSODataset(Dataset):
    __metaclass__ = ABCMeta

    def __init__(self, json_file, seg_len=220500, n_src=2, sample_rate=22050, train=False):
        super(BBCSODataset, self).__init__()
        # Task setting
        self.json_file = json_file
        self.sample_rate = sample_rate
        self.n_src = n_src
        self.train = train
        self.segment = seg_len
        self.target_names = self.source_names = ['Violin', 'Viola', 'Cello', 'Bass']
        self.randomise = False
        with open(json_file, "r") as f:
            sources_infos = json.load(f)
        
        #self.sources = np.array(sources_infos)
        temp_list = []
        temp = np.array(sources_infos)
        
        for song in temp:
            tracks = {}
            for i in range(n_src):
                tracks.update({song[n_src+i]:song[i]})
            temp_list.append((tracks,song[-1]))
            '''
            temp_samples = sf.read(song[0])
            i = 0
            #temp_list.append(len(temp_samples))
            while (i < len(temp_samples[0])-220500):
                sub_song1 = temp_samples[0][i:i+220500]
                sub_song2 = sf.read(song[1])[0][i:i+220500]
                if sub_song1.any():
                    if sub_song2.any():
                        for inst in range(len(song)-3):
                            temp_list.append(list([song[0],song[1],i, song[inst+2]]))
                
                i += 220500
                '''
        self.sources = temp_list
        #with open(os.path.join('/data/EECS-Sandler-Lab/BBCSO/21_test_out.json'), "w") as f:
        #    json.dump(self.sources, f, indent=4)


    def __len__(self):
        return len(self.sources)

    def __getitem__(self, idx):
        """ Gets a mixture/sources pair.
        Returns:
            mixture, vstack([source_arrays])
        """
        seg_len = int(self.segment)
        """
        for src in self.sources[idx][:-1]:
            s, sr = sf.read(src, start=start, stop=stop, dtype="float32", always_2d=True)
            #s, sr = sf.read(src, start=0, stop=2205000, dtype="float32", always_2d=True)
            #s, sr = sf.read(src, dtype="float32", always_2d=True)
            #s = np.zeros((seg_len,))
            s = s.mean(axis=1)
            #sr = 44100
            source_arrays.append(s)
        """
        source_arrays = []
        if self.randomise:
            idx = random.randint(0, len(self.sources))
        instNames = list(self.sources[idx][0].keys())
        rand_target = instNames[random.randint(0, len(instNames) - 1)]
        for key in instNames:
            filename = self.sources[idx][0][key]
            start = int(self.sources[idx][-1])
            stop = start + seg_len
            s, sr = sf.read(filename, start=start, stop=stop, dtype="float32", always_2d=True)
            s = s.mean(axis=1, keepdims=True)
            if key is rand_target:
                source = torch.from_numpy(s)
            source_arrays.append(s)

        input_condition = np.array(self.source_names.index(rand_target), dtype=np.long)
        sources = torch.from_numpy(np.stack(source_arrays))

        mix = torch.stack(list(sources)).sum(0)
        return mix, source, torch.tensor(input_condition, dtype=torch.long)

        

    def get_infos(self):
        """ Get dataset infos (for publishing models).

        Returns:
            dict, dataset infos with keys `dataset`, `task` and `licences`.
        """
        infos = dict()
        infos["dataset"] = self.dataset_name
        infos["task"] = "chamber_sep_eval"
        infos["licenses"] = [BBCSO_license]
        return infos

