import torch
from torch.utils import data
import numpy as np
import pickle 
import os   
import csv  
from multiprocessing import Process, Manager   


class Accents(data.Dataset):
    """Dataset class for the Accent dataset."""

    def __init__(self, root_dir, file_name, len_crop, mode, label_csv):
        """Initialize and preprocess the Accent dataset."""
        self.root_dir = root_dir
        self.file_name = file_name
        self.len_crop = len_crop
        self.step = 10

        if mode == "train":
            metaname = os.path.join(self.root_dir, "train.pkl")
        elif mode == "val":
            metaname = os.path.join(self.root_dir, "val.pkl")
        
        meta = pickle.load(open(metaname, "rb"))
        
        """Load data using multiprocessing"""
        manager = Manager()
        meta = manager.list(meta)
        dataset = manager.list(len(meta)*[None])  
        processes = []
        for i in range(0, len(meta), self.step):
            p = Process(target=self.load_data, 
                        args=(meta[i:i+self.step],dataset,i))  
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
            
        self.train_dataset = list(dataset)
        self.num_tokens = len(self.train_dataset)

        self.index_dict, self.labels = self.make_index_dict(label_csv)
        self.label_num = len(self.labels)
        
        print('Finished loading the dataset...')
        
        
    def load_data(self, submeta, dataset, idx_offset):  
        for k, sbmt in enumerate(submeta):    
            uttrs = len(sbmt)*[None]
            for j, tmp in enumerate(sbmt):
                if j < 2:  # fill in speaker id and embedding
                    uttrs[j] = tmp
                else: # load the mel-spectrograms
                    uttrs[j] = np.load(os.path.join(self.root_dir, tmp))
            dataset[idx_offset+k] = uttrs
    

    def make_index_dict(self, label_csv):
        speaker_accent_lookup = {}
        labels = set()
        with open(label_csv, 'r') as f:
            csv_reader = csv.DictReader(f)
            for row in csv_reader:
                speaker_accent_lookup[row['speaker']] = int(row['label'])
                labels.add(int(row['label']))
        return speaker_accent_lookup, labels
                   
        
    def __getitem__(self, index):
        dataset = self.train_dataset 
        uttr_data = dataset[index]

        # Generate labels
        speaker_id = uttr_data[0]
        labels = np.zeros(self.label_num)
        labels[int(self.index_dict[speaker_id.lower()])] = 1.0
        labels = torch.FloatTensor(labels)

        speaker_emb = uttr_data[1]
        
        # pick random uttr with random crop
        a = np.random.randint(2, len(uttr_data))
        uttr_raw = uttr_data[a]
        if uttr_raw.shape[0] < self.len_crop:
            len_pad = self.len_crop - uttr_raw.shape[0]
            uttr = np.pad(uttr_raw, ((0,len_pad),(0,0)), 'constant')
        elif uttr_raw.shape[0] > self.len_crop:
            left = np.random.randint(uttr_raw.shape[0] - self.len_crop)
            uttr = uttr_raw[left:left+self.len_crop, :]
        else:
            uttr = uttr_raw
        
        return uttr, labels
    

    def __len__(self):
        """Return the number of spkrs."""
        return self.num_tokens