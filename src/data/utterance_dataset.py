import torch
from torch.utils import data
import numpy as np
import pickle 
import os    
       
from multiprocessing import Process, Manager   

np.random.seed(1)

class Utterances(data.Dataset):
    """Dataset class for the Utterances dataset."""

    def __init__(self, root_dir, file_name, len_crop, use_accent=False):
        """Initialize and preprocess the Utterances dataset."""
        self.root_dir = root_dir
        self.file_name = file_name
        self.len_crop = len_crop
        self.use_accent = use_accent
        self.step = 10
        
        metaname = os.path.join(self.root_dir, file_name)
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
                   
        
    def __getitem__(self, index):
        # pick a random speaker
        dataset = self.train_dataset 
        list_uttrs = dataset[index]
        emb_org = list_uttrs[1]
        
        # pick random uttr with random crop
        a = np.random.randint(2, len(list_uttrs))
        tmp = list_uttrs[a]
        if tmp.shape[0] < self.len_crop:
            len_pad = self.len_crop - tmp.shape[0]
            uttr = np.pad(tmp, ((0,len_pad),(0,0)), 'constant')
        elif tmp.shape[0] > self.len_crop:
            left = np.random.randint(tmp.shape[0]-self.len_crop)
            uttr = tmp[left:left+self.len_crop, :]
        else:
            uttr = tmp

        if not self.use_accent:
            emb_org = np.split(emb_org, 2)[0]
        
        return uttr, emb_org
    

    def __len__(self):
        """Return the number of spkrs."""
        return self.num_tokens
