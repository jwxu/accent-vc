import torch
from torch.utils import data
import numpy as np
import pickle 
import os   
import csv  
import random


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
                    
        self.train_dataset = self.load_data(meta)
        random.shuffle(self.train_dataset)
        
        self.num_tokens = len(self.train_dataset)
        print("Number of " + mode + " samples: ", self.num_tokens)

        self.index_dict, self.labels = self.make_index_dict(label_csv)
        self.label_num = len(self.labels)
        
        print('Finished loading the dataset...')
    

    def load_data(self, meta):
        """
        Meta is in the form of [
            [speaker_1, speaker_1_emb, melspec_path_1, melspec_path_2, ...], 
            [speaker_2, speaker_2_emb, melspec_path_1, melspec_path_2, ...],
            ...]

        Loads data in the form of [
            [speaker_1, melspec_1], 
            [speaker_1, melspec_2], 
            ...]
        """
        speaker_utt_data = []
        for i, speaker_info in enumerate(meta):
            speaker_id = speaker_info[0]
            for j, utterance in enumerate(speaker_info):
                if j >= 2: # indices 0 and 1 are speaker id and speaker embedding
                    melspec = np.load(os.path.join(self.root_dir, utterance))
                    speaker_utt_data.append([speaker_id, melspec])

        return speaker_utt_data
    

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

        # Generate input utterance
        uttr_raw = uttr_data[1]
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