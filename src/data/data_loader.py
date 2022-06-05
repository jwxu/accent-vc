from torch.utils import data
import torch
import numpy as np

from src.data.utterance_dataset import Utterances
from src.data.accent_dataset import Accents


def get_loader(root_dir, dataset, batch_size=16, len_crop=128, num_workers=0, file_name='train.pkl', mode="train", label_csv="", use_accent=False):
    """Build and return a data loader."""
    
    if dataset == "utterances":
        dataset = Utterances(root_dir, file_name, len_crop, use_accent)
        worker_init_fn = lambda x: np.random.seed((torch.initial_seed()) % (2**32))
        data_loader = data.DataLoader(dataset=dataset,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=num_workers,
                                    drop_last=True,
                                    worker_init_fn=worker_init_fn)
        return data_loader

    elif dataset == "accents":
        dataset = Accents(root_dir, file_name, len_crop, mode, label_csv)
        dataloader = data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True)
        return dataloader

    else:
        raise ValueError("Dataset [%s] not recognized." % dataset)
    
    






