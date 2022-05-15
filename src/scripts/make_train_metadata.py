"""
Generate speaker embeddings and metadata for training
"""
import argparse
from collections import defaultdict, OrderedDict
import math
import os
import pickle
import random
import re

import numpy as np
import torch
from tqdm import tqdm

from src.models.model_bl import D_VECTOR

random.seed(12345)

def get_arg_parse():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec_dir', type=str, required=True,
                default="dataset/spmel", help="Output spectrogram directory")
    parser.add_argument('--encoder_ckpt', type=str, required=True, 
                default="dataset/trained_models/3000000-BL.ckpt", help="Path to trained speaker encoder model")

    args = parser.parse_args()
    return args


def generate_metadata_files_v2(speaker_id, spectr_dir, metadata_dir, encoder_ckpt, config={}):
    name_re = re.compile("arctic_(\w)(\d{4})\.npy")

    input_dim = config.get("input_dim", 80)
    cell_dim = config.get("cell_dim", 768)
    embed_dim = config.get("embed_dim", 256)
    len_crop = config.get("len_crop", 128)
    num_uttrs = config.get("num_uttrs", 10)
    train_test_ratio = config.get("train_test_ratio")

    # If you specify a train_test_ratio, then no limit on num_uttrs
    if train_test_ratio:
        num_uttrs = math.inf

    train_metadata_path = os.path.join(metadata_dir, 'train.pkl')
    test_metadata_path = os.path.join(metadata_dir, 'test.pkl')
    partial_spectr_dir = os.path.basename(os.path.normpath(spectr_dir))

    # Currently only one type of encoder being used, but if we want to
    # try with other models, we can abstract this function out
    C = D_VECTOR(dim_input=input_dim, dim_cell=cell_dim, dim_emb=embed_dim).eval().cuda()
    c_checkpoint = torch.load(encoder_ckpt)
    new_state_dict = OrderedDict()
    for key, val in c_checkpoint['model_b'].items():
        new_key = key[7:]
        new_state_dict[new_key] = val
    C.load_state_dict(new_state_dict)

    # Divide spectrograms into train and test
    train_embedings, train_files = [], []
    test_embeddings, test_files = [], []
    for spectr_name in tqdm(os.listdir(spectr_dir), desc="Embedding Spectrogram Files"):
        if random.choice([True, False], 2, p=[train_test_ratio, 1-train_test_ratio]):
            embeddings = train_embedings
            files = train_files
        else:
            embeddings = test_embedings
            files = test_files
        spectr_path = os.path.join(spectr_dir, spectr_name)
        partial_spectr_path = os.path.join(partial_spectr_dir, spectr_name)
        name_match = name_re.search(spectr_name)
        if not name_match:
            print('No name match')
            continue
        speaker_name = name_match.group(1)
        clip_number = name_match.group(2)

        # If you reach num_uttrs for a given speaker, skip
        if len(embeddings) >= num_uttrs:
            continue

        # If spectrogram is not long enough, skip
        spectrogram = np.load(spectr_path)
        if spectrogram.shape[0] < len_crop:
            print(f'Len too short: {spectrogram.shape[0]}')
            continue

        left = np.random.randint(0, spectrogram.shape[0] - len_crop)
        mel_spectrogram = torch.from_numpy(spectrogram[np.newaxis, left:left+len_crop, :]).cuda()
        embedding = C(mel_spectrogram)
        embeddings.append(embedding.detach().squeeze().cpu().numpy())
        files.append(partial_spectr_path)

    # Each L2-Arctic file only has one speaker
    train = [[speaker_id, np.mean(train_embedings, axis=0), *train_files]]
    test = [[speaker_id, np.mean(test_embeddings, axis=0), *test_files]]

    with open(train_metadata_path, 'wb') as f:
        pickle.dump(train, f)
    with open(test_metadata_path, 'wb') as f:
        pickle.dump(test, f)


def generate_metadata_files(rootDir, encoder_ckpt):
    """
    Generates metadata file in form of (speaker_id, speaker embedding, speaker spectrogram filepaths)
    """
    C = D_VECTOR(dim_input=80, dim_cell=768, dim_emb=256).eval().cuda()
    c_checkpoint = torch.load(encoder_ckpt)
    new_state_dict = OrderedDict()
    for key, val in c_checkpoint['model_b'].items():
        new_key = key[7:]
        new_state_dict[new_key] = val
    C.load_state_dict(new_state_dict)
    num_uttrs = 10
    len_crop = 128

    # Directory containing mel-spectrograms
    dirName, subdirList, _ = next(os.walk(rootDir))
    print('Found directory: %s' % dirName)

    speakers = []
    for speaker in sorted(subdirList):
        print('Processing speaker: %s' % speaker)
        utterances = []
        utterances.append(speaker)
        _, _, fileList = next(os.walk(os.path.join(dirName,speaker)))
        
        # make speaker embedding
        assert len(fileList) >= num_uttrs
        idx_uttrs = np.random.choice(len(fileList), size=num_uttrs, replace=False)
        embs = []
        for i in range(num_uttrs):
            tmp = np.load(os.path.join(dirName, speaker, fileList[idx_uttrs[i]]))
            candidates = np.delete(np.arange(len(fileList)), idx_uttrs)
            # choose another utterance if the current one is too short
            while tmp.shape[0] < len_crop:
                idx_alt = np.random.choice(candidates)
                tmp = np.load(os.path.join(dirName, speaker, fileList[idx_alt]))
                candidates = np.delete(candidates, np.argwhere(candidates==idx_alt))
            left = np.random.randint(0, tmp.shape[0]-len_crop)
            melsp = torch.from_numpy(tmp[np.newaxis, left:left+len_crop, :]).cuda()
            emb = C(melsp)
            embs.append(emb.detach().squeeze().cpu().numpy())     
        utterances.append(np.mean(embs, axis=0))
        
        # create file list
        for fileName in sorted(fileList):
            utterances.append(os.path.join(speaker,fileName))
        speakers.append(utterances)
        
    with open(os.path.join(rootDir, 'train.pkl'), 'wb') as handle:
        pickle.dump(speakers, handle)


if __name__ == "__main__":
    args = get_arg_parse()
    generate_metadata_files(args.spec_dir, args.encoder_ckpt)
    
