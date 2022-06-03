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

from src.models.model_bl import D_VECTOR, D_ACCENT_VECTOR

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


def generate_metadata_files_v2(speaker_id, spectr_dir, metadata_dir, encoder_ckpt, accent_encoder_ckpt, config={}):
    print(f"Generating metadata for {speaker_id}")
    name_re = re.compile("arctic_(\w)(\d{4})\.npy")

    input_dim = config.get("input_dim", 80)
    cell_dim = config.get("cell_dim", 768)
    embed_dim = config.get("embed_dim", 256)
    len_crop = config.get("len_crop", 128)
    num_uttrs = config.get("num_uttrs", 10)
    train_test_ratio = config.get("train_test_ratio")   # By default test ratio will in half for dev and test

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

    if accent_encoder_ckpt:
        A = D_ACCENT_VECTOR(dim_input=input_dim, dim_cell=cell_dim, dim_emb=embed_dim).eval().cuda()
        a_checkpoint = torch.load(accent_encoder_ckpt)
        new_state_dict = OrderedDict()
        for key, val in a_checkpoint['model_accent'].items():
            new_state_dict[key] = val
        A.load_state_dict(new_state_dict)

    # Split data into train and test
    spectr_files = os.listdir(spectr_dir)
    num_files = len(spectr_files)
    if train_test_ratio:
        random.shuffle(spectr_files)
        train_split_ind = round(num_files * train_test_ratio)
        val_split_ind = round(num_files * (train_test_ratio + (1 - train_test_ratio) / 2))
        
        data = [
            ('train', spectr_files[:train_split_ind]),
            ('val', spectr_files[train_split_ind:val_split_ind]),
            ('test', spectr_files[val_split_ind:])
        ]
        num_uttrs = math.inf
    else:
        random_spectr_files = random.sample(spectr_files, 3 * num_uttrs)
        data = [
            ('train', random_spectr_files[:num_uttrs]),
            ('val', random_spectr_files[num_uttrs:2*num_uttrs]),
            ('test', random_spectr_files[2*num_uttrs:])
        ]

    # Divide spectrograms into train and test
    for subset in data:
        dataset_type, spectr_file_list = subset
        embeddings, files = [], []
        aux_paths, spectrogram_list = [], []
        for spectr_name in tqdm(spectr_file_list, desc=f"Embedding {dataset_type} spectrograms"):
            spectr_path = os.path.join(spectr_dir, spectr_name)
            partial_spectr_path = os.path.join(partial_spectr_dir, spectr_name)

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
            style_embedding = C(mel_spectrogram)
            if accent_encoder_ckpt:
                accent_embedding = A(mel_spectrogram)
                combined_embedding = np.concatenate((style_embedding.detach().squeeze().cpu().numpy(), accent_embedding.detach().squeeze().cpu().numpy()))
            else:
                combined_embedding = style_embedding.detach().squeeze().cpu().numpy()
            embeddings.append(combined_embedding)
            if dataset_type == "test":
                spectrogram_list.append(spectrogram)
                files.append(combined_embedding)
                aux_paths.append(partial_spectr_path)
            else:
                files.append(partial_spectr_path)

        if dataset_type == "test":
            speaker_data = [[speaker_id, np.mean(embeddings, axis=0), spectrogram_list]]
        else:
            speaker_data = [[speaker_id, np.mean(embeddings, axis=0), *files]]
        metadata_path = os.path.join(metadata_dir, f'{dataset_type}.pkl')
        with open(metadata_path, 'wb') as f:
            pickle.dump(speaker_data, f)


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
    
