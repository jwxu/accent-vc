"""
Generate speaker embeddings and metadata for training
"""
import argparse
from collections import defaultdict, OrderedDict
import os
import pickle
import re

import numpy as np
import torch
from tqdm import tqdm

from src.models.model_bl import D_VECTOR


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


def generate_metadata_files_v2(spectr_dir, metadata_dir, encoder_ckpt, config={}):
    name_re = re.compile("arctic_(\w)(\d{4})\.npy")

    input_dim = config.get("input_dim", 80)
    cell_dim = config.get("cell_dim", 768)
    embed_dim = config.get("embed_dim", 256)
    len_crop = config.get("len_crop", 128)
    num_uttrs = config.get("num_uttrs", 10)     # per speaker

    metadata_path = os.path.join(metadata_dir, f'{num_uttrs}_{len_crop}_train.pkl')

    # Currently only one type of encoder being used, but if we want to
    # try with other models, we can abstract this function out
    C = D_VECTOR(dim_input=input_dim, dim_cell=cell_dim, dim_emb=embed_dim).eval().cuda()
    c_checkpoint = torch.load(encoder_ckpt)
    new_state_dict = OrderedDict()
    for key, val in c_checkpoint['model_b'].items():
        new_key = key[7:]
        new_state_dict[new_key] = val
    C.load_state_dict(new_state_dict)

    speaker_map = defaultdict(dict)
    for spectr_name in tqdm(os.listdir(spectr_dir), desc="Embedding Spectrogram Files"):
        spectr_path = os.path.join(spectr_dir, spectr_name)
        name_match = name_re.search(spectr_name)
        if not name_match:
            print('No name match')
            continue
        speaker_name = name_match.group(1)
        clip_number = name_match.group(2)

        # If you reach num_uttrs for a given speaker, skip
        if 'embeddings' in speaker_map[speaker_name] and len(speaker_map[speaker_name]['embeddings']) >= num_uttrs:
            continue

        # If spectrogram is not long enough, skip
        spectrogram = np.load(spectr_path)
        if spectrogram.shape[0] < len_crop:
            print(f'Len too short: {spectrogram.shape[0]}')
            continue

        left = np.random.randint(0, spectrogram.shape[0] - len_crop)
        mel_spectrogram = torch.from_numpy(spectrogram[np.newaxis, left:left+len_crop, :]).cuda()
        embedding = C(mel_spectrogram)

        if 'embeddings' not in speaker_map[speaker_name]:
            speaker_map[speaker_name]['embeddings'] = []
        speaker_map[speaker_name]['embeddings'].append(embedding.detach().squeeze().cpu().numpy())

        if 'files' not in speaker_map[speaker_name]:
            speaker_map[speaker_name]['files'] = []
        speaker_map[speaker_name]['files'].append(spectr_path)

    speakers = []
    for speaker, speaker_data in speaker_map.items():
        embed_mean = np.mean(speaker_data['embeddings'], axis=0)
        utterances = [speaker, embed_mean, *speaker_data['files']]
        speakers.append(utterances)

    with open(metadata_path, 'wb') as f:
        pickle.dump(speakers, f)


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
    
