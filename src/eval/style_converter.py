import os
import pickle
import torch
import numpy as np
import argparse
from math import ceil
from src.models.model_vc import Generator


def get_arg_parse():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True,
                default="dataset/inference_results/demo/metadata.pkl", help="Path to pkl file of data")
    parser.add_argument('--model_checkpoint', type=str, required=True,
                default="dataset/trained_models/autovc.ckpt", help="Path to trained WaveNet vocoder")
    parser.add_argument('--output_filepath', type=str, required=True,
                default="dataset/inference_results/demo/results.pkl", help="Path to pkl file to save vocoder outputs")

    args = parser.parse_args()
    return args


def pad_seq(x, base=32):
    len_out = int(base * ceil(float(x.shape[0])/base))
    len_pad = len_out - x.shape[0]
    assert len_pad >= 0
    return np.pad(x, ((0,len_pad),(0,0)), 'constant'), len_pad


def run_style_converter(data_path, model_checkpoint, output_filepath):
    """
    Run input data through style conversion
    Pickle file should be in formate (speaker_id, speech embedding, speech spectrogram)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    g_checkpoint = torch.load(model_checkpoint, map_location=torch.device(device))
    # If its an accent model, load accent checkpoint
    if 'model_accent' in g_checkpoint:
        G = Generator(32,512,512,32, use_accent=True).eval().to(device)
        G.load_state_dict(g_checkpoint['model_accent'])
    else:
        G = Generator(32,256,512,32).eval().to(device)
        G.load_state_dict(g_checkpoint['model'])

    metadata = pickle.load(open(data_path, "rb"))

    spect_vc = []

    for data_i in metadata:
                
        spect_org = data_i[2]
        # Just pick the first spect_org
        if isinstance(spect_org, list):
            spect_org = spect_org[0]
        spect_org, len_pad = pad_seq(spect_org)
        uttr_org = torch.from_numpy(spect_org[np.newaxis, :, :]).to(device)
        emb_org = torch.from_numpy(data_i[1][np.newaxis, :]).to(device)
        
        for data_j in metadata:
                    
            emb_trg = torch.from_numpy(data_j[1][np.newaxis, :]).to(device)
            
            with torch.no_grad():
                _, x_identic_psnt, _ = G(uttr_org, emb_org, emb_trg)
                
            if len_pad == 0:
                uttr_trg = x_identic_psnt[0, 0, :, :].cpu().numpy()
            else:
                uttr_trg = x_identic_psnt[0, 0, :-len_pad, :].cpu().numpy()
            
            spect_vc.append( ('{}x{}'.format(data_i[0], data_j[0]), uttr_trg) )
            
            
    with open(output_filepath, 'wb') as handle:
        pickle.dump(spect_vc, handle) 


if __name__ == "__main__":
    args = get_arg_parse()
    run_style_converter(args.data_path, args.model_checkpoint, args.output_filepath)
