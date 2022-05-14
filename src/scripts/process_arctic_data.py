import argparse
import os
import shutil
import sys
import zipfile

import gdown

from src.scripts.make_spect import generate_spectrogram_v2
from src.scripts.make_train_metadata import generate_metadata_files_v2


def get_arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arctic_data_name', type=str, required=True,
                default="ABA", help="Name of the speaker accent to use ABA,ASI,BWC,etc.")
    parser.add_argument('--num_utterances', type=int, default=20, help="Number of utterances to use for train")
    parser.add_argument('--encoder_ckpt', type=str, required=False,
                default="dataset/trained_models/3000000-BL.ckpt", help="Path to embedding encoder for MEL spectogram")
    args = parser.parse_args()
    return args

def _get_arctic_data_map():
    return {
        'ABA': 'https://drive.google.com/uc?export=download&id=1LibCPfV6ezlJDcmBQ7m0cE5m_3ge8lO2',
        'ASI': 'https://drive.google.com/uc?export=download&id=1nbKukx2ZuviIjZav5WNnJV5MMJY7BXeY',
        'BWC': 'https://drive.google.com/uc?export=download&id=1_ZA060S-aFqVYzZaDtHYUNR8fnuoivA6',
        'EBVS': 'https://drive.google.com/uc?export=download&id=1Xs7u8Nm7MyYpixAxP5JAC5P16xSutbFk',
        'ERMS': 'https://drive.google.com/uc?export=download&id=1htV1mSl7DgISgKVagGzu63csiBViPps8',
        'HKK': 'https://drive.google.com/uc?export=download&id=13nGEWhGVELAiUoDQpSH3PcC4xoLFIwQ6',
        'NJS': 'https://drive.google.com/uc?export=download&id=17pLoWUhhOalFhcQZCcE96lZftqT5fauu',
        'HQTV': 'https://drive.google.com/uc?export=download&id=1ehNQ0NXjhqLPjQCTvtqH1KOWj-2y7Svh',
        'SKA': 'https://drive.google.com/uc?export=download&id=1DVjg9EYhmrRfg_KlTVL4KIwvXt0fa-vz',
        'YDCK': 'https://drive.google.com/uc?export=download&id=1VtJBTeG0Iul1ZHvZsQqKTmXz8d-FS6tT',
        'TNI': 'https://drive.google.com/uc?export=download&id=18uuQzD9WV3HGjMEu72aSELWKp6TW-2QR',
        'PNV': 'https://drive.google.com/uc?export=download&id=1Qj6MLABb_fnBaR-LsQVy9y9XDnY7f3w5',
        'HJK': 'https://drive.google.com/uc?export=download&id=1fxOKmhOqhGZVsMyzV_bwKpjFAFWjOMfA',
        'NCC': 'https://drive.google.com/uc?export=download&id=18Pxrx2NxsEds3UHxwGzQEF5yDiqBWkQf',
        'MBMPS': 'https://drive.google.com/uc?export=download&id=16CArT2LpGA1A7xJn_wGvzndsdeFTLVPs',
        'LXC': 'https://drive.google.com/uc?export=download&id=1dY9BG-TTVB-14wz1f656EKBCYfv_IINp',
        'TLV': 'https://drive.google.com/uc?export=download&id=11gQ3hWzzx_slqhcu_gyQg2uauq4iWANn',
        'SVBI': 'https://drive.google.com/uc?export=download&id=1iR9a-fiu3mTTrf2-20pMCLN3P1Zq-Pb1',
        'YBAA': 'https://drive.google.com/uc?export=download&id=1kd8SHx6fH9IwQab-Y-_HAEfURzngsYpF',
        'YKWK': 'https://drive.google.com/uc?export=download&id=1Jq13epxqWmc-oJizvacjDTzVdIMHi5rG',
        'ZHAA': 'https://drive.google.com/uc?export=download&id=1GrhaazNNU4iZvJJwshsxoiuPuBjMVBpA',
        'RRBI': 'https://drive.google.com/uc?export=download&id=1sqFnh5PrU9VNAoQhCNw1VqbwaD4HMkN0',
        'THV': 'https://drive.google.com/uc?export=download&id=1RXB6iGZep5a2ihvITkAwVKyUC_-Xf5VV',
        'TXHC': 'https://drive.google.com/uc?export=download&id=1dYC1qEyuvrky_Os9-1_XnuC0hYwBCc7v'
    }

def _download_gdrive_file(url, name, save_dir, unzip=False):
    print(f"Downloading data from: {url}")
    gdown.download(url, name, quiet=False)
    full_file_path = os.path.join(os.getcwd(), save_dir, name)
    full_file_dir = os.path.dirname(full_file_path)
    if not os.path.isdir(full_file_dir):
        os.makedirs(full_file_dir)
    shutil.move(os.path.join(os.getcwd(), name), full_file_path)
    print(f"Downloaded {name} to {full_file_dir}")

    if unzip:
        with zipfile.ZipFile(full_file_path, 'r') as zip_ref:
            zip_ref.extractall(full_file_dir)
        os.remove(full_file_path)
        print(f"Unzipped and deleted {full_file_path}")
    return full_file_path


def process_arctic_data(args):
    arctic_data_name = args.arctic_data_name
    num_utterances = args.num_utterances
    arctic_dir = os.path.join('dataset', arctic_data_name)
    encoder_path = args.encoder_ckpt

    arctic_map = _get_arctic_data_map()
    if arctic_data_name != 'all' and arctic_data_name not in arctic_map.keys():
        print(f"--arctic_data_name must equal 'all' or one of the following values: {arctic_map.keys()}")
        return

    # If the data does not exist, download and unzip it
    if not os.path.isdir(arctic_dir):
        if arctic_data_name == 'all':
            for name, url in arctic_map.items():
                f_path = _download_gdrive_file(url, f'{name}.zip', 'dataset', unzip=True)
        else:
            f_path = _download_gdrive_file(arctic_map[arctic_data_name], f'{arctic_data_name}.zip', 'dataset', unzip=True)


    # If the encoder does not exist, and the default is being used, download it
    if not os.path.isfile(encoder_path) and '3000000-BL.ckpt' in encoder_path:
        print(f"No encoder model found. Downloading encoder used by AutoVC from Google Drive")
        encoder_download_url = "https://drive.google.com/uc?export=download&id=1ORAeb4DlS_65WDkQN6LHx5dPyCM5PAVV"
        _download_gdrive_file(encoder_download_url, '3000000-BL.ckpt', os.path.dirname(encoder_path))


    # WAV dir comes with the Arctic data
    wav_dir = os.path.join(arctic_dir, "wav")
    assert os.path.isdir(wav_dir)

    # Create a spectrogram dir for the data if not yet exists
    spectr_dir = os.path.join(arctic_dir, "spectrogram")
    if not os.path.isdir(spectr_dir):
        os.makedirs(spectr_dir)

    # To work with default AutoVC code, put the train.pkl file at top level data dir
    autovc_meta_dir = arctic_dir
    if not os.path.isdir(autovc_meta_dir):
        os.makedirs(autovc_meta_dir)

    # Produce spectrograms
    # L2-Arctic uses 44.1 kHz sampling rate according to:
    # https://psi.engr.tamu.edu/wp-content/uploads/2018/08/zhao2018interspeech.pdf
    generate_spectrogram_v2(wav_dir, spectr_dir, {'sr': 44100})

    # Generate metadata files from spectrograms (train)
    metadata_config = {'num_uttrs': num_utterances}
    generate_metadata_files_v2(spectr_dir, autovc_meta_dir, encoder_path, metadata_config)


if __name__ == "__main__":
    args = get_arg_parse()
    process_arctic_data(args)
