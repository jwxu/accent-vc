import argparse

import gdown

from src.scripts.make_spect import generate_spectrogram_v2
from src.scripts.make_train_metadata import generate_metadata_files_v2


def get_arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arctic_data_dir', type=str, required=True,
                default="dataset/ABA", help="Top level directory of Arctic data")
    parser.add_argument('--encoder_ckpt', type=str, required=True,
                default="dataset/trained_models/3000000-BL.ckpt", help="Path to embedding encoder for MEL spectogram")
    args = parser.parse_args()
    return args


def process_arctic_data(args):
    arctic_dir = args.arctic_data_dir
    encoder_path = args.encoder_ckpt

    # If the encoder does not exist, and the default is being used, download it
    if not os.path.isfile(encoder_path) and '3000000-BL.ckpt' in encoder_path:
        encoder_download_url = "https://drive.google.com/file/d/1ORAeb4DlS_65WDkQN6LHx5dPyCM5PAVV"
        gdown.downlod(encoder_download_url, '3000000-BL.ckpt', quiet=False)
        os.replace(os.path.join(os.getcwd(), '3000000-BL.ckpt'), encoder_path)

    # WAV dir comes with the Arctic data
    wav_dir = os.path.join(arctic_dir, "wav")
    assert os.path.isdir(wav_dir)

    # Create a spectrogram dir for the data if not yet exists
    spectr_dir = os.path.join(arctic_dir, "spectrogram")
    if not os.path.isdir(spectr_dir):
        os.makedirs(spectr_dir)

    # Create a metadata dir
    autovc_meta_dir = os.path.join(arctic_dir, "autovc_metadata")
    if not os.path.isdir(autovc_meta_dir):
        os.makedirs(autovc_meta_dir)

    # Produce spectrograms
    # L2-Arctic uses 44.1 kHz sampling rate according to:
    # https://psi.engr.tamu.edu/wp-content/uploads/2018/08/zhao2018interspeech.pdf
    generate_spectrogram_v2(wav_dir, spectr_dir, {'sr': 44100})

    # Generate metadata files from spectrograms (train)
    genereate_metadata_files_v2(spectr_dir, autovc_meta_dir, encoder_path)


if __name__ == "__main__":
    args = get_arg_parse()
    process_arctic_data(args)
