import os
import torch
import argparse
import soundfile as sf
import pickle
from src.utils.synthesis import build_model
from src.utils.synthesis import wavegen


def get_arg_parse():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec_data_path', type=str, required=True, 
                default="dataset/inference_results/demo/results.pkl", help="Path to pkl file of melspec data")
    parser.add_argument('--vocoder_path', type=str, required=True,
                default="dataset/trained_models/checkpoint_step001000000_ema.pth", help="Path to trained WaveNet vocoder")
    parser.add_argument('--output_dir', type=str, required=True,
                default="dataset/inference/demo", help="Path to save vocoder outputs")

    args = parser.parse_args()
    return args


def run_vocoder(spec_data_path, vocoder_path, output_dir):
    """
    Run WaveNet vocoder
    Pickle file should be in the format of (filename, melspectrogram)
    """

    # Load model  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(vocoder_path, map_location=torch.device(device))
    model = build_model().to(device)
    model.load_state_dict(checkpoint["state_dict"])

    spect_vc = pickle.load(open(spec_data_path, 'rb'))
    for spect in spect_vc:
        filename = spect[0]
        spectrogram = spect[1]

        print("Processing ", filename)
        waveform = wavegen(model, c=spectrogram) 

        output_path = os.path.join(output_dir, filename + '.wav')  
        sf.write(output_path, waveform, samplerate=16000)


if __name__ == "__main__":
    args = get_arg_parse()
    run_vocoder(args.spec_data_path, args.vocoder_path, args.output_dir)
    