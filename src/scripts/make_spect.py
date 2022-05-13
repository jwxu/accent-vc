import argparse
import os
import pickle

from librosa.filters import mel
import numpy as np
from numpy.random import RandomState
from scipy import signal
from scipy.signal import get_window
import soundfile as sf
from tqdm import tqdm


def get_arg_parse():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_dir', type=str, required=True, 
                default="dataset/wavs", help="Root audio directory")
    parser.add_argument('--output_spec_dir', type=str, required=True,
                default="dataset/spmel", help="Output spectrogram directory")

    args = parser.parse_args()
    return args


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a
    
    
def pySTFT(x, fft_length=1024, hop_length=256):
    
    x = np.pad(x, int(fft_length//2), mode='reflect')
    
    noverlap = fft_length - hop_length
    shape = x.shape[:-1]+((x.shape[-1]-noverlap)//hop_length, fft_length)
    strides = x.strides[:-1]+(hop_length*x.strides[-1], x.strides[-1])
    result = np.lib.stride_tricks.as_strided(x, shape=shape,
                                             strides=strides)
    
    fft_window = get_window('hann', fft_length, fftbins=True)
    result = np.fft.rfft(fft_window * result, n=fft_length).T
    
    return np.abs(result)    


def generate_spectrogram_v2(wav_dir, out_dir, config={}):
    """ Aux function to convert WAV files to spectrograms (primarily
    for the L2-Arctic dataset)
    """
    # TODO: Might need to adjust the min/max for L2-Arctic
    sr = config.get('sr', 16000)
    n_fft = config.get('n_fft', 1024)
    f_min = config.get('f_min', 90)
    f_max = config.get('f_max', 7600)
    n_mels = config.get('n_mels', 80)

    mel_basis = mel(sr=sr, n_fft=n_fft, fmin=f_min, fmax=f_max, n_mels=n_mels).T
    min_level = np.exp(-100 / 20 * np.log(10))
    b, a = butter_highpass(30, 16000, order=5)

    for wav_name in tqdm(os.listdir(wav_dir), desc="Converting WAV to Spectrogram"):
        wav_path = os.path.join(wav_dir, wav_name)
        out_path = os.path.join(out_dir, wav_name.replace('.wav', ''))
        
        # If the output spectrogram already exists, skip
        if os.path.isfile(out_path + '.npy'):
            continue

        wav, file_sr = sf.read(wav_path)
        # Remove drifting noise
        augmented_wav = signal.filtfilt(b, a, wav)
        # Skip adding random noise
        spectr = pySTFT(augmented_wav).T
        # Convert to MEL and normalize
        mel_spectr = np.dot(spectr, mel_basis)
        augmented_mel_spectr = 20 * np.log10(np.maximum(min_level, mel_spectr)) - 16
        augmented_mel_spectr = np.clip((augmented_mel_spectr + 100) / 100, 0, 1)
        np.save(out_path, augmented_mel_spectr.astype(np.float32), allow_pickle=False)


def generate_spectrogram(rootDir, targetDir, config={}):
    mel_basis = mel(16000, 1024, fmin=90, fmax=7600, n_mels=80).T
    min_level = np.exp(-100 / 20 * np.log(10))
    b, a = butter_highpass(30, 16000, order=5)

    dirName, subdirList, _ = next(os.walk(rootDir))
    print('Found directory: %s' % dirName)

    for subdir in sorted(subdirList):
        print(subdir)
        if not os.path.exists(os.path.join(targetDir, subdir)):
            os.makedirs(os.path.join(targetDir, subdir))
        _,_, fileList = next(os.walk(os.path.join(dirName,subdir)))
        prng = RandomState(int(subdir[1:])) 
        for fileName in sorted(fileList):
            # Read audio file
            x, fs = sf.read(os.path.join(dirName,subdir,fileName))
            # Remove drifting noise
            y = signal.filtfilt(b, a, x)
            # Ddd a little random noise for model roubstness
            wav = y * 0.96 + (prng.rand(y.shape[0])-0.5)*1e-06
            # Compute spect
            D = pySTFT(wav).T
            # Convert to mel and normalize
            D_mel = np.dot(D, mel_basis)
            D_db = 20 * np.log10(np.maximum(min_level, D_mel)) - 16
            S = np.clip((D_db + 100) / 100, 0, 1)    
            # save spect    
            np.save(os.path.join(targetDir, subdir, fileName[:-4]),
                    S.astype(np.float32), allow_pickle=False)    

    
if __name__ == "__main__":
    args = get_arg_parse()
    generate_spectrogram(args.audio_dir, args.output_spec_dir)
    
