# accent-vc

## Install requirements
`pip3 install -r requirements.txt`

## Prep environment
Run `python3 src/scripts/process_arctic_data.py --arctic_data_name <ARCTIC_DATA_ABBREVIATION>` to preprocess L2-Arctic data into the corresponding spectrogram .npy and embedding .pkl files. Running this script will do the following:
1. Download a pretrained spectrogram encoder from AutoVC's GitHub and save it do datset/trained_models/3000000-BL.ckpt
2. Download the specified arctic data (ABA, ASI, etc) and unzip it to dataset/. A full list of abbreviations can be found here: https://psi.engr.tamu.edu/l2-arctic-corpus/.
3. Run src/scripts/make_spect.py on the downloaded WAV files to produce spectrograms that are saved to dataset/<ARCTIC_DATA_ABBREVIATION>/spectrogram/.
4. Run src/scripts/make_train_metadata.py on the .npy spectrogram files to produce .pkl metadata files that are saved to dataset/<ARCTIC_DATA_ABBREVIATION>/autovc_metadata/. These .pkl files are what the existing model uses for train/eval.

Things to note: in the process_arctic_data.py file, the generate_metadata_files_v2() function has a num_uttrs parameter passed to it. This specifies the number of utterances for a given speaker will be included in the final metadata .pkl file.

If there are any path issues with inter-directory dependencies, run `pip install -e .` to ensure that this module is also added on your path.

## Get pretrained models
Download all three pretrained models from autovc into dataset/trained_models

## Run demo
Demo Inference: From root accent-vc directory, run `sh bash_scripts/run_demo_inference.sh` to test on demo examples of autovc (need to use a gpu - the vocoder takes awhile to process)
- This first runs the style converter to convert an origin speaker to a target speaker using the pretrained AutoVC model
- Then runs a WaveNet vocoder to turn style-converted melspectrogram into speech audio files

Demo Train: From root accent-vc directory, run `sh bash_scripts/run_demo_train.sh` to train on demo examples of autovc (also needs gpu)
- This first creates all melspectrogram for all the audio files
- Then creates a train metadata file of the speaker, the speaker embedding (runs a pretrained speaker encoder), and the speaker melspectrogram filepaths
- Then trains

## wandb Logging
Create a json file with the content `{"key": <YOUR_WANDB_API_KEY>, "entity": <YOUR_WANDB_USERNAME>}`
Use the flags `--wandb <PROJECT_NAME>` and `--wandb_json <YOUR_WANDB.JSON_FILEPATH>`
