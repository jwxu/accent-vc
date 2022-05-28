# accent-vc

## Install requirements
`pip3 install -r requirements.txt`

## Prep environment and train
1. Run `python3 src/scripts/process_arctic_data.py --arctic_data_name all --train_test_ratio 0.90` to pull all L2-Arctic data and produce train, val, and test pickle files for them. The split ratio will be 90% train, 5% val, and 5% test. This will create the following directory dataset/l2-arctic with the various L2-Arctic data within.
2. Run `python3 src/scripts/combine_train_data.py --data_files all` to aggregate the train, val, and test from various speakers into one dataset. The 'all' parameter for --data_files can be replaced by a space separated list of all the speaker codes to use (e.g. ABA, ASI).
3. Run `python3 src/train_accent_emb.py --data_dir "dataset/l2-arctic/" --label_csv "dataset/speaker_to_accent_map.csv"` to train the accent encoder on L2-Arctic data.

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
