# accent-vc

## Install requirements
`pip3 install -r requirements.txt`

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