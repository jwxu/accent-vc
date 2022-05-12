python3 -m src.scripts.make_spect \
    --audio_dir "dataset/demo_wavs" \
    --output_spec_dir "dataset/demo_spmel";

python3 -m src.scripts.make_train_metadata \
    --spec_dir "dataset/demo_spmel" \
    --encoder_ckpt "dataset/trained_models/3000000-BL.ckpt";

python3 -m src.main \
    --data_dir "dataset/demo_spmel" \
    --num_iters 1000;