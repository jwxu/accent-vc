python3 -m src.eval.style_converter \
    --data_path "dataset/inference_results/demo/metadata.pkl" \
    --model_checkpoint "dataset/trained_models/autovc.ckpt" \
    --output_filepath "dataset/inference_results/demo/results.pkl";

python3 -m src.eval.vocoder \
    --spec_data_path "dataset/inference_results/demo/results.pkl" \
    --vocoder_path "dataset/trained_models/checkpoint_step001000000_ema.pth" \
    --output_dir "dataset/inference/demo";