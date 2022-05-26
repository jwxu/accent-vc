import os
import argparse
from src.models.solver_accent_emb import AccentEmbSolver
from src.data.data_loader import get_loader
from torch.backends import cudnn


def get_arg_parse():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser()

    # Training configuration.
    parser.add_argument('--data_dir', type=str, default='dataset/spmel')
    parser.add_argument('--file_name', type=str, default='train.pkl', help='Name of .pkl metadata file to use for training')
    parser.add_argument('--label_csv', type=str, default='dataset/speaker_to_accent_map.csv', help="Name of .csv file to map speaker to accent")
    parser.add_argument('--batch_size', type=int, default=32, help='mini-batch size')
    parser.add_argument('--epochs', type=int, default=50, help='number of total epochs')
    parser.add_argument('--validation_freq', type=int, default=5, help='validate every n epochs')
    parser.add_argument('--len_crop', type=int, default=128, help='dataloader output sequence length')
    
    # Miscellaneous.
    parser.add_argument('--log_step', type=int, default=100)
    parser.add_argument('--load_ckpt', type=str, default=None, help='Load from pretrained checkpoint')
    parser.add_argument('--checkpoints_dir', type=str, default="checkpoints", help='Directory to save checkpoints')
    parser.add_argument('--wandb', default=None, type=str, help="Wandb project name")
    parser.add_argument('--wandb_json', default=None, type=str, 
                        help="File path to wandb config json file")

    args = parser.parse_args()
    return args


def main(config):
    # For fast training.
    cudnn.benchmark = True

    # Data loader.
    accent_data_train = get_loader(root_dir=config.data_dir, dataset="accents", batch_size=config.batch_size, len_crop=config.len_crop, file_name=config.file_name, mode="train", label_csv=config.label_csv)
    print("Train Data: ", len(accent_data_train))

    accent_data_val = None
    if config.validation_freq > 0:
        accent_data_val = get_loader(root_dir=config.data_dir, dataset="accents", batch_size=config.batch_size, len_crop=config.len_crop, file_name=config.file_name, mode="val", label_csv=config.label_csv)
        print("Valdation Data: ", len(accent_data_train))
    
    solver = AccentEmbSolver(accent_data_train, config, accent_data_val)

    solver.train()
             

if __name__ == '__main__':
    args = get_arg_parse()
    print(args)
    main(args)
