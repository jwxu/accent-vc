import os
import argparse
from src.models.solver_encoder import Solver
from src.data.data_loader import get_loader
from torch.backends import cudnn


def get_arg_parse():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--lambda_cd', type=float, default=1, help='weight for hidden code loss')
    parser.add_argument('--dim_neck', type=int, default=32)
    parser.add_argument('--dim_emb', type=int, default=256)
    parser.add_argument('--dim_pre', type=int, default=512)
    parser.add_argument('--freq', type=int, default=32)
    parser.add_argument('--use_accent', type=bool, default=False)
    
    # Training configuration.
    parser.add_argument('--data_dir', type=str, default='dataset/spmel')
    parser.add_argument('--file_name', type=str, default='train.pkl', help='Name of .pkl metadata file to use for training')
    parser.add_argument('--batch_size', type=int, default=2, help='mini-batch size')
    parser.add_argument('--num_iters', type=int, default=1000000, help='number of total iterations')
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
    vcc_loader = get_loader(root_dir=config.data_dir, dataset="utterances", batch_size=config.batch_size, len_crop=config.len_crop, file_name=config.file_name)
    
    solver = Solver(vcc_loader, config)

    solver.train()
             

if __name__ == '__main__':
    args = get_arg_parse()
    print(args)
    main(args)
