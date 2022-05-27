import argparse
import os
import pickle

from tqdm import tqdm

from src.scripts.process_arctic_data import get_arctic_data_map

def get_arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_files', nargs='+', required=True,
                help="List of L2-Arctic datasets to combine for training. You can also pass in the keyword `all`")
    args = parser.parse_args()
    return args


def combine_train_data(args):
    data_files = args.data_files
    arctic_map = get_arctic_data_map()
    if len(data_files) == 1 and data_files[0] == 'all':
        data_files = list(arctic_map.keys())

    # Verify that all list items are in 
    assert all(name in arctic_map for name in data_files)

    for dataset_type in ['train', 'test', 'val']:
        combined = []
        for name in tqdm(data_files):
            train_fp = os.path.join('dataset/l2-arctic', name, f'{dataset_type}.pkl')
            print(train_fp)
            assert os.path.isfile(train_fp)
            with open(train_fp, 'rb') as f:
                train_data = pickle.load(f)
            # Update all file paths to include the arctic data dir it belongs to
            if dataset_type != 'test':
                train_data[0][2:] = [os.path.join(name, path) for path in train_data[0][2:]]
            combined.append(train_data[0])
        with open(os.path.join('dataset/l2-arctic', f'{dataset_type}.pkl'), 'wb') as f:
            pickle.dump(combined, f)


if __name__ == "__main__":
    args = get_arg_parse()
    combine_train_data(args)
