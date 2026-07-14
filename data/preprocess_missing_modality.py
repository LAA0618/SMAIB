import pandas as pd
import numpy as np
import argparse
from fractions import Fraction

def split_arr_(arr, k = 4) :
    res = []
    l = len(arr) / k
    check_ = 0
    for i in range(k) :
        from_, to_ = int(i * l), int((i+1) * l)
        res.append(arr[from_:to_])
        check_ += len(res[i])
    assert check_ == len(arr)
    return res

def parse_missing_ratio(value):
    try:
        ratio = float(Fraction(str(value)))
    except (ValueError, ZeroDivisionError):
        raise argparse.ArgumentTypeError('missing_ratio must be a float or fraction, e.g. 0.666 or 2/3')
    if ratio < 0 or ratio > 1:
        raise argparse.ArgumentTypeError('missing_ratio must be in [0, 1]')
    return ratio

def format_missing_ratio_name(ratio):
    return ('%.3f' % ratio).rstrip('0').rstrip('.')

if __name__ == "__main__" :

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', type=str, default='baby')
    parser.add_argument('--missing_ratio', type=parse_missing_ratio, default=parse_missing_ratio('0.666'))
    parser.add_argument('--seed', type=int, default=1225)
    args = parser.parse_args()

    dataset_name = args.dataset
    missing_ratio = args.missing_ratio
    missing_ratio_name = format_missing_ratio_name(missing_ratio)

    df = pd.read_csv(f"{dataset_name}/{dataset_name}.inter", sep = '\t')
    n_items = df['itemID'].nunique()

    all_items = np.arange(n_items)
    rng = np.random.default_rng(args.seed)
    missing_items = rng.choice(all_items, size = int(missing_ratio * n_items), replace = False)
    missing_item_groups = split_arr_(missing_items)

    missing_items_dict = {}
    missing_items_dict['t'] = missing_item_groups[0]
    missing_items_dict['v'] = missing_item_groups[1]
    missing_items_dict['all'] = np.concatenate((missing_item_groups[2], missing_item_groups[3]))

    np.save(f"{dataset_name}/missing_items_{missing_ratio_name}", missing_items_dict, allow_pickle = True)
