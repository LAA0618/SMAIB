import os
import argparse
from fractions import Fraction
from pathlib import Path
from utils.quick_start import quick_start

os.environ['NUMEXPR_MAX_THREADS'] = '48'

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = REPOSITORY_ROOT / 'data'


def parse_missing_ratio(value):
    try:
        ratio = float(Fraction(str(value)))
    except (ValueError, ZeroDivisionError):
        raise argparse.ArgumentTypeError('missing_ratio must be a float or fraction, e.g. 0.666 or 2/3')
    if ratio < 0 or ratio > 1:
        raise argparse.ArgumentTypeError('missing_ratio must be in [0, 1]')
    return ratio


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='SAFIB', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='baby', help='name of datasets')
    parser.add_argument('--gpu_id', '-g', type=str, default='0', help='gpu_id')
    parser.add_argument('--missing_modal', type=int, default=1, help='missing_modal')
    parser.add_argument('--missing_ratio', type=parse_missing_ratio, default=parse_missing_ratio('0.666'), help='missing_ratio')
    parser.add_argument('--user_cold_start', '--cold_start_users', dest='cold_start_users', type=int, default=0,
                        help='user cold-start setting (0: standard evaluation [default], 1: user cold-start evaluation)')
    parser.add_argument('--cold_start_max_interactions', type=int, default=5, help='max train interactions for paper-style cold-start users')
    parser.add_argument('--data_root', type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument('--missing_items_file', type=str, default='')
    parser.add_argument('--seed', type=int, default=999)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--stopping_step', type=int, default=20)
    parser.add_argument('--train_batch_size', type=int, default=2048)
    parser.add_argument('--eval_batch_size', type=int, default=4096)

    args, _ = parser.parse_known_args()
    config_dict = {
        'gpu_id': args.gpu_id,
        'missing_modal': args.missing_modal,
        'missing_ratio': args.missing_ratio,
        'data_path': str(args.data_root.resolve()) + os.sep,
        'missing_items_file': args.missing_items_file,
        'cold_start_users': bool(args.cold_start_users),
        'cold_start_max_interactions': args.cold_start_max_interactions,
        'hyper_parameters': [],
        'seed': [args.seed],
        'epochs': args.epochs,
        'stopping_step': args.stopping_step,
        'train_batch_size': args.train_batch_size,
        'eval_batch_size': args.eval_batch_size,
        'metrics': ['Recall', 'NDCG', 'Precision', 'MAP'],
        'topk': [5, 10, 20, 50],
        'valid_metric': 'Recall@20',
    }

    quick_start(model=args.model, dataset=args.dataset, config_dict=config_dict, save_model=False)
