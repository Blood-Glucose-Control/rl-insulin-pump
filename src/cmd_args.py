import argparse
import yaml


def parse_args() -> argparse.Namespace:
    r"""Parses the command line arguments."""
    parser = argparse.ArgumentParser(description='RL Insulin Pump')
    parser.add_argument('--cfg', dest='cfg_file', type=str, required=True,
                        help='The configuration file path.')
    args=parser.parse_args()
    with open(args.cfg_file) as f:
        args_dict = yaml.safe_load(f)
    return args_dict
