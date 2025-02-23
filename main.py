import argparse
from pathlib import Path
import pandas as pd
import os

from src.data_preparation import preprocess
from src.hyper_param_optimization import hp_optimize
from src.cross_validation import cross_val
from src.held_out_validation import held_out_val

def main(args):
    preprocess()
    if args.hp_optimize:
        hp_optimize(args.model)
    if args.cross_val:
        cross_val(args.model)
    else:
        held_out_val(args.model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="basic desc")

    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--hp_optimize", action="store_true", required=False)
    parser.add_argument("--cross_val", action="store_true", required=False)

    args = parser.parse_args()
    main(args)