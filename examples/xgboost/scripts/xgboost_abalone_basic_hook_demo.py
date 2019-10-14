import argparse
import json
import os
import pickle
import random
import tempfile
import urllib.request

import xgboost
from tornasole import SaveConfig
from tornasole.xgboost import TornasoleHook


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--eta", type=float, default=0.2)
    parser.add_argument("--gamma", type=int, default=4)
    parser.add_argument("--min_child_weight", type=int, default=6)
    parser.add_argument("--subsample", type=float, default=0.7)
    parser.add_argument("--silent", type=int, default=0)
    parser.add_argument("--objective", type=str, default="reg:squarederror")
    parser.add_argument("--num_round", type=int, default=50)
    parser.add_argument("--tornasole_path", type=str, default=None)
    parser.add_argument("--tornasole_frequency", type=int, default=1)
    parser.add_argument(
        "--output_uri",
        type=str,
        default="/opt/ml/output/tensors",
        help="S3 URI of the bucket where tensor data will be stored.",
    )

    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--validation", type=str, default=os.environ.get("SM_CHANNEL_VALIDATION"))

    args = parser.parse_args()

    return args


def load_abalone(train_split=0.8, seed=42):

    if not (0 < train_split <= 1):
        raise ValueError("'train_split' must be between 0 and 1.")

    url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/abalone"

    response = urllib.request.urlopen(url).read().decode("utf-8")
    lines = response.strip().split("\n")
    n = sum(1 for line in lines)
    indices = list(range(n))
    random.seed(seed)
    random.shuffle(indices)
    train_indices = set(indices[: int(n * 0.8)])

    with tempfile.NamedTemporaryFile(mode="w", delete=False) as train_file:
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as valid_file:
            for idx, line in enumerate(lines):
                if idx in train_indices:
                    train_file.write(line + "\n")
                else:
                    valid_file.write(line + "\n")

    return train_file.name, valid_file.name


def create_tornasole_hook(out_dir, train_data=None, validation_data=None, frequency=1):

    save_config = SaveConfig(save_interval=frequency)
    hook = TornasoleHook(
        out_dir=out_dir,
        save_config=save_config,
        train_data=train_data,
        validation_data=validation_data,
    )

    return hook


def main():

    args = parse_args()

    if args.train and args.validation:
        train, validation = args.train, args.validation
    else:
        train, validation = load_abalone()

    dtrain = xgboost.DMatrix(train)
    dval = xgboost.DMatrix(validation)

    watchlist = [(dtrain, "train"), (dval, "validation")]

    params = {
        "max_depth": args.max_depth,
        "eta": args.eta,
        "gamma": args.gamma,
        "min_child_weight": args.min_child_weight,
        "subsample": args.subsample,
        "silent": args.silent,
        "objective": args.objective,
    }

    # The output_uri is a the URI for the s3 bucket where the metrics will be
    # saved.
    output_uri = args.tornasole_path if args.tornasole_path is not None else args.output_uri

    hook = create_tornasole_hook(
        out_dir=output_uri, frequency=args.tornasole_frequency, train_data=dtrain
    )

    bst = xgboost.train(
        params=params,
        dtrain=dtrain,
        evals=watchlist,
        num_boost_round=args.num_round,
        callbacks=[hook],
    )


if __name__ == "__main__":

    main()
