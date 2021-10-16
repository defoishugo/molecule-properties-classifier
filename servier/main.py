import argparse
import sys
import os
from servier.dataset import build_training_dataset, build_evaluation_dataset
from servier.model import tune_model, train_model, import_model, evaluate_model
import pathlib


def train(input_table):
    if not input_table:
        print("Error: the input table should be "
              "specified with --input-table parameter.")
        return 1
    training = build_training_dataset(2, 2048, path=input_table)
    X_train, y_train, X_valid, y_valid = training
    tuner, best_hps = tune_model(X_train, y_train, X_valid, y_valid)
    train_model(X_train, y_train, tuner, best_hps)
    return 0


def evaluate(input_table):
    if not input_table:
        print("Error: the input table should be "
              "specified with --input-table parameter.")
        return 1
    X, y = build_evaluation_dataset(2, 2048, path=input_table)
    model = import_model()
    evaluate_model(model, X, y)
    return 0


def main():
    desc = 'Classify molecule properties using neural networks'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('command', type=str,
                        help='script to execute',
                        choices=['train', 'evaluate', 'predict'])
    parser.add_argument('--input-table', type=pathlib.Path,
                        help='path to the input')
    parser.add_argument('--model', type=int,
                        help='specify a model number')

    args = parser.parse_args()
    if args.command == "train":
        if not os.path.exists("__save__"):
            os.mkdir("__save__")
        sys.exit(train(args.input_table))
    elif args.command == "evaluate":
        sys.exit(evaluate(args.input_table))


if __name__ == "__main__":
    main()
