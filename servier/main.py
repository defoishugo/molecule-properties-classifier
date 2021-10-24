import argparse
import sys
import os
import servier.dataset as ds
import pathlib
import servier.api as api
from servier.FCNNModel import FCNNModel
from servier.CNNModel import CNNModel


def train(input_table, model):
    if not input_table:
        print("Error: the input table should be "
              "specified with --input-table parameter.")
        return 1
    if not model:
        print("Error: the model id should be "
              "specified with --model parameter.")
        return 1
    X_train, y_train, groups = ds.build_training_dataset(2, 2048,
                                                         path=input_table,
                                                         model=model)
    if model == 1:
        model = FCNNModel(X_train, y_train, groups=groups, batch_size=2048,
                          tune_epochs=75, train_epochs=150, tune_trials=30)
    else:
        model = CNNModel(X_train, y_train, groups=groups, batch_size=2048,
                         tune_epochs=75, train_epochs=150, tune_trials=30)
    model.tune()
    model.train()
    return 0


def evaluate(input_table, model):
    if not input_table:
        print("Error: the input table should be "
              "specified with --input-table parameter.")
        return 1
    if not model:
        print("Error: the model id should be "
              "specified with --model parameter.")
        return 1
    X, y = ds.build_evaluation_dataset(2, 2048, path=input_table, model=model)
    if model == 1:
        model = FCNNModel(X, y)
    else:
        model = CNNModel(X, y)
    model.import_model()
    model.evaluate()
    return 0


def predict(smiles, model):
    if not smiles:
        print("Error: the smiles should be "
              "specified with --smiles parameter.")
        return 1
    if not model:
        print("Error: the model id should be "
              "specified with --model parameter.")
    X = ds.build_prediction_dataset(smiles, 2, 2048, model=model)
    if model == 1:
        model = FCNNModel(X, [])
    else:
        model = CNNModel(X, [])
    model.import_model()
    pred = model.predict()
    print(f"The smiles {smiles} gives a probability of {pred}.")
    return 0


def serve(model):
    model = FCNNModel([], [])
    model.import_model()
    api.serve(model)


def cli():
    desc = 'Classify molecule properties using neural networks'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('command', type=str,
                        help='script to execute',
                        choices=['train', 'evaluate', 'predict', 'serve'])
    parser.add_argument('--input-table', type=pathlib.Path,
                        help='path to the input')
    parser.add_argument('--smiles',
                        help='smiles to predict')
    parser.add_argument('--model', type=int,
                        help='specify a model number',
                        choices=[1, 2])

    args = parser.parse_args()
    if args.command == "train":
        if not os.path.exists("__save__"):
            os.mkdir("__save__")
        sys.exit(train(args.input_table, args.model))
    elif args.command == "evaluate":
        sys.exit(evaluate(args.input_table, args.model))
    elif args.command == "predict":
        sys.exit(predict(args.smiles, args.model))
    elif args.command == "serve":
        sys.exit(serve(args.model))


def docker():
    if 'COMMAND' not in os.environ:
        print("Please set the environment variable COMMAND.")
        sys.exit(1)
    if os.environ["COMMAND"] == "train":
        sys.exit(train(f'/root/data/{os.environ["CSV"]}', int(os.environ["MODEL"])))
    elif os.environ["COMMAND"] == "evaluate":
        sys.exit(evaluate(f'/root/data/{os.environ["CSV"]}', int(os.environ["MODEL"])))
    elif os.environ["COMMAND"] == "predict":
        sys.exit(predict(os.environ["SMILES"], int(os.environ["MODEL"])))
    elif os.environ["COMMAND"] == "serve":
        sys.exit(serve(int(os.environ["MODEL"])))
    else:
        print("Unknown COMMAND.")
        sys.exit(1)


def main():
    if 'IS_DOCKER_EXEC' in os.environ:
        return docker()
    else:
        return cli()


if __name__ == "__main__":
    main()
