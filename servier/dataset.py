import pandas as pd
import numpy as np
import pickle as pk
from rdkit import Chem
from sklearn.model_selection import train_test_split
from servier.feature_extractor import fingerprint_features


TEST_RATIO = 0.2
RANDOM_SMILES = 1
USEFUL_FEATURES_PATH = "__save__/useful_features.pkl"


def randomize_smiles(smiles):
    m = Chem.MolFromSmiles(smiles)
    ans = list(range(m.GetNumAtoms()))
    np.random.shuffle(ans)
    nm = Chem.RenumberAtoms(m, ans)
    return Chem.MolToSmiles(nm, canonical=False, isomericSmiles=False)


def upsample_dataset(X, y):
    X_upsampled = X.copy()
    y_upsampled = y.copy()
    for smiles, y_score in zip(X, y):
        for i in range(RANDOM_SMILES):
            random_smiles = randomize_smiles(smiles)
            X_upsampled.append(random_smiles)
            y_upsampled.append(y_score)
    unique_smiles, unique_indexes = np.unique(X_upsampled, return_index=True)
    X_upsampled = np.array(X_upsampled)[unique_indexes]
    y_upsampled = np.array(y_upsampled)[unique_indexes]
    return X_upsampled, y_upsampled


def balance_dataset(X, y):
    minority = 0 if sum(y) > int(len(X) / 2) else 1
    keep_elements = len(y[y == minority])
    X_balanced = np.append(X[y == minority][:keep_elements],
                           X[y != minority][:keep_elements])
    y_balanced = np.append(y[y == minority][:keep_elements],
                           y[y != minority][:keep_elements])
    return X_balanced, y_balanced


def load_existing_dataset():
    with open('../vectors_save.pkl', 'rb') as f:
        (X_train, y_train, X_test, y_test) = pk.load(f)
    print("Dataset loaded from existing file.")
    return X_train, X_test, y_train, y_test


def save_dataset(X_train, y_train, X_test, y_test):
    with open('../vectors_save.pkl', 'wb') as f:
        data = X_train, y_train, X_test, y_test
        pk.dump(data, f)
    print("Dataset saved.")


def filter_features(X_train, X_test=None, load_features=False):
    if load_features:
        useful_features = pk.load(open(USEFUL_FEATURES_PATH, "rb"))
    else:
        unuseful_features = np.bitwise_and.reduce(X_train) == X_train[0]
        useful_features = [i for i, k in enumerate(unuseful_features) if k]
        pk.dump(useful_features, open(USEFUL_FEATURES_PATH, "wb"))
    X_train = np.delete(X_train, useful_features, axis=1)
    if X_test:
        X_test = np.delete(X_test, useful_features, axis=1)
        return X_train, X_test
    else:
        return X_train


def keep_unique_samples(X, y):
    unique_values, unique_indexes = np.unique(X, axis=0, return_index=True)
    X = np.array(X)[unique_indexes]
    y = np.array(y)[unique_indexes]
    return X, y


def build_training_dataset(radius, size, save=False,
                           use_save=False, path="ds.csv"):
    if use_save:
        return load_existing_dataset()

    print("Loading and preprocessing the dataset...")

    raw_dataset = pd.read_csv(path, sep=",")
    X_train, X_test, y_train, y_test = train_test_split(raw_dataset["smiles"],
                                                        raw_dataset["P1"],
                                                        test_size=TEST_RATIO,
                                                        random_state=0,
                                                        shuffle=True)
    X_train, y_train = upsample_dataset(X_train.tolist(), y_train.tolist())
    X_test, y_test = upsample_dataset(X_test.tolist(), y_test.tolist())
    X_train, y_train = keep_unique_samples(X_train, y_train)
    X_train, y_train = balance_dataset(X_train, y_train)
    X_train = [list(fingerprint_features(x, radius, size)) for x in X_train]
    X_test = [list(fingerprint_features(x, radius, size)) for x in X_test]
    X_train, X_test = filter_features(X_train, X_test)
    print("Dataset built.")

    X_train, y_train, X_test, y_test = [np.array(arr) for arr in [
                                            X_train,
                                            y_train,
                                            X_test,
                                            y_test
                                        ]]

    if save:
        save_dataset(X_train, y_train, X_test, y_test)

    return X_train, y_train, X_test, y_test


def build_evaluation_dataset(radius, size, path="../dataset_single.csv"):
    print("Loading and preprocessing the dataset...")
    raw_dataset = pd.read_csv(path, sep=",")
    X = raw_dataset["smiles"].to_numpy()
    y = raw_dataset["P1"].to_numpy()
    X = [list(fingerprint_features(x, radius, size)) for x in X]
    X = filter_features(X, load_features=True)
    print("Dataset built.")
    return X, y
