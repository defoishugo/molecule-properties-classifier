import pandas as pd
import numpy as np
import pickle as pk
from rdkit import Chem
from feature_extractor import fingerprint_features

TEST_RATIO = 0.2
RANDOM_SMILES = 60
USEFUL_FEATURES_PATH = "__save__/useful_features.pkl"
SMILES_CHARS = [' ', '#', '%', '(', ')', '+', '-', '.', '/',
                '0', '1', '2', '3', '4', '5', '6', '7', '8',
                '9', '=', '@', 'A', 'B', 'C', 'F', 'H', 'I',
                'K', 'L', 'M', 'N', 'O', 'P', 'R', 'S', 'T',
                'V', 'X', 'Z', '[', '\\', ']', 'a', 'b', 'c',
                'e', 'g', 'i', 'l', 'n', 'o', 'p', 'r', 's',
                't', 'u', '!', 'E']


def randomize_smiles(smiles):
    m = Chem.MolFromSmiles(smiles)
    ans = list(range(m.GetNumAtoms()))
    np.random.shuffle(ans)
    nm = Chem.RenumberAtoms(m, ans)
    return Chem.MolToSmiles(nm, canonical=False, isomericSmiles=False)


def upsample_dataset(X, y, groups):
    X_upsampled = X.copy()
    y_upsampled = y.copy()
    for j, (smiles, y_score) in enumerate(zip(X, y)):
        for i in range(RANDOM_SMILES):
            random_smiles = randomize_smiles(smiles)
            groups.append(groups[j])
            X_upsampled.append(random_smiles)
            y_upsampled.append(y_score)
    unique_smiles, unique_indexes = np.unique(X_upsampled, return_index=True)
    X_upsampled = np.array(X_upsampled)[unique_indexes]
    y_upsampled = np.array(y_upsampled)[unique_indexes]
    return X_upsampled, y_upsampled, groups


def balance_dataset(X, y, groups):
    minority = 0 if sum(y) > int(len(X) / 2) else 1
    keep_elements = len(y[y == minority])
    minority_indices = np.random.randint(len(y[y == minority]),
                                         size=keep_elements)
    majority_indices = np.random.randint(len(y[y != minority]),
                                         size=keep_elements)
    groups = np.append(groups[y == minority][minority_indices],
                       groups[y != minority][majority_indices])
    X_balanced = np.append(X[y == minority][minority_indices],
                           X[y != minority][majority_indices])
    y_balanced = np.append(y[y == minority][minority_indices],
                           y[y != minority][majority_indices])
    return X_balanced, y_balanced, groups


def load_existing_dataset():
    with open('__save__/vectors_save.pkl', 'rb') as f:
        (X_train, y_train, groups) = pk.load(f)
    print("Dataset loaded from existing file.")
    return X_train, y_train, groups


def save_dataset(X_train, y_train, groups):
    with open('__save__/vectors_save.pkl', 'wb') as f:
        data = X_train, y_train, groups
        pk.dump(data, f)
    print("Dataset saved.")


def filter_features(X_train, load_features=False):
    if load_features:
        useful_features = pk.load(open(USEFUL_FEATURES_PATH, "rb"))
    else:
        unuseful_features = np.bitwise_and.reduce(X_train) == X_train[0]
        useful_features = [i for i, k in enumerate(unuseful_features) if k]
        pk.dump(useful_features, open(USEFUL_FEATURES_PATH, "wb"))
    X_train = np.delete(X_train, useful_features, axis=1)
    return X_train


def keep_unique_samples(X, y, groups):
    unique_values, unique_indexes = np.unique(X, axis=0, return_index=True)
    X = np.array(X)[unique_indexes]
    y = np.array(y)[unique_indexes]
    groups = np.array(groups)[unique_indexes]
    return X, y, groups


def smiles2vec(X, vec_size):
    char_to_int = {v: i for i, v in enumerate(SMILES_CHARS)}
    one_hot = np.zeros((X.shape[0], vec_size,
                       len(char_to_int)), dtype=np.int8)
    for i, smile in enumerate(X):
        one_hot[i, 0, char_to_int["!"]] = 1
        for j, c in enumerate(smile):
            one_hot[i, j+1, char_to_int[c]] = 1
        one_hot[i, len(smile)+1:, char_to_int["E"]] = 1
    return one_hot


def build_training_dataset(radius, size, save=False, use_save=False,
                           path="ds.csv", model=1):
    if use_save:
        return load_existing_dataset()

    print("Loading and preprocessing the dataset...")

    raw_dataset = pd.read_csv(path, sep=",")
    X_train = raw_dataset["smiles"]
    y_train = raw_dataset["P1"]
    groups = list(range(len(X_train)))
    X_train, y_train, groups = upsample_dataset(X_train.tolist(),
                                                y_train.tolist(),
                                                groups)
    X_train, y_train, groups = keep_unique_samples(X_train, y_train, groups)
    X_train, y_train, groups = balance_dataset(X_train, y_train,
                                               groups)
    if model == 1:
        X_train = [list(fingerprint_features(x, radius, size)) for x in X_train]
        X_train = filter_features(X_train)
    else:
        vec_size = len(max(X_train, key=len)) + 1
        X_train = smiles2vec(X_train, vec_size)

    print("Dataset built.")

    X_train, y_train, groups = [np.array(arr) for arr in [
                                            X_train,
                                            y_train,
                                            groups
                                        ]]

    if save:
        save_dataset(X_train, y_train, groups)

    return X_train, y_train, groups


def build_evaluation_dataset(radius, size, path="../dataset_single.csv"):
    print("Loading and preprocessing the dataset...")
    raw_dataset = pd.read_csv(path, sep=",")
    X = raw_dataset["smiles"].to_numpy()
    y = raw_dataset["P1"].to_numpy()
    X = [list(fingerprint_features(x, radius, size)) for x in X]
    X = filter_features(X, load_features=True)
    print("Dataset built.")
    return X, y
