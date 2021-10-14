import pandas as pd
import numpy as np
from rdkit import Chem
from sklearn.model_selection import train_test_split
from feature_extractor import fingerprint_features


TEST_RATIO = 0.2
RANDOM_SMILES = 10


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


def build_dataset(radius, size):
    raw_dataset = pd.read_csv("../dataset_single.csv", sep=",")
    X_train, X_test, y_train, y_test = train_test_split(raw_dataset["smiles"],
                                                        raw_dataset["P1"],
                                                        test_size=TEST_RATIO,
                                                        random_state=0,
                                                        shuffle=True)

    X_train, y_train = upsample_dataset(X_train.tolist(), y_train.tolist())
    X_train, y_train = balance_dataset(X_train, y_train)
    X_train = [list(fingerprint_features(x, radius, size)) for x in X_train]
    X_test = [list(fingerprint_features(x, radius, size)) for x in X_test]

    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = build_dataset(2, 2048)
print(X_train)
