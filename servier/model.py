import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)
import kerastuner as kt
import tensorflow as tf
import numpy as np
import pickle
import os
from sklearn.model_selection import GroupKFold
from keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.layers import Input
from keras.models import Model
from servier.dataset import filter_features
from servier.feature_extractor import fingerprint_features

tf.compat.v1.enable_eager_execution()
HYPERPARAMETERS_PATH = "__save__/best_hyperparameters.pkl"
PARAMETERS_PATH = "__save__/best_parameters"


# For adding new activation function
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects

def swish(x):
    return (K.sigmoid(x) * x)

get_custom_objects().update({'swish': Activation(swish)})

class CVTuner(kt.engine.tuner.Tuner):
    def run_trial(self, trial, X, y, splits, batch_size=32, epochs=1, callbacks=None):
        val_losses = []
        for train_indices, test_indices in splits:
            X_train, X_test = X[train_indices], X[test_indices]
            y_train, y_test = y[train_indices], y[test_indices]
            model = self.hypermodel.build(trial.hyperparameters)
            hist = model.fit(X_train,y_train,
                      validation_data=(X_test,y_test),
                      epochs=epochs,
                        batch_size=batch_size,
                      callbacks=callbacks)
            val_losses.append([hist.history[k][-1] for k in hist.history])
        val_losses = np.asarray(val_losses)
        self.oracle.update_trial(trial.trial_id, {k:np.mean(val_losses[:,i]) for i,k in enumerate(hist.history.keys())})
        self.save_model(trial.trial_id, model)


def build_model(hp, input_shape=0):
    input = Input(shape=(hp.Int('nb_cols', input_shape, input_shape),))
    x = Dense(hp.Int('num_units_0', 4, 64), activation=hp.Choice('act_0', ["relu", "swish"]))(input)
    x = Dropout(hp.Float('dropout_0', 0.0, 0.3))(x)
    x = Dense(hp.Int('num_units_1', 4, 64), activation=hp.Choice('act_1', ["relu", "swish"]))(x)
    x = Dropout(hp.Float('dropout_1', 0.0, 0.3))(x)
    output = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=input, outputs=output)
    model.compile(loss="binary_crossentropy",
                  optimizer=Adam(learning_rate=0.001),
                  metrics=['accuracy'])
    return model


def tune_model(X_train, y_train, groups):
    group_kfold = GroupKFold(n_splits=4)
    folds = list(group_kfold.split(X_train, y_train, groups))
    tuner = CVTuner(
        hypermodel=lambda hp: build_model(hp, input_shape=X_train.shape[1]),
        oracle=kt.oracles.BayesianOptimization(
            objective=kt.Objective('val_accuracy', direction='max'),
            max_trials=10
        ),
        directory=os.path.normpath('C:/'),
        overwrite=True
    )
    tuner.search(X_train, y_train, splits=folds, batch_size=128, epochs=50,
                 callbacks=[EarlyStopping('val_accuracy', mode='max', patience=4)])
    best_hps = tuner.get_best_hyperparameters()[0]
    pickle.dump(best_hps, open(HYPERPARAMETERS_PATH, "wb"))
    return tuner, best_hps


def train_model(X_train, y_train, tuner, best_hps):
    hypermodel = tuner.hypermodel.build(best_hps)
    hypermodel.fit(X_train, y_train, batch_size=128, epochs=300)
    hypermodel.save_weights(PARAMETERS_PATH)
    return hypermodel


def import_model():
    best_hps = pickle.load(open(HYPERPARAMETERS_PATH, "rb"))
    model = build_model(best_hps)
    model.load_weights(PARAMETERS_PATH)
    return model


def evaluate_model(model, X, y):
    eval_result = model.evaluate(X, y)
    print("[test loss, test accuracy]:", eval_result)


def predict_model(model, smiles, radius, size):
    ecfp = fingerprint_features(str(smiles), radius, size)
    X = filter_features([ecfp], load_features=True)
    prediction = model.predict(X)
    return prediction[0][0]