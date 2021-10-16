import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)
import kerastuner as kt
import tensorflow as tf
import pickle
import os
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from servier.dataset import filter_features
from servier.feature_extractor import fingerprint_features

tf.compat.v1.enable_eager_execution()
HYPERPARAMETERS_PATH = "__save__/best_hyperparameters.pkl"
PARAMETERS_PATH = "__save__/best_parameters"


class Tuner(kt.engine.tuner.Tuner):
    def run_trial(self, trial, X_train, y_train, X_test=[], y_test=[],
                  batch_size=1, epochs=1, callbacks=None):
        model = self.hypermodel.build(trial.hyperparameters)
        hist = model.fit(X_train, y_train,
                         validation_data=(X_test, y_test),
                         epochs=epochs,
                         batch_size=batch_size,
                         callbacks=callbacks)
        loss = [hist.history[k][-1] for k in hist.history]
        loss = {k: loss[i] for i, k in enumerate(hist.history.keys())}
        self.oracle.update_trial(trial.trial_id, loss)
        self.save_model(trial.trial_id, model)


def build_model(hp, input_shape=0):
    input = Input(shape=(hp.Int('nb_cols', input_shape, input_shape),))
    x = Dense(hp.Int('num_units_0', 4, 64), activation="relu")(input)
    x = Dropout(hp.Float('dropout', 0.0, 0.3))(x)
    output = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=input, outputs=output)
    model.compile(loss="binary_crossentropy",
                  optimizer=Adam(learning_rate=0.001),
                  metrics=['accuracy'])
    return model


def tune_model(X_train, y_train, X_test, y_test):
    tuner = Tuner(
        hypermodel=lambda hp: build_model(hp, input_shape=X_train.shape[1]),
        oracle=kt.oracles.BayesianOptimization(
            objective=kt.Objective('val_acc', direction='max'),
            max_trials=10
        ),
        directory=os.path.normpath('C:/'),
        overwrite=True
    )
    tuner.search(X_train, y_train, X_test=X_test,
                 y_test=y_test, batch_size=128, epochs=1,
                 callbacks=[EarlyStopping('val_acc', mode='max', patience=4)])
    best_hps = tuner.get_best_hyperparameters()[0]
    pickle.dump(best_hps, open(HYPERPARAMETERS_PATH, "wb"))
    return tuner, best_hps


def train_model(X_train, y_train, tuner, best_hps):
    hypermodel = tuner.hypermodel.build(best_hps)
    hypermodel.fit(X_train, y_train, batch_size=128, epochs=1)
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