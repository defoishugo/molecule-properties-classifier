from sklearn.model_selection import GroupKFold
from keras.callbacks import EarlyStopping
from abc import abstractmethod
import kerastuner as kt
import numpy as np
import pickle
import os

HYPERPARAMETERS_PATH = "__save__/best_hyperparameters.pkl"
PARAMETERS_PATH = "__save__/best_parameters"


class CVTuner(kt.engine.tuner.Tuner):
    def run_trial(self, trial, X, y, splits, batch_size=32,
                  epochs=1, callbacks=None):
        loss = []
        for train_indices, test_indices in splits:
            X_train, X_test = X[train_indices], X[test_indices]
            y_train, y_test = y[train_indices], y[test_indices]
            model = self.hypermodel.build(trial.hyperparameters)
            hist = model.fit(X_train, y_train,
                             validation_data=(X_test, y_test),
                             epochs=epochs,
                             batch_size=batch_size,
                             callbacks=callbacks)
            loss.append([hist.history[k][-1] for k in hist.history])
        loss = np.asarray(loss)
        t = {k: np.mean(loss[:, i]) for i, k in enumerate(hist.history.keys())}
        self.oracle.update_trial(trial.trial_id, t)
        self.save_model(trial.trial_id, model)


class Model():
    def __init__(self, X, y, groups=None, batch_size=1024,
                 tune_epochs=1, train_epochs=1, tune_trials=1):
        self.X = X
        self.y = y
        self.groups = groups
        self.batch_size = batch_size
        self.tune_epochs = tune_epochs
        self.train_epochs = train_epochs
        self.tune_trials = tune_trials

    def tune(self):
        group_kfold = GroupKFold(n_splits=5)
        folds = list(group_kfold.split(self.X, self.y, self.groups))
        shape = self.X.shape
        self.tuner = CVTuner(
            hypermodel=lambda hp: self.build_model(hp, input_shape=shape),
            oracle=kt.oracles.BayesianOptimization(
                objective=kt.Objective('val_accuracy', direction='max'),
                max_trials=self.tune_trials
            ),
            directory=os.path.normpath('C:/'),
            overwrite=True
        )
        self.tuner.search(self.X, self.y, splits=folds, batch_size=1024,
                          epochs=self.tune_epochs,
                          callbacks=[EarlyStopping('val_accuracy',
                                     mode='max', patience=4), EarlyStopping('accuracy',
                                     mode='max', patience=3)])
        self.best_hps = self.tuner.get_best_hyperparameters()[0]
        pickle.dump(self.best_hps, open(HYPERPARAMETERS_PATH, "wb"))

    def import_model(self):
        self.best_hps = pickle.load(open(HYPERPARAMETERS_PATH, "rb"))
        self.model = self.build_model(self.best_hps)
        self.model.summary()
        self.model.load_weights(PARAMETERS_PATH)

    def train(self):
        self.model = self.tuner.hypermodel.build(self.best_hps)
        self.model.fit(self.X, self.y, batch_size=1024,
                       epochs=self.train_epochs)
        self.model.save_weights(PARAMETERS_PATH)

    def evaluate(self):
        results = self.model.evaluate(self.X, self.y, verbose=0)
        print("[test loss, test accuracy]:", results)

    def predict(self, overwrite_X=None):
        prediction = self.model.predict(self.X)
        return prediction[0][0]

    @abstractmethod
    def build_model(self, hp, input_shape=(0, 0)):
        pass
