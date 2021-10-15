import kerastuner as kt
import tensorflow as tf
import pickle as pk
import os
from dataset import build_dataset
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

tf.enable_eager_execution()


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
        self.oracle.update_trial(trial.trial_id, {k: loss[i] for i, k in enumerate(hist.history.keys())})
        self.save_model(trial.trial_id, model)


def build_model(hp, input_shape):
    input = Input(shape=input_shape)
    x = Dense(hp.Int('num_units_0', 4, 64), activation="relu")(input)
    x = Dropout(hp.Float('dropout', 0.0, 0.3))(x)
    output = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=input, outputs=output)
    model.compile(loss="binary_crossentropy",
                  optimizer=Adam(learning_rate=0.001),
                  metrics=['accuracy'])
    return model


def tune(X_train, y_train, X_test, y_test):
    tuner = Tuner(
        hypermodel=lambda hp: build_model(hp, (X_train.shape[1],)),
        oracle=kt.oracles.BayesianOptimization(
            objective=kt.Objective('val_acc', direction='max'),
            max_trials=10
        ),
        directory=os.path.normpath('C:/')
    )
    tuner.search(X_train, y_train, X_test=X_test, y_test=y_test, batch_size=128, epochs=30, callbacks=[EarlyStopping('val_acc', mode='max', patience=4)])
    best_hps = tuner.get_best_hyperparameters()[0]
    return tuner, best_hps


def train(X_train, y_train, tuner, best_hp):
    hypermodel = tuner.hypermodel.build(best_hps)
    hypermodel.fit(X_train, y_train, batch_size=128, epochs=200)
    with open('model.pkl', 'wb') as f:
        pk.dump(hypermodel, f)
        f.close()
    return hypermodel


X_train, X_test, y_train, y_test = build_dataset(2, 2048, use_save=True)
tuner, best_hps = tune(X_train, y_train, X_test, y_test)
model = train(X_train, y_train, tuner, best_hps)
eval_result = model.evaluate(X_test, y_test)
print("[test loss, test accuracy]:", eval_result)
