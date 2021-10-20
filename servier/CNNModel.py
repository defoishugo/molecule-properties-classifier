
from keras.layers import Dense, Flatten, Conv2D, Dropout
from keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from servier.Model import Model as ServierModel
from servier.dataset import SMILES_CHARS, VEC_SIZE


class CNNModel(ServierModel):
    def build_model(self, hp, input_shape=(0, 0)):
        model = Sequential()
        model.add(Conv2D(hp.Int("t1", 20, 96),
                         kernel_size=(hp.Int("s1", 4, 20), len(SMILES_CHARS)),
                         activation='relu',
                         input_shape=(VEC_SIZE, len(SMILES_CHARS), 1)))
        model.add(Conv2D(hp.Int("t2", 20, 96),
                  kernel_size=(hp.Int("s2", 1, 5), 1),
                  activation='relu'))
        model.add(Flatten())
        model.add(Dense(hp.Int("s3", 4, 50), activation="relu"))
        model.add(Dropout(hp.Float("d", 0, 0.3)))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss="binary_crossentropy",
                      optimizer=Adam(lr=0.00025),
                      metrics=['accuracy'])
        return model
