
from keras.layers import Dense, Flatten, Conv2D
from keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from servier.Model import Model as ServierModel
from servier.dataset import SMILES_CHARS, VEC_SIZE


class CNNModel(ServierModel):
    def build_model(self, hp, input_shape=(0, 0)):
        t = hp.Int("t1", 20, 196)
        model = Sequential()
        model.add(Conv2D(192, kernel_size=(10, len(SMILES_CHARS)),
                         activation='relu',
                         input_shape=(VEC_SIZE, len(SMILES_CHARS), 1)))
        model.add(Conv2D(192, kernel_size=(5, 1), activation='relu'))
        model.add(Conv2D(192, kernel_size=(3, 1), activation='relu'))
        model.add(Flatten())
        model.add(Dense(5, activation="relu"))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss="binary_crossentropy",
                      optimizer=Adam(lr=0.00025),
                      metrics=['accuracy'])
        return model
