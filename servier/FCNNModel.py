
import tensorflow as tf
from keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import Adam
from keras.layers import Input
from keras.models import Model
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
from servier.Model import Model as ServierModel

tf.compat.v1.enable_eager_execution()


def swish(x):
    return (K.sigmoid(x) * x)


get_custom_objects().update({'swish': Activation(swish)})


class FCNNModel(ServierModel):
    def build_model(self, hp, input_shape=(0, 0)):
        i = Input(shape=hp.Choice('s', [input_shape[1]]))
        x = Dense(units=hp.Int('n0', 4, 64),
                  activation=hp.Choice('a0', ["relu", "swish"]))(i)
        x = Dropout(hp.Float('d0', 0, 0.3))(x)
        x = Dense(units=hp.Int('n1', 4, 64),
                  activation=hp.Choice('a1', ["relu", "swish"]))(x)
        x = Dropout(hp.Float('d1', 0, 0.3))(x)
        output = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=i, outputs=output)
        model.compile(loss="binary_crossentropy",
                      optimizer=Adam(learning_rate=0.001),
                      metrics=['accuracy'])
        return model
