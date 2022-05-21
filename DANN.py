import numpy as np
import adapt
from adapt.feature_based import DANN
from tensorflow import keras
from keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense, Reshape, Conv2D, MaxPool2D, Flatten, Activation
from tensorflow.keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
import warnings

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")


def get_encoder(input_shape=(32, 16, 3)):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=[3, 3], padding="same", activation="relu", input_shape=input_shape))
    model.add(MaxPool2D(pool_size=[2, 2], strides=2, padding="same"))
    model.add(Conv2D(48, kernel_size=[3, 3], padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=[2, 2], strides=2, padding="same"))
    model.add(Flatten())
    model.add(Activation("sigmoid"))
    return model


def get_task(input_shape=(1536,)):
    model = Sequential()
    model.add(Dense(1024, activation="relu", input_shape=input_shape))
    model.add(Dense(512, activation="relu"))
    model.add(Dense(128, activation="relu"))
    model.add(Dense(34, activation="softmax"))
    return model


def get_discriminator(input_shape=(1536,)):
    model = Sequential()
    model.add(Dense(1024, activation="relu", input_shape=input_shape))
    model.add(Dense(512, activation="relu"))
    model.add(Dense(128, activation="relu"))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    return model


save_check_pt = "./checkpoints_DANN/"

# Sample to load model
# Dummy inputs to load model
xs = np.zeros((8446, 32, 16, 3))
ys = np.zeros((8446, 34))
xt = np.zeros((762, 32, 16, 3))
yt = np.zeros((762, 34))
dann_model = DANN(
    get_encoder(),
    get_task(),
    get_discriminator(),
    lambda_=0.1,
    optimizer=Adam(0.0001),
    loss="CategoricalCrossentropy",
    metrics=["acc"],
    random_state=0,
)
dann_model.fit(xs, ys, xt, yt, epochs=0, verbose=1, batch_size=32)  # NO fit happen, just dummy step to load model
dann_scenario_dict = {
    "Dark night": "db",
    "Rainy, snow, or fog": "weather",
    "Far or near to the camera": "fn",
    "Other challenging scenarios": "challenge",
}
