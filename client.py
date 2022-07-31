import os

import flwr as fl
import tensorflow as tf

import pandas as pd
import numpy as np
# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# Load model and data 
df = pd.read_csv('creditcard.csv')
x = np.asanyarray(df.drop('Class',1))
y = np.asanyarray(df['Class'])
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.7)

from imblearn.over_sampling import SMOTE
sm = SMOTE()

x_train,y_train = sm.fit_resample(x_train,y_train)


x_train = x_train.astype(np.float32)/x_train.max()
y_train = y_train.astype(np.int32)/y_train.max()
x_test = x_test.astype(np.float32)/x_test.max()
y_test = y_test.astype(np.int32)/y_test.max()

n_input = x_train.shape[1]
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(65, input_shape=(n_input,), kernel_initializer='he_normal', activation='relu'))
model.add(tf.keras.layers.Dense(1, kernel_initializer='he_normal', activation='sigmoid'))
model.compile(tf.keras.optimizers.Adam(lr=0.01), loss='binary_crossentropy', metrics=['accuracy'])




# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        r = model.fit(x_train, y_train, epochs=1, validation_data=(x_test, y_test), verbose=0)
        hist = r.history
        print("Fit history : " ,hist)
        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
        print("Eval accuracy : ", accuracy)
        return loss, len(x_test), {"accuracy": accuracy}

# Start Flower client
fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=FlowerClient())