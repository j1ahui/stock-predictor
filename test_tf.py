import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"

import tensorflow as tf
import numpy as np

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

X = np.random.rand(200, 60, 1).astype("float32")
y = np.random.rand(200, 1).astype("float32")

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(60,1)),
    tf.keras.layers.LSTM(4),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer="adam", loss="mse")

print("starting fit")

model.fit(X, y, epochs=1, verbose=1)

print("finished fit")