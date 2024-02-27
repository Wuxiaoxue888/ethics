import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Do not print tf info in terminal - ! needs to be above tf import
import numpy as np
from tensorflow import keras
import tensorflow as tf
tf.get_logger().setLevel('ERROR') # Do not print tf info to terminal

class Discriminator:

    def __init__(self):
        self.model = keras.models.load_model("discriminator.h5")

    def estimate(self, images):
        estimates = self.model.predict(images, verbose=None).flatten()
        return estimates

discriminator = Discriminator()