import numpy as np
from tensorflow import keras

class Discriminator:

    def __init__(self):
        self.model = keras.models.load_model("discriminator.h5")

    def estimate(self, image):
        image = np.array([image])
        estimate = self.model.predict(image)
        return estimate