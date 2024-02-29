import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Do not print tf info in terminal - ! needs to be above tf import
import numpy as np
from tensorflow import keras
import tensorflow as tf
import pandas as pd
tf.get_logger().setLevel('ERROR') # Do not print tf info to terminal

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

REAL_IMAGES = pd.read_csv("real_images.csv") / 255.
REAL_IMAGES["label"] = 1.0

class Discriminator:

    def __init__(self):
        self.model = keras.models.load_model("discriminator.h5")
        self.saved_generated_images = []

    def estimate(self, images):
        estimates = self.model.predict(images, verbose=None).flatten()
        return estimates
    
    def train(self, generated_images):
        generated_images = pd.DataFrame(generated_images)
        generated_images.columns = generated_images.columns.astype(str) # Turn the column name type to str to match with the df for the real_images
        generated_images["label"] = 0.0 # Give labels to generated images
        real_images = REAL_IMAGES[len(self.saved_generated_images):len(self.saved_generated_images)+len(generated_images)] # get subset of real images to train on
        training_images = pd.concat([real_images, generated_images], ignore_index=True)  # Combine the real and fake images
        training_images = training_images.sample(frac=1).reset_index(drop=True) # Shuffle
        features = training_images.drop("label", axis=1)
        labels = training_images["label"]
        self.model.fit(features, labels, epochs=1) # Train model again
        self.saved_generated_images.append(generated_images) # Save generated images

discriminator = Discriminator()