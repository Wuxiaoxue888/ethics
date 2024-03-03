import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = (
    "3"  # Do not print tf info in terminal - ! needs to be above tf import
)
import numpy as np
from tensorflow import keras
import tensorflow as tf
import pandas as pd
import random

tf.get_logger().setLevel("ERROR")  # Do not print tf info to terminal

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

REAL_IMAGES = pd.read_csv("real_images.csv") / 255.0
REAL_IMAGES.columns = REAL_IMAGES.columns.astype(
    int
)  # Change column name types to match with the generated_images DataFrame
REAL_IMAGES["label"] = 1.0


class Discriminator:

    def __init__(self):
        self.models = [
            keras.models.load_model(f"discriminator_models/discriminator_{i + 1}.keras")
            for i in range(5)
        ]

    def estimate(self, images):
        model = random.choice(self.models)
        estimates = model.predict(images, verbose=None).flatten()
        return estimates

    def train(self, generated_images):
        generated_images = pd.DataFrame(generated_images)
        generated_images["label"] = 0.0

        # Random subset of real images to train on
        # real_images = REAL_IMAGES.sample(frac=1).reset_index(drop=True)[: len(generated_images)]

        # Combine the real and fake images
        # combined_images = pd.concat([real_images, generated_images], ignore_index=True)
        # combined_images = combined_images.sample(frac=1).reset_index(drop=True)

        features = generated_images.drop("label", axis=1)
        labels = generated_images["label"]

        training_size = int(len(generated_images) * 0.8)

        X_test, X_train = features[training_size:], features[:training_size]
        y_test, y_train = labels[training_size:], labels[:training_size]

        # Check the current accuracy of the model. if its high enough we should not train again
        _, accuracy = random.choice(self.models).evaluate(X_test, y_test, verbose=None)
        if accuracy > 0.9:
            print(f"No training, accuracy is {accuracy}")
            return

        # Train models again
        for i, model in enumerate(self.models):
            history = model.fit(
                X_train,
                y_train,
                epochs=1,
                validation_data=(X_test, y_test),
                verbose=None,
            )
            print(
                "Training again - accuracy",
                history.history["accuracy"][0],
                "Validation accuracy",
                history.history["val_accuracy"][0],
            )


discriminator = Discriminator()
