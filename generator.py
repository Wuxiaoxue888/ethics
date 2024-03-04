import numpy as np
import pandas as pd
import random

from population import Population
from discriminator import discriminator

# RETRAIN defines after how many generations the discriminator models are retrained.
RETRAIN_GENERATIONS = 7
RETRAIN_FITNESS = 0.85

ALMOST_REAL_IMAGES = pd.read_csv("almost_real_fake_images.csv").drop("label", axis=1).to_numpy().tolist()

def main():
    generated_images = []
    population = Population(size=512, doping_size=512)
    for generation_number in range(1, 31):

        if generation_number % 10 == 0:
            print(f"-----------------Generation {generation_number}-----------------------")

        population.dope(20)

        population.order_by_fitness()

        if (
                generation_number % RETRAIN_GENERATIONS == 0
                and len(generated_images) > 2000
                # small optimization - python if statements are executed in order they are given. Meaning that if the
                # first two are not BOTH True the third one won't be evaluated and the average_fitness method executed.
                and population.average_fitness() > RETRAIN_FITNESS
        ):
            generated_training_images = generated_images[:500]
            discriminator.train(
                generated_training_images + ALMOST_REAL_IMAGES[:len(generated_training_images)]
            )
            generated_images = []

        population.create_offsprings(
            256, 256, selection_function="roulette_wheel"
        )

        for image in population.images:
            if not image.doped:
                generated_images.append(image.chromosome)
        generated_images = generated_images[-25_000:]
        random.shuffle(generated_images)
        random.shuffle(ALMOST_REAL_IMAGES)

    population.print()

    # Save the generated images
    population.order_by_fitness()
    np.save(
        "generated_images",
        np.array(list(map(lambda image: image.chromosome, population.images))),
    )


if __name__ == "__main__":
    main()
