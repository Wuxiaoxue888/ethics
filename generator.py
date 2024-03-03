import numpy as np

from population import Population
from discriminator import discriminator

# RETRAIN defines after how many generations the discriminator models are retrained.
RETRAIN_GENERATIONS = 100
RETRAIN_FITNESS = 0.85


def main():
    generated_images = []
    population = Population()
    for generation_number in range(1, 200):

        if generation_number % 10 == 0:
            print(f"-----------------Generation {generation_number}-----------------------")

        population.order_by_fitness()

        if (
                generation_number % RETRAIN_GENERATIONS == 0
                and len(generated_images) > 2000
                # small optimization - python if statements are executed in order they are given. Meaning that if the
                # first two are not BOTH True the third one won't be evaluated and the average_fitness method executed.
                and population.average_fitness() > RETRAIN_FITNESS
        ):
            discriminator.train(
                [image.chromosome for image in population.images if not image.doped]
            )
            generated_images = []

        population.create_offsprings(
            20, 20, selection_function="roulette_wheel", tournament_replacement=False
        )

        for image in population.images:
            generated_images.append(image)

    population.print()

    # Save the generated images
    population.order_by_fitness()
    np.save(
        "generated_images",
        np.array(list(map(lambda image: image.chromosome, population.images))),
    )


if __name__ == "__main__":
    main()
