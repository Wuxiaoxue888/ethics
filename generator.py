import numpy as np

from population import Population
from discriminator import discriminator

def main():
    generated_images = []
    population = Population()
    generations_before_training_again = 30
    for generation in range(1, 11):

        if generation % 10 == 0:
            print(f"-----------------Generation {generation}-----------------------")

        population.order_by_fitness()

        generations_before_training_again -= 1
        if population.is_fit() and generations_before_training_again < 0 and len(generated_images) > 2000:
            discriminator.train(list(map(lambda image: image.chromosome, generated_images[-2000:])))
            generations_before_training_again = 64

        population.create_offsprings(20, 20, selection_function="roulette_wheel", tournament_replacement=False)

        for image in population.images:
            generated_images.append(image)
        generated_images = generated_images[-25_000:]


    population.print()

    # Save the generated images
    population.images = generated_images[-25_000:]
    population.order_by_fitness()
    np.save("generated_images", np.array(list(map(lambda image: image.chromosome, population.images))))

if __name__ == '__main__':
    main()
