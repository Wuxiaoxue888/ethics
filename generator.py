import numpy as np

from population import Population
from discriminator import discriminator

def main():
    generated_images = []
    population = Population()
    for generation in range(1, 501):
        #population.print()
        print(f"-----------------Generation {generation}-----------------------")
        population.order_by_fitness()
        #population.print()
        #print("---------------------------------------")
        population.create_offsprings(10, 10, selection_function="tournament", tournament_replacement=False)
        #population.print()
        #print("---------------------------------------")
        #population.print()
        generated_images.append([image.chromosome for image in population.images])
        if generation % 100 == 0:
            last_generated_images = np.array(generated_images).reshape(generation*len(population.images),784)[-1000:]
            discriminator.train(last_generated_images)

    population.print()

    np.save("generated_images", np.array(generated_images))

if __name__ == '__main__':
    main()
