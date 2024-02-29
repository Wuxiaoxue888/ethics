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
        if generation % 10 == 0: # save generated images
            generated_images.append([image.chromosome for image in population.images[:10]])

        if generation % 100 == 0: # train the network more every 100 generations
            generated_images_1d_array = np.array(generated_images).reshape(np.array(generated_images).shape[0]*np.array(generated_images).shape[1],784)
            last_generated_images = generated_images_1d_array[-1000:]
            discriminator.train(last_generated_images)

    population.print()

    np.save("generated_images", np.array(generated_images))

if __name__ == '__main__':
    main()
