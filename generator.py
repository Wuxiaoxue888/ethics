import numpy as np

from population import Population

def main():
    generated_images = []
    population = Population()
    for i in range(2):
        #population.print()
        print(f"-----------------Generation {i+1}-----------------------")
        population.order_by_fitness()
        #population.print()
        #print("---------------------------------------")
        population.create_offsprings(10, 10, selection_function="tournament", tournament_replacement=False)
        #population.print()
        #print("---------------------------------------")
        population.print()
        generated_images.append([image.chromosome for image in population.images])

    np.save("generated_images", np.array(generated_images))

if __name__ == '__main__':
    main()
