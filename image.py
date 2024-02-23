import numpy as np
import random
from discriminator import discriminator

N_GENES = 784
CHROMOSOME_SIZE = 256


class Image:
    def __init__(self, chromosome=None):
        if chromosome is None:
            self.chromosome = np.random.randint(0, 256, size=784)
        else:
            self.chromosome = np.array(chromosome)

        self.fitness = 0

    def __repr__(self):
        return "fitness: {fitness} | chromosome: {chromosome}\n".format(fitness=self.fitness, chromosome=self.chromosome[:10])

    # TODO alternatively we can compute the fitness in the constructor when an Image instance
    # is created. For now lets keep it simple
    def compute_fitness(self):
        # TODO ask model about the fitness
        x = discriminator.estimate(self.chromosome)
        if x > 0:
            print(self.chromosome)
        self.fitness = x
        
