import numpy as np

N_GENES = 784
CHROMOSOME_SIZE = 256


class Image:
    def __init__(self, chromosome=None):
        if chromosome is None:
            self.chromosome = np.random.randint(0, 256, size=784) / 255.
        else:
            self.chromosome = chromosome

        self.fitness = np.float32(0)

    def __repr__(self):
        return "fitness: {fitness} | chromosome: {chromosome}\n".format(fitness=self.fitness, chromosome=self.chromosome[:10])