import numpy as np
import random

N_GENES = 784
CHROMOSOME_SIZE = 256

P_TURN_WHITE = 0.01
P_TURN_WHITE_ADJACENT = 0.50

IMAGE_WIDTH, IMAGE_HEIGHT = 28, 28


class Image:
    def __init__(self, chromosome=None):
        if chromosome is None:
            self.chromosome = np.zeros([IMAGE_WIDTH, IMAGE_HEIGHT])
            for i in range(IMAGE_WIDTH):
                for j in range(IMAGE_HEIGHT):
                    if self._has_adjacent_white_pixels(i, j) and self._turn_pixel_white(P_TURN_WHITE_ADJACENT):
                        self.chromosome[i, j] = np.random.randint(1, 256, size=1) / 255
                        continue

                    if self._turn_pixel_white(P_TURN_WHITE):
                        self.chromosome[i, j] = np.random.randint(1, 256, size=1) / 255

            self.chromosome = self.chromosome.flatten()
        else:
            self.chromosome = chromosome

        self.fitness = np.float32(0)

    def _has_adjacent_white_pixels(self, i: int, j: int) -> bool:
        # west
        if j > 0 and self.chromosome[i, j - 1] > 0:
            return True

        # east
        if j < IMAGE_WIDTH - 1 and self.chromosome[i, j + 1] > 0:
            return True

        # north
        if i > 0 and self.chromosome[i - 1, j] > 0:
            return True

        # south
        if i < IMAGE_HEIGHT - 1 and self.chromosome[i + 1, j] > 0:
            return True

        return False

    @staticmethod
    def _turn_pixel_white(probability):
        return random.random() < probability

    def __repr__(self):
        return "fitness: {fitness} | chromosome: {chromosome}\n".format(fitness=self.fitness,
                                                                        chromosome=self.chromosome[:10])
