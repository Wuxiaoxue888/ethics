from image import Image, N_GENES
import operator
import random
import numpy as np


class Population:
    def __init__(self, size=99):
        self.images = [Image() for _ in range(size)]
        self.images.append(Image(chromosome=[  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,  49, 139,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   3, 206, 255,  29,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   6, 182, 255, 255, 250, 207,  47,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   4, 172, 254, 255, 255, 236, 255, 129,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,  41, 168, 254, 203, 107, 167, 248,
       234, 255, 121,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,  19, 219, 255, 253, 114, 169,  77,
         0, 164, 255, 206, 235,   4,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,  90, 255, 122,  24,  21,
        46,   5,   0, 174, 242, 248, 255,  69,   1,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  45, 251, 255,
       250, 255, 255, 167,   1, 226, 164, 192, 255, 249, 240, 180, 159,
       141, 113,  92,  45,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        56, 110, 128, 102, 137, 255,  77, 255, 106,   4, 145, 236, 232,
       197, 226, 255, 255, 255, 255, 233,  52,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   9, 245, 172, 255,  54,   0,   0,
       240, 139,   0, 102, 255,  32,  31, 199, 249, 219,   9,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0, 179, 233, 250,   8,
         0,  27, 255,  98,   0, 145, 238,   0,   0, 164, 230, 255, 128,
         5,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 105, 255,
       212,   0,   0,  68, 255,  57,   0, 188, 193,   0,   0, 157, 224,
       255, 255, 161,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        26, 245, 209,  27,   0, 109, 255,  17,   0, 231, 150,   0,   0,
        22, 164, 240, 146, 248,   8,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,  70, 243, 242, 116, 161, 230,   0,  18, 255, 107,
         0,   0,  68, 247, 155,  91, 255,  31,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,  28, 177, 255, 255, 253, 185, 163,
       255, 209, 217, 216, 255, 213,  15,  70, 255,  52,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,  55, 255, 246, 255,
       227, 238, 242, 254, 249, 217, 254,  17,   0,  49, 255,  73,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  49, 255,
       162, 255,  50,   0,   0, 166, 211, 147, 231,   0,   0,  38, 255,
       202,   3,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        43, 255, 137, 255,  63,   0,   0, 160, 217, 176, 201,   0,   0,
        87, 255, 255,  72,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,  37, 255,  90, 182,  66,   0,   0, 153, 222, 108, 104,
         0,   0,  49, 172, 173,  52,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   6, 148,  29,   0,   0,   0,   0,  54,  96,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0]))

    def print(self):
        for image in self.images:
            print(image)

    def order_by_fitness(self):
        for i in self.images:
            i.compute_fitness()

        self.images.sort(key=operator.attrgetter('fitness'), reverse=True)

    """
    mating_function has to be one of:
    """
    VALID_MATING_FUNCTIONS = ["top_n"]  # TODO add more

    def create_offsprings(self, mating_function="top_n", **kwargs):
        if mating_function == "top_n":
            n_parents = kwargs.get("n_parents", 10)
            offsprings = kwargs.get("offsprings", 10)
            return self._select_top_n(n_parents=n_parents, offsprings=offsprings)

        raise Exception("Invalid mating function given")

    def _select_top_n(self, n_parents=10, offsprings=10):
        parents = self.images[:n_parents]
        random.shuffle(parents)

        for i in range(1, offsprings, 2):
            split_point = int(N_GENES/2)
            offspring_a = Image(chromosome=parents[i-1].chromosome[:split_point] + parents[i].chromosome[split_point:])
            offspring_b = Image(chromosome=parents[i].chromosome[:split_point] + parents[i-1].chromosome[split_point:])

            im_len = len(self.images) - 1
            self.images[im_len - (i - 1)] = offspring_a
            self.images[im_len - i] = offspring_b
