from image import Image, N_GENES
import operator
import random


class Population:
    def __init__(self, size=100):
        self.images = [Image() for _ in range(size)]

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
