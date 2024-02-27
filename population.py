from typing import List, Any

from image import Image, N_GENES
import operator
import random
import numpy as np
import pandas as pd
from discriminator import discriminator


class Population:
    def __init__(self, size=33, doping_size=10):
        self.images = [Image() for _ in range(size - doping_size)]

        # Dope population with a few good images
        if doping_size > 0:
            real_images_df = pd.read_csv("real_images.csv")[:doping_size] / 255.
            doping_images = list(map(lambda x: Image(chromosome=x), real_images_df.values))
            self.images += doping_images

    def print(self):
        for image in self.images:
            print(image)

    def order_by_fitness(self):
        chromosomes = np.array(list(map(lambda image: image.chromosome, self.images)))
        fitnesses = discriminator.estimate(chromosomes)
        for image, fitness in zip(self.images, fitnesses):
            image.fitness = fitness
        self.images.sort(key=operator.attrgetter('fitness'), reverse=True)

    """
    selection_function has to be one of:
    """
    VALID_SELECTION_FUNCTIONS = ["top_n", "tournament"]

    def create_offsprings(self, n_offsprings: int, n_parents: int, selection_function: str = "top_n", **kwargs) -> None:
        """
        create_offprints MUTATES THE POPULATION (IMAGES) LIST by in place replacing the worst n_offsprings solution with
        the new n_offsprings children.

        :param n_offsprings: defines the number of children to be created
        :param n_parents: defines the number of parents to be selected
        :param selection_function: defines the selection function to be used. Has to be on of function defined in
        VALID_SELECTION_FUNCTIONS.
        :param kwargs: includes the optional settings for the selection functions:
            possible tournament settings:
                tournament_size: defines number of participants in each tournament
                tournament_replacement: defines whether one participant can attend multiple tournaments
        """
        if selection_function == "top_n":
            parents = self._select_top_n(n_parents=n_parents)
        elif selection_function == "tournament":
            tournament_size = kwargs.get("tournament_size", 3)
            tournament_replacement = kwargs.get("tournament_replacement", True)
            parents = self._tournament(tournament_size, n_parents, with_replacement=tournament_replacement)
        else:
            raise Exception("Invalid selection function")

        for i in range(1, n_offsprings, 2):
            split_point = int(N_GENES / 2)
            offspring_a = Image(chromosome=np.concatenate(
                [parents[i - 1].chromosome[:split_point], parents[i].chromosome[split_point:]]))
            offspring_b = Image(chromosome=np.concatenate(
                [parents[i].chromosome[:split_point], parents[i - 1].chromosome[split_point:]]))

            im_len = len(self.images) - 1
            self.images[im_len - (i - 1)] = offspring_a
            self.images[im_len - i] = offspring_b

    def _select_top_n(self, n_parents: int = 10) -> list[Image]:
        """

        :param n_parents: n_parents: declares how many parents will be created. The amount of tournaments is equal to the
            number of parents.
        :return: first n_parents solutions in the self.Images list
        """
        parents = self.images[:n_parents]
        random.shuffle(parents)

        return parents

    def _tournament(self, tournament_size: int, n_parents: int, with_replacement: bool = True) -> list[Image]:
        """
        :param tournament_size: declares how many Images will be included in one tournament. Smaller sizes of
            tournaments give higher changes for weaker Images.
        :param n_parents: declares how many parents will be created. The amount of tournaments is equal to the
            number of parents.
        :param with_replacement: decides whether one solution can attend multiple tournaments
        :return: a list of Images that won their respective tournaments
        """

        winners: list[Image] = []
        selected: list[int] = []

        for _ in range(n_parents):
            best_fitness = 0
            best_solution = None
            for _ in range(tournament_size):
                participant_index = random.randint(0, len(self.images) - 1)
                participant = self.images[participant_index]

                if not with_replacement and participant_index in selected:
                    continue

                if participant.fitness >= best_fitness:
                    best_fitness = participant.fitness
                    best_solution = participant

            winners.append(best_solution)

        return winners
