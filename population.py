from typing import List, Any
from image import Image
import operator
import random
import numpy as np
import pandas as pd

from image import Image, N_GENES
from discriminator import discriminator
import csv
from mutate import mutate

REAL_IMAGES = pd.read_csv("real_images.csv") / 255.

class Population:
    def __init__(self, size=256):
        self.images = [Image() for _ in range(size)]
        self.dope()

    def dope(self, doping_size=16):
        """
        Replaces images in the population with real images
        """
        doping_images = [Image(chromosome=REAL_IMAGES.iloc[np.random.randint(0, len(REAL_IMAGES)-1)].values) for _ in range(doping_size)]
        self.images = self.images[:-doping_size]
        self.images += doping_images

    def print(self):
        for image in self.images:
            print(image)

    def is_fit(self):
        """
        Checks if many of the images in the population have a high fitness
        """
        for image in self.images[:int(len(self.images)/2)]:
            if image.fitness < 0.85:
                return False
        return True

    def order_by_fitness(self):
        chromosomes = np.array(list(map(lambda image: image.chromosome, self.images)))
        fitnesses = discriminator.estimate(chromosomes)
        for image, fitness in zip(self.images, fitnesses):
            image.fitness = fitness
        self.images.sort(key=operator.attrgetter('fitness'), reverse=True)

    """
    selection_function has to be one of:
    """
    VALID_SELECTION_FUNCTIONS = ["top_n", "tournament","roulette_wheel"]

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
        elif selection_function == "roulette_wheel":
            parents = self._roulette_wheel_selection(n_parents)
        else:
            raise Exception("Invalid selection function")

        for i in range(1, n_offsprings, 2):
            parent_1 = parents[i - 1]
            parent_2 = parents[i]
            offspring_a, offspring_b = self._crossover(parent_1, parent_2)

            mutate(offspring_a)
            mutate(offspring_b)

            im_len = len(self.images) - 1
            self.images[im_len - (i - 1)] = offspring_a
            self.images[im_len - i] = offspring_b 

    def _crossover(self, parent_1, parent_2):
        """
        Checks all possible splits both horizontally and vertically and chooses the best one
        Splits the parents in two parts, combines one part from each parent, and returns them as offsprings
        """
        parent_1 = parent_1.chromosome.reshape((28, 28))
        parent_2 = parent_2.chromosome.reshape((28, 28))

        best_vertical_split_value = -1
        best_horizontal_split_value = -1
        best_vertical_split_index = 14
        best_horizontal_split_index = 14
        for i in range(6, 22):
            vertical_split_value = np.abs(parent_1[:, i] - parent_2[:, i+1]).mean()
            if vertical_split_value > best_vertical_split_value:
                best_vertical_split_value = vertical_split_value
                best_vertical_split_index = i+1
            horizontal_split_value = np.abs(parent_1[i, :] - parent_2[i+1, :]).mean()
            if horizontal_split_value > best_horizontal_split_value:
                best_horizontal_split_value = horizontal_split_value
                best_horizontal_split_index = i+1

        offspring_a, offspring_b = parent_1.copy(), parent_2.copy()
        if best_horizontal_split_value > best_vertical_split_value:
            # Split horizontally on best_horizontal_split_index
            offspring_a[:best_horizontal_split_index, :] = parent_2[:best_horizontal_split_index, :]
            offspring_b[:best_horizontal_split_index, :] = parent_1[:best_horizontal_split_index, :]
        else:
            # Split vertically on best_vertical_split_index
            offspring_a[:, :best_vertical_split_index] = parent_2[:, :best_vertical_split_index]
            offspring_b[:, :best_vertical_split_index] = parent_1[:, :best_vertical_split_index]

        return Image(chromosome=offspring_a.flatten()), Image(chromosome=offspring_b.flatten())

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
        winners_indexes: list[int] = []

        for _ in range(n_parents):
            best_fitness = 0
            best_solution = None
            for _ in range(tournament_size):
                participant_index = random.randint(0, len(self.images) - 1)

                while not with_replacement and participant_index in winners_indexes: # needs to loop until a valid participant is chosen
                    participant_index = random.randint(0, len(self.images) - 1)
                
                participant = self.images[participant_index]

                if participant.fitness >= best_fitness:
                    best_fitness = participant.fitness
                    best_solution = participant
                    best_index = participant_index

            winners.append(best_solution)
            winners_indexes.append(best_index)

        return winners

    def _roulette_wheel_selection(self, n_parents):
        total_fitness = sum(image.fitness for image in self.images)
        selection_probs = [image.fitness / total_fitness for image in self.images]
        return np.random.choice(self.images, n_parents, p=selection_probs)

    def _multi_point_crossover(self, parent1, parent2, points=4):
        crossover_points = sorted(np.random.choice(range(1, N_GENES), points - 1, replace=False))
        child_chromosome = np.array(parent1.chromosome)
        for i, point in enumerate(crossover_points):
            if i % 2 == 0:
                child_chromosome[point:] = parent2.chromosome[point:]
            else:
                child_chromosome[point:] = parent1.chromosome[point:]
        return Image(chromosome=child_chromosome)

    def export_to_csv(self, file_name):
        with open(file_name, 'w', newline='') as file:
            writer = csv.writer(file)
            column_names = [i for i in range(784)]

            writer.writerow(column_names)
            for image in self.images:
                writer.writerow(list(image.chromosome))
