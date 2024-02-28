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
            parent_1 = parents[i - 1]
            parent_2 = parents[i]
            offspring_a, offspring_b = self._crossover(parent_1, parent_2)

            self._mutate(offspring_a)
            self._mutate(offspring_b)

            im_len = len(self.images) - 1
            self.images[im_len - (i - 1)] = offspring_a
            self.images[im_len - i] = offspring_b
    
    """
    _crossover creates two children by combining parts of the two parents
        Splits the parents either horizontally or vertically
    """
    def _crossover(self, parent_1, parent_2):
        horizontal_split = True if random.choice([True, False]) else False
        split_point = int(N_GENES / 2)
        if horizontal_split:
            offspring_a = Image(chromosome=np.concatenate(
                [parent_1.chromosome[:split_point], parent_2.chromosome[split_point:]]))
            offspring_b = Image(chromosome=np.concatenate(
                [parent_2.chromosome[:split_point], parent_1.chromosome[split_point:]]))
        else:
            offspring_a = Image(chromosome=np.concatenate(
                [parent_1.chromosome.reshape((28, 28)).T.flatten()[:split_point],
                 parent_2.chromosome.reshape((28, 28)).T.flatten()[split_point:]
                 ]).reshape((28, 28)).T.flatten())
            offspring_b = Image(chromosome=np.concatenate(
                [parent_2.chromosome.reshape((28, 28)).T.flatten()[:split_point],
                 parent_1.chromosome.reshape((28, 28)).T.flatten()[split_point:]
                 ]).reshape((28, 28)).T.flatten())

        return offspring_a, offspring_b
    
    """
    _mutate mutates the offspring.
        It works by changing the value of random pixel in the image,
        and then moving with a probability to any of the 4 adjacent pixels and updating that one etc ....
    """
    def _mutate(self, offspring):
        index = np.random.randint(4, 24) * 28 + np.random.randint(4, 24) # a random pixel somewhat in the middle of the image
        pixel_change = np.random.randint(-255, 255) / 255. # the amount the pixel value will change
        p = np.random.random() # probability of moving to a pixel to the left or right, (1-p) is up or down
        probabilities = [p, 1-p]
        directions = [random.choice([-1, 1]) * 1,random.choice([-1, 1]) * 28] # the directions are left/right (-1, 1) and up/down (-28, 28) 
        for _ in range(np.random.randint(6, 15)): # number of pixels to be updated
            pixel_change += random.choice([-1, 1]) * np.random.random() / 20 # the pixel change is a bit different for each pixel
            new_pixel_value = offspring.chromosome[index] + pixel_change # the new pixel value
            if new_pixel_value > 0 and new_pixel_value < 1: # check that the new pixel value is between 0 and 1
                offspring.chromosome[index] = new_pixel_value # change the value of the pixel
            direction = np.random.choice(directions, p=probabilities) # choose a direction (left/right or up/down)
            new_index = index + direction # index of the next pixel to be updated
            if new_index >= 0 and new_index < 784: # check that it is not out of bounds
                index = new_index

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
