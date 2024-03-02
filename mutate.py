import numpy as np
import random

IMAGE_CENTER = 13.5


def mutate(offspring):
    p = np.random.random()
    if p < 1 / 3:
        _mutate_1(offspring)
    elif p < 2 / 3:
        _mutate_2(offspring)
    else:
        _mutate_3(offspring)


def _mutate_1(offspring):
    """
    It works by changing the value of random pixel in the image,
    and then moving with a probability to any of the 4 adjacent pixels and updating that one etc ....
    """
    index = np.random.randint(4, 24) * 28 + np.random.randint(
        4, 24
    )  # a random pixel somewhat in the middle of the image
    pixel_change = (
        np.random.randint(-255, 255) / 255.0
    )  # the amount the pixel value will change
    p = (
        np.random.random()
    )  # probability of moving to a pixel to the left or right, (1-p) is up or down
    probabilities = [p, 1 - p]
    directions = [
        random.choice([-1, 1]) * 1,
        random.choice([-1, 1]) * 28,
    ]  # the directions are left/right (-1, 1) and up/down (-28, 28)
    for _ in range(np.random.randint(6, 10)):  # number of pixels to be updated
        pixel_change += (
            random.choice([-1, 1]) * np.random.random() / 20
        )  # the pixel change is a bit different for each pixel
        new_pixel_value = (
            offspring.chromosome[index] + pixel_change
        )  # the new pixel value
        if (
            new_pixel_value > 0 and new_pixel_value < 1
        ):  # check that the new pixel value is between 0 and 1
            offspring.chromosome[index] = (
                new_pixel_value  # change the value of the pixel
            )
        direction = np.random.choice(
            directions, p=probabilities
        )  # choose a direction (left/right or up/down)
        new_index = index + direction  # index of the next pixel to be updated
        if new_index >= 0 and new_index < 784:  # check that it is not out of bounds
            index = new_index


def _draw_line(x, y, chromosome):
    """
    This function is called from _mutate_2
    x and why are the coordinates of a random pixel
    The idea is to check surrounding pixels and draw a line either vertically or horizontally
    A line is drawn by turning the pixels on the line blacker, and the surrounding pixels whiter
    """
    vertical = chromosome[max(0, x - 1) : x + 2, y].mean()
    horizontal = chromosome[x, max(0, y - 1) : y + 2].mean()

    if horizontal > vertical:
        # Strengthen horizintal line
        chromosome[max(0, x - 1) : x + 2, y] += np.random.random()
        chromosome[max(0, x - 2) : x + 3, y] += np.random.random() / 2
        # Weaken horizontal lines next to it
        if np.random.random() < 0.5:
            chromosome[max(0, x - 1) : x + 2, max(0, y - 1)] += np.random.random() / 2
            chromosome[max(0, x - 1) : x + 2, min(27, y + 1)] -= np.random.random() / 2
        else:
            chromosome[max(0, x - 1) : x + 2, max(0, y - 1)] -= np.random.random() / 2
            chromosome[max(0, x - 1) : x + 2, min(27, y + 1)] += np.random.random() / 2
        # Clearly weaken horizontal lines 2 pixels away
        chromosome[max(0, x - 1) : x + 2, max(0, y - 2)] -= np.random.random()
        chromosome[max(0, x - 1) : x + 2, min(27, y + 2)] -= np.random.random()
    else:
        # Strengthen vertical line
        chromosome[x, max(0, y - 1) : y + 2] += np.random.random()
        chromosome[x, max(0, y - 2) : y + 3] += np.random.random() / 2
        # Weaken vertical lines next to it
        if np.random.random() < 0.5:
            chromosome[max(0, x - 1), max(0, y - 1) : y + 2] += np.random.random() / 2
            chromosome[min(27, x + 1), max(0, y - 1) : y + 2] -= np.random.random() / 2
        else:
            chromosome[max(0, x - 1), max(0, y - 1) : y + 2] -= np.random.random() / 2
            chromosome[min(27, x + 1), max(0, y - 1) : y + 2] += np.random.random() / 2
        # Clearly weaken vertical lines 2 pixels away
        chromosome[max(0, x - 2), max(0, y - 1) : y + 2] -= np.random.random()
        chromosome[min(27, x + 2), max(0, y - 1) : y + 2] -= np.random.random()

    # No pixel should have a value below 0 or above 1
    chromosome[chromosome < 0] = 0.0
    chromosome[chromosome > 1] = 1.0


def _mutate_2(offspring):
    """
    A random pixel in the image is selected.
        - if the pixel is near the edge, turn it and its nearby pixels white
        - otherwise
            - if the pixel is almost white, turn it and its nearby almost white pixels white
            - otherwize draw a black line
    """
    offspring = offspring.chromosome.reshape((28, 28))
    x, y = np.random.randint(0, 28), np.random.randint(0, 28)
    distance_to_center = np.sqrt((x - IMAGE_CENTER) ** 2 + (y - IMAGE_CENTER) ** 2)

    # if pixel is near the edge turn every nearby pixel in a 3x3 area white
    if distance_to_center > 13:
        offspring[max(0, x - 1) : x + 2, max(0, y - 1) : y + 2] = 0.0
        return offspring

    # if pixel is almost white, turn it and surrounding almost white pixels fully white
    if offspring[x][y] < 0.4:
        x_size, y_size = np.random.randint(2, 5), np.random.randint(2, 5)
        part = offspring[x : x + x_size, y : y + y_size]
        part[part < 0.4] = 0.0
    else:
        # call strengten line, to turn pixels black
        _draw_line(x, y, offspring)


def _mutate_3(offspring):
    """
    Turns a portion of the image white
    """
    offspring = offspring.chromosome.reshape((28, 28))
    x, y = np.random.randint(-4, 28), np.random.randint(-4, 28)
    x_size, y_size = np.random.randint(1, 6), np.random.randint(1, 6)
    offspring[max(0, x) : max(0, x + x_size), max(0, y) : max(0, y + y_size)] = 0
