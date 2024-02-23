from discriminator import discriminator
import numpy as np
image = np.zeros(784)
print(discriminator.estimate(image))