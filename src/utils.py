import random
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from torchvision.io import read_image
import torch


# TODO
def visulize_data(dataset, samples_per_class=10):
    start = 0
    end = len(dataset)
    step = 6000

    # [(0, 5000), (5000, 10000), ...]
    ranges = [(i, i + step) for i in range(start, end, step)]
    indices = []
    for start, end in ranges:
        indices.extend(random.sample(range(start, end), samples_per_class))

    images = [read_image(dataset.samples[i][0]) for i in indices]

    grid = make_grid(images, nrow=samples_per_class)
    grid = grid.permute(1, 2, 0)  # CHW --> HWC
    grid = grid.div(255.0)  # normilize

    plt.figure(figsize=(10, 10))

    for image, label in dataset.class_to_idx.items():
        plt.text(x=-4, y=34 * label + 18, s=image, ha='right')

    plt.imshow(grid)
    plt.axis(False)
    plt.show()
