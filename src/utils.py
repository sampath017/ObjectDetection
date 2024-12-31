import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from torchvision.transforms import v2
import torchvision


def plot_detection_datapoint(image, target):
    if isinstance(image, torch.Tensor):
        image = (image * 255).permute(1, 2, 0)

    plt.imshow(image)
    ax = plt.gca()

    for obj in target["annotation"]["object"]:
        bbox = obj['bndbox']
        xmin, ymin, xmax, ymax = map(
            int, [bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']])

        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                 linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.text(xmin, ymin - 5, obj['name'], color='red',
                 fontsize=10, backgroundcolor='white')

    plt.title(str(image.size))
    plt.axis('off')
    plt.show()


def plot_datapoint(image, target, inverse_transform=None):
    if inverse_transform:
        image = inverse_transform(image)
        image = (image * 255).permute(1, 2, 0)
        image = image.to(torch.int16)

    H, W = image.shape[:-1]

    plt.axis('off')
    plt.title(f"{target} {H}x{W}")
    plt.imshow(image)
    plt.show()


def plot_datapoints(image1, image2, target1, target2, inverse_transform1=None, inverse_transform2=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    if inverse_transform1:
        image1 = inverse_transform1(image1)
        image1 = (image1 * 255).permute(1, 2, 0)
        image1 = image1.to(torch.int16)

    if inverse_transform2:
        image2 = inverse_transform2(image2)
        image2 = (image2 * 255).permute(1, 2, 0)
        image2 = image2.to(torch.int16)

    H1, W1 = image1.shape[:-1]
    H2, W2 = image2.shape[:-1]

    axes[0].imshow(image1)
    axes[0].axis('off')
    axes[0].set_title(f"{target1} {H1}x{W1}")

    axes[1].imshow(image2)
    axes[1].axis('off')
    axes[1].set_title(f"{target2} {H2}x{W2}")

    plt.show()
