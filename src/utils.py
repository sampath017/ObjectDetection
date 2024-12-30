import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch


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
