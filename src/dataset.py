from torch.utils.data import Dataset
from torchvision.datasets import VOCDetection
import torch


class ObjectDetectionDataset(Dataset):
    def __init__(self, path):
        data_path = path/"single_object_images.pt"
        if not data_path.exists():
            dataset = VOCDetection(
                path, year="2007", image_set="trainval", download=True)

            print("preparing data!")
            self.single_object_images = []
            for index in range(len(dataset)):
                image, target = dataset[index]
                objects = target["annotation"]["object"]
                if len(objects) == 1 and int(objects[0]["difficult"]) == 0:
                    self.single_object_images.append(dataset[index])

            torch.save(self.single_object_images,
                       path/"single_object_images.pt")
        else:
            print("Loading from disk!")
            self.single_object_images = torch.load(data_path)

        classes = [
            "person",
            "bird",
            "cat",
            "cow",
            "dog",
            "horse",
            "sheep",
            "aeroplane",
            "bicycle",
            "boat",
            "bus",
            "car",
            "motorbike",
            "train",
            "bottle",
            "chair",
            "diningtable",
            "pottedplant",
            "sofa",
            "tvmonitor"
        ]

        # Create the class_to_idx dictionary
        self.class_to_idx = {cls_name: idx for idx,
                             cls_name in enumerate(classes)}

    def __getitem__(self, index):
        image = self.single_object_images[index][0]
        target = self.single_object_images[index][1]["annotation"]["object"][0]["name"]
        target = self.class_to_idx[target]

        return image, target

    def __len__(self):
        return len(self.single_object_images)
