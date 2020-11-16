from torchvision.datasets import VisionDataset

from PIL import Image

import os
import os.path
import sys


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def make_dataset(directory, split, class_to_idx):
    instances = []
    directory = os.path.expanduser(directory)
    split_filepath = 'Caltech101/' + split + '.txt'
    split_instances = open(split_filepath).read().splitlines()

    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        if 'BACKGROUND_Google' in target_dir:
            continue
        for root, _, filenames in sorted(os.walk(target_dir, followlinks=True)):
            for filename in sorted(filenames):
                path = os.path.join(root, filename)
                instance = '/'.join([root.split('\\')[1], filename])
                if instance in split_instances:
                    item = path, class_index
                    instances.append(item)
    return instances


class Caltech(VisionDataset):
    """A generic data loader where the samples are arranged in this way: ::
           root/class_x/xxx.ext
           root/class_x/xxy.ext
           root/class_y/123.ext

       Args:
           root (string): Root directory path.
           split (string): List of images to include in the split.
           loader (callable): A function to load a sample given its path.
               both extensions and is_valid_file should not be passed.
           transform (callable, optional): A function/transform that takes in
               a sample and returns a transformed version.
           target_transform (callable, optional): A function/transform that takes
               in the target and transforms it.

       Attributes:
           classes (list): List of the class names sorted alphabetically.
           class_to_idx (dict): Dict with items (class_name, class_index).
           samples (list): List of (sample path, class_index) tuples
           targets (list): The class_index value for each image in the dataset
       """
    def __init__(self, root, split='train', transform=None, target_transform=None):
        super(Caltech, self).__init__(root, transform=transform, target_transform=target_transform)

        self.split = split  # This defines the split you are going to use
        # (split files are called 'train.txt' and 'test.txt')

        """
        - Here you should implement the logic for reading the splits files and accessing elements
        - If the RAM size allows it, it is faster to store all data in memory
        - PyTorch Dataset classes use indexes to read elements
        - You should provide a way for the __getitem__ method to access the image-label pair
          through the index
        - Labels should start from 0, so for Caltech you will have labels 0...100 (excluding the background class) 
        """

        classes, class_to_idx = self._find_classes(self.root)
        samples = make_dataset(self.root, self.split, class_to_idx)

        self.loader = pil_loader
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

    @staticmethod
    def _find_classes(directory):
        """
        Finds the class folders in a dataset.

        Args:
            directory (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """

        classes = [d.name for d in os.scandir(directory) if d.is_dir()]
        classes.remove('BACKGROUND_Google')
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def __getitem__(self, index):
        """
        __getitem__ should access an element through its index

        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """

        # Provide a way to access image and label via index
        # Image should be a PIL Image, label can be int
        path, label = self.samples[index]
        image = self.loader(path)

        # Applies preprocessing when accessing the image
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        """
        The __len__ method returns the length of the dataset
        It is mandatory, as this is used by several other components
        """

        # Provide a way to get the length (number of elements) of the dataset
        length = len(self.samples)
        return length
