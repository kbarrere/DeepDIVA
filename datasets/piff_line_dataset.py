"""
Load a dataset of images by specifying the folder where its located.
"""

# Utils
import logging
import os
import sys
from multiprocessing import Pool
import cv2
import numpy as np

# Torch related stuff
import torch.utils.data as data
import torchvision
from PIL import Image

from util.misc import get_all_files_in_folders_and_subfolders, has_extension

import json


def load_dataset(piff_json_file, in_memory=False, workers=1):
    """
    Read the content of a json file following the Pivot File Format (PiFF) :
    https://gitlab.univ-nantes.fr/mouchere-h/PiFFgroup

    It then get the line value and image according to what is inside the format

    Parameters
    ----------
    piff_json_file: string
        Path to the json file following the PiFF format

    in_memory : boolean
        Load the whole dataset in memory. If False, only file names are stored and images are loaded
        on demand. This is slower than storing everything in memory.

    workers: int
        Number of workers to use for the dataloaders

    Returns
    -------
    train_ds : data.Dataset

    val_ds : data.Dataset

    test_ds : data.Dataset
        Train, validation and test splits
    """

    if in_memory:
        # store all the sketches in memory
        train_ds = LineImageInMemory(piff_json_file, "training")
        val_ds = LineImageInMemory(piff_json_file, "validation")
        test_ds = LineImageInMemory(piff_json_file, "test")

    else:
        # TODO
        logging.error("in_memory = False not implemented.")
    sys.exit(-1)


class LineImageInMemory(data.Dataset):
    """
    TODO
    """

    def __init__(self, piff_json_file, split, transform=None, target_transform=None, workers=1):
        """
        Load the data in memory and prepares it as a dataset.

        Parameters
        ----------
        piff_json_file : string
            Path to the json file following the PiFF format
            This file contains the line value and the path to the image line
        split: string
            Indicates which split it should load
            Should be either train, val or test
        transform : torchvision.transforms
            Transformation to apply on the data
        target_transform : torchvision.transforms
            Transformation to apply on the labels
        workers: int
            Number of workers to use for the dataloaders
        """

        if split not in ["training", "validation", "test"]:
            logging.error("provided split (" + split + ") is not train, valid or test")
            sys.exit(-1)

        self.split = split
        self.image_paths = []
        self.line_values = []

        f = open(piff_json_file, 'r')
        piff_dict = json.load(f)

        self.line_split_search(piff_dict, False)

        f.close()

    def line_split_search(self, dict, is_in_split=False):
        if is_in_split:
            if 'type' in dict and dict['type'] == 'line':
                path = ""
                if 'path' in dict:
                    path = dict['path']
                    print(path)

                value = ""
                if 'value' in dict:
                    value = dict['value']

                self.line_values.append(value)

            else:
                if 'children' in dict:
                    for child in dict['children']:
                        self.line_split_search(child, is_in_split=True)

        else:
            if 'split' in dict:
                if dict['split'] == self.split:
                    self.line_split_search(dict, is_in_split=True)
                else:
                    return
            else:
                if 'children' in dict:
                    for child in dict['children']:
                        self.line_split_search(child, is_in_split=False)



    def __getitem__(self, index):
        """
        Retrieve a sample by index

        Parameters
        ----------
        index : int

        Returns
        -------
        img : FloatTensor
        target : int
            label of the image
        """

        img, target = self.data[index], self.labels[index]

        # Doing this so that it is consistent with all other datasets to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


class ImageFolderApply(data.Dataset):
    """
    TODO fill me
    """

    def __init__(self, path, transform=None, target_transform=None, classify=False):
        """
        TODO fill me

        Parameters
        ----------
        path : string
            Path to the dataset on the file System
        transform : torchvision.transforms
            Transformation to apply on the data
        target_transform : torchvision.transforms
            Transformation to apply on the labels
        """
        self.dataset_folder = os.path.expanduser(path)
        self.transform = transform
        self.target_transform = target_transform

        if classify is True:
            # Get an online dataset
            dataset = torchvision.datasets.ImageFolder(path)

            # Extract the actual file names and labels as entries
            self.file_names = np.asarray([item[0] for item in dataset.imgs])
            self.labels = np.asarray([item[1] for item in dataset.imgs])
        else:
            # Get all files in the folder that are images
            self.file_names = self._get_filenames(self.dataset_folder)

            # Extract the label for each file (assuming standard format of root_folder/class_folder/img.jpg)
            self.labels = [item.split('/')[-2] for item in self.file_names]

        # Set expected class attributes
        self.classes = np.unique(self.labels)

    def _get_filenames(self, path):
        file_names = []
        for item in get_all_files_in_folders_and_subfolders(path):
            if has_extension(item, ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']):
                file_names.append(item)
        return file_names

    def __getitem__(self, index):
        """
        Retrieve a sample by index and provides its filename as well

        Parameters
        ----------
        index : int

        Returns
        -------
        img : FloatTensor
        target : int
            label of the image
        filename : string
        """

        # Weird way to open things due to issue https://github.com/python-pillow/Pillow/issues/835
        with open(self.file_names[index], 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')

        target, filename = self.labels[index], self.file_names[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, filename

    def __len__(self):
        return len(self.file_names)
