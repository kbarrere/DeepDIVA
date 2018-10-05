"""
Load a dataset of images by specifying the folder where its located.
"""

# Utils
import logging
import sys
import json
import numpy as np

# Torch related stuff
import torch.utils.data as data
from PIL import Image
import torch


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
        # Store all images and transcription directly in the memory

        logging.error("Loading a PiFF json file and storing images in memeory not implemented !")
        sys.exit(-1)

        train_ds = LineImageInMemory(piff_json_file, "training")
        val_ds = LineImageInMemory(piff_json_file, "validation")
        test_ds = LineImageInMemory(piff_json_file, "test")

        return train_ds, val_ds, test_ds

    else:
        # Store the path to the images in memory and the transcriptions
        train_ds = LineImageNotInMemory(piff_json_file, "training")
        val_ds = LineImageNotInMemory(piff_json_file, "validation")
        test_ds = LineImageNotInMemory(piff_json_file, "test")

        return train_ds, val_ds, test_ds


class LineImageNotInMemory(data.Dataset):
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

        self.piff_json_file = piff_json_file
        self.split = split
        self.transform = transform
        self.target_transform = target_transform

        self.piff_json_folder = self.piff_json_file[:-len(self.piff_json_file.split('/')[-1])]
        self.image_paths = []
        self.line_values = []

        f = open(piff_json_file, 'r')
        piff_dict = json.load(f)

        self.line_split_search(piff_dict, False)

        len_images_path = len(self.image_paths)
        len_line_values = len(self.line_values)

        if len_images_path != len_line_values:
            logging.error("Error while loading PiFF Json file !")
            logging.error("Found " + str(len_images_path) + "image path but " + str(len_line_values) + "line transciptions.")
            sys.exit(-1)

        # TODO remove, temp classification task to test
        self.labels = ['P' in item for item in self.line_values]
        self.classes = np.unique(self.labels)

        logging.info("Loaded " + str(self.__len__()) + " lines for the " + self.split + " split.")

        f.close()

    def line_split_search(self, dict, is_in_split=False):
        if is_in_split:
            if 'type' in dict and dict['type'] == 'line':
                path = ""
                if 'path' in dict:
                    path = dict['path']

                self.image_paths.append(self.piff_json_folder + path)

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
        target : string
            transciption of the text inside the image
        """

        # Weird way to open things due to issue https://github.com/python-pillow/Pillow/issues/835
        with open(self.image_paths[index], 'rb') as f:
            img = Image.open(f)
            # Copy the image to avoid bug when the file is closed later
            img = img.copy()
            img = img.resize([227, 227], Image.ANTIALIAS) # TODO remove

        #target = self.line_values[index]

        #target = torch.LongTensor([0])
        target = 0
        # TODO Remove
        if 'P' in self.line_values[index]:
            #target = torch.LongTensor([1])
            target = 1


        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.image_paths)
