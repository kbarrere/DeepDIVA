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


def load_dataset(piff_json_file, text_type, resize_height, in_memory=False, workers=1):
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

        logging.error("Loading a PiFF json file and storing images in memory is not implemented yet, sorry!")
        sys.exit(-1)

        train_ds = PiFFImageInMemory(piff_json_file, text_type, resize_height, "training")
        val_ds = PiFFImageInMemory(piff_json_file, text_type, resize_height, "validation")
        test_ds = PiFFImageInMemory(piff_json_file, text_type, resize_height, "test")

        return train_ds, val_ds, test_ds

    else:
        # Store the path to the images in memory and the transcriptions
        train_ds = PiFFImageNotInMemory(piff_json_file, text_type, resize_height, "training")
        val_ds = PiFFImageNotInMemory(piff_json_file, text_type, resize_height, "validation")
        test_ds = PiFFImageNotInMemory(piff_json_file, text_type, resize_height, "test")

        return train_ds, val_ds, test_ds


class PiFFImageNotInMemory(data.Dataset):
    """
    TODO
    """

    def __init__(self, piff_json_file, text_type, resize_height, split, transform=None, target_transform=None, workers=1):
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
        
        if text_type not in ["line", "word"]:
            logging.error("provided text_type (" + split + ") is not line or word")
            sys.exit(-1)

        self.piff_json_file = piff_json_file
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.resize_height = resize_height

        self.piff_json_folder = self.piff_json_file[:-len(self.piff_json_file.split('/')[-1])]
        self.image_paths = []
        self.line_values = []

        f = open(piff_json_file, 'r', encoding='utf-8')
        piff_dict = json.load(f)

        self.line_split_search(piff_dict, text_type, False)

        len_images_path = len(self.image_paths)
        len_line_values = len(self.line_values)

        if len_images_path != len_line_values:
            logging.error("Error while loading PiFF Json file !")
            logging.error("Found " + str(len_images_path) + "image paths but " + str(len_line_values) + "line transciptions.")
            sys.exit(-1)

        logging.info("Loaded " + str(self.__len__()) + " images for the " + self.split + " split.")

        self.shuffle_ind = []
        for i in range(self.__len__()):
            self.shuffle_ind.append(i)
            
        f.close()
    
    def shuffle(self):
        np.random.shuffle(self.shuffle_ind)

    def line_split_search(self, dict, text_type, is_in_split=False):
        if is_in_split:
            if 'type' in dict and dict['type'] == text_type:
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
                        self.line_split_search(child, text_type, is_in_split=True)

        else:
            if 'split' in dict:
                if dict['split'] == self.split:
                    self.line_split_search(dict, text_type, is_in_split=True)
                else:
                    return
            else:
                if 'children' in dict:
                    for child in dict['children']:
                        self.line_split_search(child, text_type, is_in_split=False)



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
        
        image_width, image_height = img.size # Size before transforms

        target = self.line_values[index] # Get the target and apply the current shuffling
        target_len = len(target)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        # Compute new image size after the resize to a fixed height
        # It does not take into account the padding and help separate true
        # pixel images from padding
        if self.resize_height:
            image_width = int(self.resize_height / image_height * image_width)

        return img, target, target_len, image_width

    def __len__(self):
        return len(self.image_paths)
