"""
Load a dataset of sketch that are represented in a pen stroke sequence format
"""
# Utils
import logging
import os
import random
import sys
from multiprocessing import Pool

import numpy as np
import torch.utils.data as data
# Torch related stuff
import torchvision
#TODO remove it ?
from tqdm import trange

def load_dataset(dataset_folder, in_memory=True, workers=1):
    """
    Loads the dataset from file system and provides the dataset splits for train validation and test.

    The dataset folder is expected to contains npz files.
    It then load all npz files in the provided folder.

    Parameters
    ----------
    dataset_folder : string
        Path to the dataset on the file System
    in_memory : boolean
        Load the whole dataset in memory. If False, only file names are stored and sketches are loaded
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

    # Sanity check on the dataset folder
    if not os.path.isdir(dataset_folder):
        logging.error("Dataset folder not found in the args.dataset_folder=" + dataset_folder)
        sys.exit(-1)
    #TODO check if there is npz files

    is_npz = False
    for filename in os.listdir(dataset_folder):
        if filename.endswith(".npz"):
            is_npz = True
            break
    if not is_npz:
        logging.error("No npz files were found in the follder " + dataset_folder)
        sys.exit(-10)

    if in_memory:
        #store all the sketches in memory
        train_ds = SketchFolderInMemory(dataset_folder, "train")
        val_ds = SketchFolderInMemory(dataset_folder, "valid")
        test_ds = SketchFolderInMemory(dataset_folder, "test")

    else:
        #TODO 
        logging.error("in_memory = False not implemented.")
        sys.exit(-1)
 
    return train_ds, val_ds, test_ds


class SketchFolderInMemory(data.Dataset):
    """
    This class loads the data provided and stores it entirely in memory as a dataset.
    """

    def __init__(self, dataset_folder, split,
                 transform=None, target_transform=None, workers=None):
        """
        Load the data in memory and prepares it as a dataset.

        Parameters
        ----------
        dataset_folder : string
            Path to the dataset on the file System
        split:
            Indicates which split it should load
            Should be either train, val or test
        transform : torchvision.transforms
            Transformation to apply on the data
        target_transform : torchvision.transforms
            Transformation to apply on the labels
        workers: int
            Number of workers to use for the dataloaders
        """

        if split not in ["train", "valid", "test"]:
            logging.error("provided split (" + split + ") is not train, valid or test")
            sys.exit(-1)

        self.dataset_folder = os.path.expanduser(dataset_folder)
        self.transform = transform
        self.target_transform = target_transform
        self.split = split
   
        data = []
        labels =[]

        for filename in os.listdir(self.dataset_folder):
            if filename.endswith(".npz"):
                label = os.path.splitext(filename)[0]

                #load the file
                array_file = np.load(os.path.join(dataset_folder, filename), encoding="latin1")
                
                #get the data from array_file
                for sketch in array_file[self.split]:
                    data.append(sketch)
                    labels.append(label)

        self.data = np.asarray(data)
        self.labels = np.asarray(labels)
        self.classes = np.unique(self.labels)
        

    def __getitem__(self, index):
        """
        Retrieve a sample by index

        Parameters
        ----------
        index : int

        Returns
        -------
        sketch : TODO
        target : int
            label of the sketch
        """
        
        sketch = self.data[index]
        target = self.labels[index]

        if self.transform is not None:
            sketch = self.transform(sketch)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sketch, target

    def __len__(self):
        return len(self.data)
