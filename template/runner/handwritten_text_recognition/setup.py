# Utils
from __future__ import print_function

import logging
import os

# Torch
import torchvision.transforms as transforms

# DeepDIVA
from datasets.piff_line_dataset import load_dataset
from template.setup import _dataloaders_from_datasets, _load_mean_std_from_file


def set_up_dataloaders(model_expected_input_size, piff_json, batch_size, workers, inmem, **kwargs):
    """
    Set up the dataloaders for the specified datasets.

    Parameters
    ----------
    model_expected_input_size : tuple
        Specify the height and width that the model expects.
    piff_json : string
        File describing the dataset following the PiFF format
    batch_size : int
        Number of datapoints to process at once
    workers : int
        Number of workers to use for the dataloaders
    inmem : boolean
        Flag : if False, the dataset is loaded in an online fashion i.e. only file names are stored
        and images are loaded on demand. This is slower than storing everything in memory.


    Returns
    -------
    train_loader : torch.utils.data.DataLoader
    val_loader : torch.utils.data.DataLoader
    test_loader : torch.utils.data.DataLoader
        Dataloaders for train, val and test.
    """

    # Recover dataset name
    file_name = os.path.basename(os.path.normpath(piff_json))
    logging.info('Loading {} from:{}'.format(file_name, piff_json))

    ###############################################################################################
    # Load the dataset splits as images
    train_ds, val_ds, test_ds = load_dataset(piff_json_file=piff_json,
                                             in_memory=inmem,
                                             workers=workers)


    # Set up dataset transforms
    logging.debug('Setting up dataset transforms')

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_ds.transform = transform
    val_ds.transform = transform
    test_ds.transform = transform

    train_loader, val_loader, test_loader = _dataloaders_from_datasets(batch_size=batch_size,
                                                                       train_ds=train_ds,
                                                                       val_ds=val_ds,
                                                                       test_ds=test_ds,
                                                                       workers=workers)

    return train_loader, val_loader, test_loader
