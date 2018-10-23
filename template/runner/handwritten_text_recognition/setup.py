# Utils
from __future__ import print_function

import logging
import os
import sys

# Torch
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data

#CTC Loss
from warpctc_pytorch import CTCLoss

# DeepDIVA
from datasets.piff_line_dataset import load_dataset #line
#from datasets.piff_word_dataset import load_dataset #word
from template.setup import _dataloaders_from_datasets, _load_mean_std_from_file

import models
from template.setup import _get_optimizer

from template.runner.handwritten_text_recognition.transforms import ResizeHeight, PadRight
from template.runner.handwritten_text_recognition.text_string_transforms import EsposallesCharToCTCLabel, PadToFixedSize, CTCLabelToTensor

def set_up_dataloaders(piff_json, batch_size, workers, inmem, **kwargs):
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
        ResizeHeight(128),
        PadRight(2048),
        transforms.ToTensor()
    ])
    """
    transform = transforms.Compose([
        ResizeHeight(80),
        PadRight(512),
        transforms.ToTensor()
    ])
    """
    train_ds.transform = transform
    val_ds.transform = transform
    test_ds.transform = transform

    
    target_transform = transforms.Compose([
        EsposallesCharToCTCLabel(),
        PadToFixedSize(98), #for line
        #PadToFixedSize(14), #for words
        CTCLabelToTensor()
    ])

    train_ds.target_transform = target_transform
    val_ds.target_transform = target_transform
    test_ds.target_transform = target_transform
    

    train_loader, val_loader, test_loader = _dataloaders_from_datasets(batch_size=batch_size,
                                                                       train_ds=train_ds,
                                                                       val_ds=val_ds,
                                                                       test_ds=test_ds,
                                                                       workers=workers)

    return train_loader, val_loader, test_loader


def set_up_model(output_channels, model_name, pretrained, optimizer_name, no_cuda, resume, load_model,
                 start_epoch, disable_databalancing, dataset_folder, inmem, workers, num_classes=None,
                 **kwargs):
    """
    Instantiate model, optimizer, criterion. Load a pretrained model or resume from a checkpoint.

    Parameters
    ----------
    output_channels : int
        Specify shape of final layer of network. Only used if num_classes is not specified.
    model_name : string
        Name of the model
    pretrained : bool
        Specify whether to load a pretrained model or not
    optimizer_name : string
        Name of the optimizer
    no_cuda : bool
        Specify whether to use the GPU or not
    resume : string
        Path to a saved checkpoint
    load_model : string
        Path to a saved model
    start_epoch : int
        Epoch from which to resume training. If if not resuming a previous experiment the value is 0
    disable_databalancing : boolean
        If True the criterion will not be fed with the class frequencies. Use with care.
    dataset_folder : String
        Location of the dataset on the file system
    inmem : boolean
        Load the whole dataset in memory. If False, only file names are stored and images are loaded
        on demand. This is slower than storing everything in memory.
    workers : int
        Number of workers to use for the dataloaders
    num_classes: int
        Number of classes for the model

    Returns
    -------
    model : nn.Module
        The actual model
    criterion : nn.loss
        The criterion for the network
    optimizer : torch.optim
        The optimizer for the model
    best_value : float
        Specifies the former best value obtained by the model.
        Relevant only if you are resuming training.
    start_epoch : int
        Specifies at which epoch was the model saved.
        Relevant only if you are resuming training.
    """

    # Initialize the model
    logging.info('Setting up model {}'.format(model_name))

    output_channels = output_channels if num_classes == None else num_classes
    model = models.__dict__[model_name](output_channels=output_channels, pretrained=pretrained)

    # Get the optimizer created with the specified parameters in kwargs (such as lr, momentum, ... )
    optimizer = _get_optimizer(optimizer_name, model, **kwargs)

    # Get the criterion
    
    """
    if disable_databalancing:
        criterion = nn.CrossEntropyLoss()
    else:
        try:
            weights = _load_class_frequencies_weights_from_file(dataset_folder, inmem, workers)
            criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(weights).type(torch.FloatTensor))
            logging.info('Loading weights for data balancing')
        except:
            logging.warning('Unable to load information for data balancing. Using normal criterion')
            criterion = nn.CrossEntropyLoss()
    """
    criterion = CTCLoss(size_average=True, length_average=True)
    
    # Transfer model to GPU (if desired)
    if not no_cuda:
        logging.info('Transfer model to GPU')
        model = torch.nn.DataParallel(model).cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    # Load saved model
    if load_model:
        if os.path.isfile(load_model):
            model_dict = torch.load(load_model)
            logging.info('Loading a saved model')
            try:
                model.load_state_dict(model_dict['state_dict'], strict=False)
            except Exception as exp:
                logging.warning(exp)
        else:
            logging.error("No model dict found at '{}'".format(load_model))
            sys.exit(-1)

    # Resume from checkpoint
    if resume:
        if os.path.isfile(resume):
            logging.info("Loading checkpoint '{}'".format(resume))
            checkpoint = torch.load(resume)
            start_epoch = checkpoint['epoch']
            best_value = checkpoint['best_value']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            # val_losses = [checkpoint['val_loss']] #not used?
            logging.info("Loaded checkpoint '{}' (epoch {})"
                         .format(resume, checkpoint['epoch']))
        else:
            logging.error("No checkpoint found at '{}'".format(resume))
            sys.exit(-1)
    else:
        best_value = 0.0

    return model, criterion, optimizer, best_value, start_epoch
