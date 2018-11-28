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
from datasets.htr_piff_dataset import load_dataset
from template.setup import _dataloaders_from_datasets, _load_mean_std_from_file

import models
from template.setup import _get_optimizer

from template.runner.handwritten_text_recognition.transforms import ResizeHeight, PadRight
from template.runner.handwritten_text_recognition.text_string_transforms import CharToCTCLabel, PadTextToFixedSize, CTCLabelToTensor

def set_up_dataloaders(piff_json, batch_size, workers, inmem, text_type, resize_height, pad_width, pad_text, dictionnary_name, **kwargs):
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
    dictionnary_name : string
        Name of the dictionnary used.
        Determine the number of characters in the dataset.


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
                                             text_type=text_type,
                                             resize_height=resize_height,
                                             in_memory=inmem,
                                             workers=workers)


    # Set up dataset transforms
    logging.debug('Setting up dataset transforms')
    
    # Dataset's Transform
    transform_composition = []
    if resize_height:
        transform_composition.append(ResizeHeight(resize_height))
    if pad_width:
        transform_composition.append(PadRight(pad_width))
    transform_composition.append(transforms.ToTensor())
    transform = transforms.Compose(transform_composition)
    
    train_ds.transform = transform
    val_ds.transform = transform
    test_ds.transform = transform
    
    # Transcription's Transform
    target_transform_composition = []
    target_transform_composition.append(CharToCTCLabel(dictionnary_name))
    if pad_text:
        target_transform_composition.append(PadTextToFixedSize(pad_text))
    target_transform_composition.append(CTCLabelToTensor())
    
    target_transform = transforms.Compose(target_transform_composition)

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
                 start_epoch, inmem, workers, resize_height, pad_width, num_characters=None,
                 **kwargs):
    """
    Instantiate model, optimizer, criterion. Load a pretrained model or resume from a checkpoint.

    Parameters
    ----------
    output_channels : int
        Specify shape of final layer of network. Only used if number_characters is not specified.
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
    inmem : boolean
        Load the whole dataset in memory. If False, only file names are stored and images are loaded
        on demand. This is slower than storing everything in memory.
    workers : int
        Number of workers to use for the dataloaders
    resize_height : int
        The height to which the dataset is resized.
    pad_width : int
        The whidth to which the dataset is padded.
    num_characters : int
        Number of characters that are predicted by the model.
        Could be used for instance if the number of characters is given by the dataloader.
    

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

    output_channels = output_channels if num_characters == None else num_characters
    
    if resize_height and pad_width:
        expected_input_size = (resize_height, pad_width)
    
        model = models.__dict__[model_name](output_channels=output_channels, pretrained=pretrained, expected_input_size=expected_input_size, no_cuda=no_cuda)
    else:
        model = models.__dict__[model_name](output_channels=output_channels, pretrained=pretrained)
    # Get the optimizer created with the specified parameters in kwargs (such as lr, momentum, ... )
    optimizer = _get_optimizer(optimizer_name, model, **kwargs)

    # Get the criterion
    criterion = CTCLoss(size_average=True, length_average=False)
    
    # Transfer model to GPU (if desired)
    if not no_cuda:
        logging.info('Transfer model to GPU')
        model = torch.nn.DataParallel(model).cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    # Load saved model
    if load_model:
        if os.path.isfile(load_model):
            model_dict = None
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
        best_value = 100.0

    return model, criterion, optimizer, best_value, start_epoch
