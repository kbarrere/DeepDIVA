"""
This file is the template for the boilerplate of train/test of a DNN for image classification

There are a lot of parameter which can be specified to modify the behaviour and they should be used 
instead of hard-coding stuff.
"""

import logging
import sys

# Utils
import numpy as np

# DeepDIVA
import models
# Delegated
from template.runner.handwritten_text_recognition import evaluate, train
from template.runner.handwritten_text_recognition.setup import set_up_dataloaders, set_up_model
from util.misc import checkpoint, adjust_learning_rate


class HandwrittenTextRecognition:
    @staticmethod
    def single_run(writer, current_log_folder, model_name, epochs, lr, decay_lr, validation_interval,
                   **kwargs):
        """
        This is the main routine where train(), validate() and test() are called.

        Parameters
        ----------
        writer : Tensorboard.SummaryWriter
            Responsible for writing logs in Tensorboard compatible format.
        current_log_folder : string
            Path to where logs/checkpoints are saved
        model_name : string
            Name of the model
        epochs : int
            Number of epochs to train
        lr : float
            Value for learning rate
        kwargs : dict
            Any additional arguments.
        decay_lr : boolean
            Decay the lr flag
        validation_interval: int
            Run evaluation on validation set every N epochs

        Returns
        -------
        train_value : ndarray[floats] of size (1, `epochs`)
            Accuracy values for train split
        val_value : ndarray[floats] of size (1, `epochs`+1)
            Accuracy values for validation split
        test_value : float
            Accuracy value for test split
        """

        # Setting up the dataloaders
        train_loader, val_loader, test_loader = set_up_dataloaders(**kwargs)
        num_classes = 64 # TODO : just temp !

        # Setting up model, optimizer, criterion
        model, criterion, optimizer, best_value, start_epoch = set_up_model(num_classes=num_classes,
                                                                            model_name=model_name,
                                                                            lr=lr,
                                                                            train_loader=train_loader,
                                                                            **kwargs)

        # Core routine
        logging.info('Begin training')
        val_value = np.zeros((epochs + 1 - start_epoch))
        train_value = np.zeros((epochs - start_epoch))

        val_value[-1] = HandwrittenTextRecognition._validate(val_loader, model, criterion, writer, -1, **kwargs)
        for epoch in range(start_epoch, epochs):
            # Train
            train_value[epoch] = HandwrittenTextRecognition._train(train_loader, model, criterion, optimizer, writer, epoch, **kwargs)

            # Validate
            if epoch % validation_interval == 0:
                val_value[epoch] = HandwrittenTextRecognition._validate(val_loader, model, criterion, writer, epoch, **kwargs)
            if decay_lr is not None:
                adjust_learning_rate(lr=lr, optimizer=optimizer, epoch=epoch, decay_lr_epochs=decay_lr)
            best_value = checkpoint(epoch, val_value[epoch], best_value, model, optimizer, current_log_folder)

        # Test
        test_value = HandwrittenTextRecognition._test(test_loader, model, criterion, writer, epochs - 1, **kwargs)
        logging.info('Training completed')

        return train_value, val_value, test_value

    ####################################################################################################################
    """
    These methods delegate their function to other classes in this package. 
    It is useful because sub-classes can selectively change the logic of certain parts only.
    """

    @classmethod
    def _train(cls, train_loader, model, criterion, optimizer, writer, epoch, **kwargs):
        return train.train(train_loader, model, criterion, optimizer, writer, epoch, **kwargs)

    @classmethod
    def _validate(cls, val_loader, model, criterion, writer, epoch, **kwargs):
        return evaluate.validate(val_loader, model, criterion, writer, epoch, **kwargs)

    @classmethod
    def _test(cls, test_loader, model, criterion, writer, epoch, **kwargs):
        return evaluate.test(test_loader, model, criterion, writer, epoch, **kwargs)
