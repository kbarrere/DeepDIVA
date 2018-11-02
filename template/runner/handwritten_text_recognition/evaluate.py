# Utils
import logging
import time
import warnings

import numpy as np
# Torch related stuff
import torch
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

from util.evaluation.metrics import accuracy
# DeepDIVA
from util.misc import AverageMeter, _prettyprint_logging_label, save_image_and_log_to_tensorboard
from util.visualization.confusion_matrix_heatmap import make_heatmap

from template.runner.handwritten_text_recognition.text_processing import sample_text, convert_batch_to_sequence, batch_cer, batch_wer


def validate(val_loader, model, criterion, writer, epoch, no_cuda=False, log_interval=20, **kwargs):
    """Wrapper for _evaluate() with the intent to validate the model."""
    return _evaluate(val_loader, model, criterion, writer, epoch, 'val', no_cuda, log_interval, **kwargs)


def test(test_loader, model, criterion, writer, epoch, no_cuda=False, log_interval=20, **kwargs):
    """Wrapper for _evaluate() with the intent to test the model"""
    return _evaluate(test_loader, model, criterion, writer, epoch, 'test', no_cuda, log_interval, **kwargs)


def _evaluate(data_loader, model, criterion, writer, epoch, logging_label, no_cuda=False, log_interval=10, **kwargs):
    """
    The evaluation routine

    Parameters
    ----------
    data_loader : torch.utils.data.DataLoader
        The dataloader of the evaluation set
    model : torch.nn.module
        The network model being used
    criterion: torch.nn.loss
        The loss function used to compute the loss of the model
    writer : tensorboardX.writer.SummaryWriter
        The tensorboard writer object. Used to log values on file for the tensorboard visualization.
    epoch : int
        Number of the epoch (for logging purposes)
    logging_label : string
        Label for logging purposes. Typically 'test' or 'valid'. Its prepended to the logging output path and messages.
    no_cuda : boolean
        Specifies whether the GPU should be used or not. A value of 'True' means the CPU will be used.
    log_interval : int
        Interval limiting the logging of mini-batches. Default value of 10.

    Returns
    -------
    top1.avg : float
        Accuracy of the model of the evaluated split
    """
    multi_run = kwargs['run'] if 'run' in kwargs else None

    # Instantiate the counters
    batch_time = AverageMeter()
    wers = AverageMeter()
    cers = AverageMeter()
    losses = AverageMeter()
    data_time = AverageMeter()

    # Switch to evaluate mode (turn off dropout & such )
    model.eval()

    # Iterate over whole evaluation set
    end = time.time()

    # Empty lists to store the predictions and target values
    preds = []
    targets = []

    pbar = tqdm(enumerate(data_loader), total=len(data_loader), unit='batch', ncols=150, leave=False)
    for batch_idx, (input, target, target_len, image_width) in pbar:
        
        # Measure data loading time
        data_time.update(time.time() - end)

        # Moving data to GPU
        if not no_cuda:
            input = input.cuda(async=True)
            target = target.cuda(async=True)

        # Convert the input and its labels to Torch Variables
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # Compute output
        output = model(input_var)
        
        # Compute and record the loss
        batch_size = len(output)
        
        acts = output.transpose(0, 1).contiguous()
        
        labels = target_var.view(-1)
        labels = labels.type(torch.IntTensor)
        labels = labels[labels.nonzero()] # Remove padding
        labels = labels.view(-1)
        
        # Only use activation before zero padding
        image_width = image_width.type(torch.IntTensor)
        acts_len = ((image_width - 2) // 2 - 2) // 2 - 5
        
        labels_len = target_len.type(torch.IntTensor)
        
        probs = output.clone()
        probs = probs.detach()
        
        # Computes CER and WER
        predictions = sample_text(probs, acts_len=acts_len)
        references = convert_batch_to_sequence(target_var)
        cer = batch_cer(predictions, references)
        wer = batch_wer(predictions, references)
        
        cers.update(cer, input.size(0))
        wers.update(wer, input.size(0))
        
        # Temp
        if batch_idx == 0:
            logging.info("Predicted Sequence: " + str(predictions))
            logging.info("True labels: " + str(references))
            logging.info("CER: " + str(cer))
            logging.info("WER: " + str(wer))
        
        loss = criterion(acts, labels, acts_len, labels_len)
        
        losses.update(loss.data[0], input.size(0))

        # Get the predictions
        _ = [preds.append(item) for item in [np.argmax(item) for item in output.data.cpu().numpy()]]
        _ = [targets.append(item) for item in target.cpu().numpy()]

        # Add loss and accuracy to Tensorboard
        if multi_run is None:
            writer.add_scalar(logging_label + '/mb_loss', loss.data[0], epoch * len(data_loader) + batch_idx)
            writer.add_scalar(logging_label + '/mb_cer', cer, epoch * len(data_loader) + batch_idx)
            writer.add_scalar(logging_label + '/mb_wer', wer, epoch * len(data_loader) + batch_idx)
        else:
            writer.add_scalar(logging_label + '/mb_loss_{}'.format(multi_run), loss.data[0], epoch * len(data_loader) + batch_idx)
            writer.add_scalar(logging_label + '/mb_cer_{}'.format(multi_run), cer, epoch * len(data_loader) + batch_idx)
            writer.add_scalar(logging_label + '/mb_wer_{}'.format(multi_run), wer, epoch * len(data_loader) + batch_idx)

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % log_interval == 0:
            pbar.set_description(logging_label +
                                 ' epoch [{0}][{1}/{2}]\t'.format(epoch, batch_idx, len(data_loader)))

            pbar.set_postfix(Time='{batch_time.avg:.3f}\t'.format(batch_time=batch_time),
                             CER='{cer.avg:.4f}\t'.format(cer=cers),
                             WER='{wer.avg:.4f}\t'.format(wer=wers),
                             Loss='{loss.avg:.4f}\t'.format(loss=losses),
                             Data='{data_time.avg:.3f}\t'.format(data_time=data_time))

    logging.info(_prettyprint_logging_label(logging_label) +
                 ' epoch[{}]: '
                 'CER={cer.avg:.4f}\t'
                 'WER={wer.avg:.4f}\t'
                 'Loss={loss.avg:.4f}\t'
                 'Batch time={batch_time.avg:.3f} ({data_time.avg:.3f} to load data)'
                 .format(epoch, batch_time=batch_time, data_time=data_time, cer=cers, wer=wers, loss=losses))

    # Generate a classification report for each epoch
    #_log_classification_report(data_loader, epoch, preds, targets, writer)

    return cers.avg


def _log_classification_report(data_loader, epoch, preds, targets, writer):
    """
    This routine computes and prints on Tensorboard TEXT a classification
    report with F1 score, Precision, Recall and similar metrics computed
    per-class.

    Parameters
    ----------
    data_loader : torch.utils.data.DataLoader
        The dataloader of the evaluation set
    epoch : int
        Number of the epoch (for logging purposes)
    preds : list
        List of all predictions of the model for this epoch
    targets : list
        List of all correct labels for this epoch
    writer : tensorboardX.writer.SummaryWriter
        The tensorboard writer object. Used to log values on file for the tensorboard visualization.

    Returns
    -------
        None
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        classification_report_string = str(classification_report(y_true=targets,
                                                                 y_pred=preds,
                                                                 target_names=[str(item) for item in
                                                                               data_loader.dataset.classes]))
    # Fix for TB writer. Its an ugly workaround to have it printed nicely in the TEXT section of TB.
    classification_report_string = classification_report_string.replace('\n ', '\n\n       ')
    classification_report_string = classification_report_string.replace('precision', '      precision', 1)
    classification_report_string = classification_report_string.replace('avg', '      avg', 1)

    writer.add_text('Classification Report for epoch {}\n'.format(epoch), '\n' + classification_report_string, epoch)
