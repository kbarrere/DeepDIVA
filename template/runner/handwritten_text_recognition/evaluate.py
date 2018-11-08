# Utils
import logging
import time

import numpy as np

# Torch related stuff
import torch
from tqdm import tqdm

# DeepDIVA
from util.misc import AverageMeter, _prettyprint_logging_label

from template.runner.handwritten_text_recognition.text_processing import sample_text, convert_batch_to_sequence, batch_cer, batch_wer


def validate(val_loader, model, criterion, writer, epoch, dictionnary_name, no_cuda, decode_val, log_interval=20, **kwargs):
    """Wrapper for _evaluate() with the intent to validate the model."""
    return _evaluate(val_loader, model, criterion, writer, epoch, dictionnary_name, 'val', no_cuda, decode_val,log_interval, **kwargs)


def test(test_loader, model, criterion, writer, epoch, dictionnary_name, no_cuda, log_interval=20, **kwargs):
    """Wrapper for _evaluate() with the intent to test the model"""
    return _evaluate(test_loader, model, criterion, writer, epoch, dictionnary_name, 'test', no_cuda, True,log_interval, **kwargs)


def _evaluate(data_loader, model, criterion, writer, epoch, dictionnary_name, logging_label, no_cuda, decode, log_interval=10, **kwargs):
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
    cers.avg : float
        Character Error Rate of the model of the evaluated split
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
        
        # TODO: use models attributes ?
        acts_len = ((image_width - 2) // 2 - 2) // 2 - 5
        
        labels_len = target_len.type(torch.IntTensor)
        
        probs = output.clone()
        probs = probs.detach()
        
        # Computes CER and WER
        if decode:
            predictions = sample_text(probs, acts_len=acts_len, dictionnary_name=dictionnary_name)
            references = convert_batch_to_sequence(target_var, dictionnary_name=dictionnary_name)
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

        # Add loss, CER and WER to Tensorboard
        if multi_run is None:
            writer.add_scalar(logging_label + '/mb_loss', loss.data[0], epoch * len(data_loader) + batch_idx)
            if decode:
                writer.add_scalar(logging_label + '/mb_cer', cer, epoch * len(data_loader) + batch_idx)
                writer.add_scalar(logging_label + '/mb_wer', wer, epoch * len(data_loader) + batch_idx)
        else:
            writer.add_scalar(logging_label + '/mb_loss_{}'.format(multi_run), loss.data[0], epoch * len(data_loader) + batch_idx)
            if decode:
                writer.add_scalar(logging_label + '/mb_cer_{}'.format(multi_run), cer, epoch * len(data_loader) + batch_idx)
                writer.add_scalar(logging_label + '/mb_wer_{}'.format(multi_run), wer, epoch * len(data_loader) + batch_idx)

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % log_interval == 0:
            pbar.set_description(logging_label +
                                 ' epoch [{0}][{1}/{2}]\t'.format(epoch, batch_idx, len(data_loader)))

            if decode:
                pbar.set_postfix(Time='{batch_time.avg:.3f}\t'.format(batch_time=batch_time),
                                 CER='{cer.avg:.4f}\t'.format(cer=cers),
                                 WER='{wer.avg:.4f}\t'.format(wer=wers),
                                 Loss='{loss.avg:.4f}\t'.format(loss=losses),
                                 Data='{data_time.avg:.3f}\t'.format(data_time=data_time))
            else:
                pbar.set_postfix(Time='{batch_time.avg:.3f}\t'.format(batch_time=batch_time),
                                 Loss='{loss.avg:.4f}\t'.format(loss=losses),
                                 Data='{data_time.avg:.3f}\t'.format(data_time=data_time))

    if decode:
        logging.info(_prettyprint_logging_label(logging_label) +
                     ' epoch[{}]: '
                     'CER={cer.avg:.4f}\t'
                     'WER={wer.avg:.4f}\t'
                     'Loss={loss.avg:.4f}\t'
                     'Batch time={batch_time.avg:.3f} ({data_time.avg:.3f} to load data)'
                     .format(epoch, batch_time=batch_time, data_time=data_time, cer=cers, wer=wers, loss=losses))
    else:
        logging.info(_prettyprint_logging_label(logging_label) +
                     ' epoch[{}]: '
                     'Loss={loss.avg:.4f}\t'
                     'Batch time={batch_time.avg:.3f} ({data_time.avg:.3f} to load data)'
                     .format(epoch, batch_time=batch_time, data_time=data_time, loss=losses))

    # Logging the epoch-wise CER and WER
    if multi_run is None:
        if decode:
            writer.add_scalar('train/cer', cers.avg, epoch)
            writer.add_scalar('train/wer', wers.avg, epoch)
    else:
        if decode:
            writer.add_scalar('train/cer_{}'.format(multi_run), cers.avg, epoch)
            writer.add_scalar('train/wer_{}'.format(multi_run), wers.avg, epoch)

    if decode:
        return cers.avg
    return losses.avg
