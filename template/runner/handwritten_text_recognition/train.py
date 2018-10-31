# Utils
import logging
import time
import math

# Torch related stuff
import torch
from tqdm import tqdm

# DeepDIVA
from util.misc import AverageMeter
from util.evaluation.metrics import accuracy

from template.runner.handwritten_text_recognition.text_processing import sample_text, convert_batch_to_sequence, batch_cer, batch_wer

def train(train_loader, model, criterion, optimizer, writer, epoch, no_cuda=False, log_interval=25,
          **kwargs):
    """
    Training routine

    Parameters
    ----------
    train_loader : torch.utils.data.DataLoader
        The dataloader of the train set.
    model : torch.nn.module
        The network model being used.
    criterion : torch.nn.loss
        The loss function used to compute the loss of the model.
    optimizer : torch.optim
        The optimizer used to perform the weight update.
    writer : tensorboardX.writer.SummaryWriter
        The tensorboard writer object. Used to log values on file for the tensorboard visualization.
    epoch : int
        Number of the epoch (for logging purposes).
    no_cuda : boolean
        Specifies whether the GPU should be used or not. A value of 'True' means the CPU will be used.
    log_interval : int
        Interval limiting the logging of mini-batches. Default value of 10.

    Returns
    ----------
    top1.avg : float
        Accuracy of the model of the evaluated split
    """
    multi_run = kwargs['run'] if 'run' in kwargs else None

    # Instantiate the counters
    batch_time = AverageMeter()
    wer_meter = AverageMeter()
    cer_meter = AverageMeter()
    loss_meter = AverageMeter()
    data_time = AverageMeter()

    # Switch to train mode (turn on dropout & stuff)
    model.train()

    # Iterate over whole training set
    end = time.time()
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), unit='batch', ncols=150, leave=False)
    for batch_idx, (input, target, target_len, image_width) in pbar:

        # Measure data loading time
        data_time.update(time.time() - end)

        # Moving data to GPU
        if not no_cuda:
            input = input.cuda(async=True)
            target = target.cuda(async=True)

        # Convert the input and its labels to Torch Variables
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        cer, wer, loss = train_one_mini_batch(model, criterion, optimizer, input_var, target_var, target_len, image_width, cer_meter, wer_meter, loss_meter)
        
        # Add loss and accuracy to Tensorboard
        if multi_run is None:
            writer.add_scalar('train/mb_loss', loss.data[0], epoch * len(train_loader) + batch_idx)
            writer.add_scalar('train/mb_cer', cer, epoch * len(train_loader) + batch_idx)
            writer.add_scalar('train/mb_wer', wer, epoch * len(train_loader) + batch_idx)
        else:
            writer.add_scalar('train/mb_loss_{}'.format(multi_run), loss.data[0], epoch * len(train_loader) + batch_idx)
            writer.add_scalar('train/mb_cer_{}'.format(multi_run), cer, epoch * len(train_loader) + batch_idx)
            writer.add_scalar('train/mb_wer_{}'.format(multi_run), wer, epoch * len(train_loader) + batch_idx)

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Log to console
        if batch_idx % log_interval == 0:
            pbar.set_description('train epoch [{0}][{1}/{2}]\t'.format(epoch, batch_idx, len(train_loader)))

            pbar.set_postfix(Time='{batch_time.avg:.3f}\t'.format(batch_time=batch_time),
                             CER='{cer.avg:.4f}\t'.format(cer=cer_meter),
                             WER='{wer.avg:.4f}\t'.format(wer=wer_meter),
                             Loss='{loss.avg:.4f}\t'.format(loss=loss_meter),
                             Data='{data_time.avg:.3f}\t'.format(data_time=data_time))

    # Logging the epoch-wise accuracy
    if multi_run is None:
        writer.add_scalar('train/cer', cer_meter.avg, epoch)
        writer.add_scalar('train/wer', wer_meter.avg, epoch)
    else:
        writer.add_scalar('train/cer_{}'.format(multi_run), cer_meter.avg, epoch)
        writer.add_scalar('train/wer_{}'.format(multi_run), wer_meter.avg, epoch)

    logging.debug('Train epoch[{}]: '
                  'Loss={loss.avg:.4f}\t'
                  'Batch time={batch_time.avg:.3f} ({data_time.avg:.3f} to load data)'
                  .format(epoch, batch_time=batch_time, data_time=data_time, loss=loss_meter))

    return cer_meter.avg


def train_one_mini_batch(model, criterion, optimizer, input_var, target_var, target_len, image_width, cer_meter, wer_meter, loss_meter):
    """
    This routing train the model passed as parameter for one mini-batch

    Parameters
    ----------
    model : torch.nn.module
        The network model being used.
    criterion : torch.nn.loss
        The loss function used to compute the loss of the model.
    optimizer : torch.optim
        The optimizer used to perform the weight update.
    input_var : torch.autograd.Variable
        The input data for the mini-batch
    target_var : torch.autograd.Variable
        The target data (labels) for the mini-batch
    loss_meter : AverageMeter
        Tracker for the overall loss

    Returns
    -------
    loss : float
        Loss for this mini-batch
    """
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
    
    cer_meter.update(cer, input_var.size(0))
    wer_meter.update(wer, input_var.size(0))
    
    loss = criterion(acts, labels, acts_len, labels_len)
    
    loss_meter.update(loss.data[0], len(input_var))
        
    # Reset gradient
    optimizer.zero_grad()
    # Compute gradients
    loss.backward()
    
    #for name, p in model.named_parameters():
    #    logging.info("Param " + name + " ; Data: " + str(torch.norm(p.data)) + " ; Grad: " + str(torch.norm(p.grad)))
    # Add gradient cliping ?
    """
    clip = 0.1
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    """
    
    # Perform a step by updating the weights
    optimizer.step()

    return cer, wer, loss
