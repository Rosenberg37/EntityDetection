import logging
import os
import pathlib

import flair
import transformers
from torch import nn, optim

import config
import modules


def init_logger(log_file=None):
    log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if log_file and log_file != '':
        os.makedirs(pathlib.Path(log_file).parent, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)

    return logger


def init_environment():
    if config.ADVANCED_OPTIONS.debug:
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    flair.device = config.ADVANCED_OPTIONS.device


def init_model(corpus: modules.FormatCorpus):
    model = modules.EntityDetection(**dict(corpus.metadata, **vars(config.MODEL_OPTIONS)))
    return model.to(config.ADVANCED_OPTIONS.device)


def init_optimizer(model: nn.Module):
    learning_rate = config.OPTIMIZE_OPTIONS.learning_rate
    weight_decay = config.OPTIMIZE_OPTIONS.weight_decay
    named_parameters = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    params_group = [
        {'params': [p for n, p in named_parameters if not any(nd in n for nd in no_decay)], 'weight_decay_rate': weight_decay},
        {'params': [p for n, p in named_parameters if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0}
    ]
    return optim.AdamW(params_group, lr=learning_rate)


def init_scheduler(corpus: modules.FormatCorpus, optimizer: optim.Optimizer):
    steps_epoch = len(corpus) // config.PROCEDURE_OPTIONS.batch_size + 1
    total_steps = steps_epoch * config.PROCEDURE_OPTIONS.epochs
    return transformers.get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.OPTIMIZE_OPTIONS.lr_warmup * total_steps,
        num_training_steps=total_steps
    )
