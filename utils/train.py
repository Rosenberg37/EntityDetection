import datetime
import logging
import os
from typing import Optional

import torch
from flair.datasets import DataLoader
from torch import nn, optim
from tqdm import tqdm

import config
import modules
import utils

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
            self,
            model: nn.Module,
            optimizer: optim.Optimizer,
            scheduler: optim.lr_scheduler,
            corpus: modules.FormatCorpus,
            max_grad_norm: float,
            checkpoint_name: str,
            save: bool,
            load: bool,
            epochs: int,
            batch_size: int,
            evaluate: list[str],
    ):
        self.evaluate = evaluate
        self.batch_size = batch_size
        self.epochs = epochs
        self.save = save
        self.max_grad_norm = max_grad_norm
        self.corpus = corpus
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.model = model

        self.checkpoint_path = config.ADVANCED_OPTIONS.root_path + f"/result/{corpus.name}/models/{checkpoint_name}.pth"
        self.best_f1 = 0
        self.analysist = utils.Analysist(corpus, model)

        if load:
            self.load_checkpoint(self.model, self.checkpoint_path, self.optimizer)

    def __call__(self):
        loader = DataLoader(
            dataset=self.corpus.train,
            shuffle=True,
            drop_last=False,
            batch_size=self.batch_size,
        )

        self.model.train()
        for epoch in range(self.epochs):
            t = tqdm(loader, desc=f"Epoch {epoch}")
            losses, runtimes = list(), list()
            for data in t:
                self.optimizer.zero_grad()
                loss = self.model(data)
                loss.backward()

                if self.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)

                self.optimizer.step()
                self.scheduler.step()

                losses.append(loss.item())
                t.set_postfix({'loss': loss.item(), 'avg_loss': sum(losses) / len(losses)})

            runtimes.append(t.format_dict['elapsed'])
            logger.info(f"Epoch {epoch}:\n"
                        f"Average_loss:{sum(losses) / len(losses)}\n"
                        f"Average_runtime:{datetime.timedelta(seconds=sum(runtimes) / len(runtimes))}")

            for name in self.evaluate:

                save_model = False
                if self.save:
                    if name == 'test' and 'dev' not in self.evaluate:
                        save_model = True
                    elif name == 'dev':
                        save_model = True

                statistic = self.analysist.evaluate(name)
                f1 = statistic['all']['f1']

                logger.info(f"{name}:\n"
                            f"Precision:{statistic}\n")

                if save_model and f1 > self.best_f1:
                    self.best_f1 = f1
                    logger.info(f"Save models with best performance on {name}\n")
                    self.save_checkpoint(self.model, self.checkpoint_path, self.optimizer)

    @staticmethod
    def save_checkpoint(model: nn.Module, checkpoint_path: str, optimizer: Optional[optim.Optimizer] = None):
        dir_path = os.path.dirname(checkpoint_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        torch.save({
            'models': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, checkpoint_path, _use_new_zipfile_serialization=False)

    @staticmethod
    def load_checkpoint(model: nn.Module, checkpoint_path: str, optimizer: Optional[optim.Optimizer] = None):
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint['models'], strict=False)
            if len(missing_keys) == 0 and len(unexpected_keys) == 0 and optimizer is not None:
                optimizer.load_state_dict(checkpoint['optimizer'])
            else:
                logger.warning(f'Missing_keys:{missing_keys}\n'
                               f'Unexpected_keys:{unexpected_keys}')
        else:
            raise RuntimeError("Checkpoint not exist!")
