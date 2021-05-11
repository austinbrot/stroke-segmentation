import logging

from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data.dataloader import DataLoader

from utils import dsc

# configure module logging for terminal
terminal_logger = logging.getLogger(__name__)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)

terminal_logger.addHandler(ch)



class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 32

    # checkpoint settings
    ckpt_path = 'ckpts'
    save_epochs = 5
    max_ckpts = 5

    log_frequency = 10 # iterations
    vis_frequency = 10 # epochs
    vis_images = 200 # number of visualization images to save in log file

    num_workers = 0 # for DataLoader
    eval_every = 1 # num epochs between evaluations

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class Trainer:

    def __init__(self, model, optimizer, loss, config, train_dataset, val_dataset, test_dataset, logger=None):
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.config = config
        self.logger = logger

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

        self.iteration = 0
        self.train_loss = []

        self.best_loss = float('inf')
        self.best_ckpt = None

        # take over whatever gpus are on the system
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)

    def save_checkpoint(self, prefix=''):
        config, model = self.config, self.model
        if self.config.ckpt_path is not None:
            path = config.ckpt_path + '/' + prefix + ('_' if prefix else '') + 'model'
            ckpt_model = model.module if hasattr(model, "module") else model
            terminal_logger.info("saving %s", path)
            torch.save(ckpt_model.state_dict(), path)

    def log_loss_summary(self, loss, it, prefix=""):
        logger = self.logger
        if logger:
            logger.scalar_summary(prefix + "_loss", np.mean(loss), it + 1)

    def worker_init(worker_id):
        np.random.seed(42 + worker_id)

    def train_epoch(self, epoch):
        model, optimizer, loss_fn, data, config = self.model, self.optimizer, self.loss, self.train_dataset, self.config
        model.train(True)
        loader = DataLoader(
            data,
            batch_size=config.batch_size,
            drop_last=True,
            num_workers=config.num_workers,
            worker_init_fn=self.worker_init,
        )

        pbar = tqdm(enumerate(loader), total=len(loader))
        for it, (x, y) in pbar:
            x, y = x.to(self.device), y.to(self.device)

            with torch.set_grad_enabled(True):
                preds = model(x)
                loss = loss_fn(preds, y)

                self.train_loss.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            if (self.iteration + 1) % config.log_frequency == 0:
                self.log_loss_summary(self.train_loss, self.iteration + 1)
                self.train_loss = []

            self.iteration += 1
            pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}.")


    def dsc_per_volume(self, validation_pred, validation_true, patient_slice_index):
        dsc_list = []
        num_slices = np.bincount([p[0] for p in patient_slice_index])
        index = 0
        for p in range(len(num_slices)):
            y_pred = np.array(validation_pred[index : index + num_slices[p]])
            y_true = np.array(validation_true[index : index + num_slices[p]])
            dsc_list.append(dsc(y_pred, y_true))
            index += num_slices[p]
        return dsc_list


    def test_epoch(self, data, split):
        model, loss_fn, config = self.model, self.loss, self.config
        model.train(False)
        loader = DataLoader(
            data,
            batch_size=config.batch_size,
            drop_last=False,
            num_workers=config.num_workers,
            worker_init_fn=self.worker_init,
        )

        epoch_pred, epoch_true = [], []
        losses = []

        pbar = tqdm(loader)
        for x, y in pbar:
            x, y = x.to(self.device), y.to(self.device)

            with torch.set_grad_enabled(False):
                preds = model(x)
                loss = loss_fn(preds, y)

                losses.append(loss.item())

                y_pred_np = preds.detach().cpu().numpy()
                epoch_pred.extend(
                    [y_pred_np[s] for s in range(y_pred_np.shape[0])]
                )
                y_true_np = y.detach().cpu().numpy()
                epoch_true.extend(
                    [y_true_np[s] for s in range(y_true_np.shape[0])]
                )

        mean_dsc = np.mean(
            self.dsc_per_volume(epoch_pred, epoch_true, data.patient_slice_index)
        )

        mean_loss = np.mean(losses)

        if split == 'valid':
            self.log_loss_summary(losses, self.iteration + 1, prefix="val")
            self.logger.scalar_summary("val_dsc", mean_dsc, self.iteration + 1)
        
        terminal_logger.info(f"{split} dsc: {mean_dsc}, loss: {mean_loss}")
        print(f"{split} dsc: {mean_dsc}, loss: {mean_loss}")
        return mean_dsc, mean_loss


    def train(self):
        config, val_data, test_data = self.config, self.val_dataset, self.test_dataset

        for epoch in range(config.max_epochs):

            self.train_epoch(epoch)

            if (epoch + 1) % self.config.eval_every == 0:
                dsc, loss = self.test_epoch(val_data, 'val')

                if (epoch + 1) % config.save_epochs == 0:
                    self.save_checkpoint()
                    if loss < self.best_loss:
                        terminal_logger.info(f'New best checkpoint at epoch {epoch}')
                        print(f'New best checkpoint at epoch {epoch}')
                        self.save_checkpoint(prefix='best')
                        self.best_loss = loss

        if val_data is not None:
            dsc, loss = self.test_epoch(val_data, 'val')
            if loss < self.best_loss:
                terminal_logger.info(f'New best checkpoint at end')
                print(f'New best checkpoint at end')
                self.save_checkpoint(prefix='best')
                self.best_loss = loss


        if test_data is not None:
            self.test_epoch(test_data, 'test')