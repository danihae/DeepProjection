from datetime import datetime

import kornia
import torch.optim as optim
from barbar import Bar
from torch.utils.data import DataLoader, random_split

from .utils import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Trainer:
    """
    Class for training of neural network for surface projection
    """
    def __init__(self, dataset, num_epochs, network, mode='binary_mask',
                 n_slices=20, batch_size=4, n_filter=32, lr=1e-4, val_split=0.25, save_dir='./trained_networks/',
                 load_weights=None, save_iterations=False, weights_edge_loss=(1, 0), loss_func='BCEDiceLoss',
                 loss_params=(1, 1)):
        """
        Parameters

        ----------
        dataset
            Training data set (object of DataProcess class)
        num_epochs : int
            Number of training epochs
        network
            Neural network class
        mode : str = 'binary_mask'
            Training mode for loss calculation ('binary_mask' or 'max_projection')
        n_slices : int
            Number of z slices
        batch_size : int
            Batch size
        n_filter : int
            Number of convolutional filters
        lr : float
            Learning rate
        val_split : float
            Validation split
        save_dir : str
            Path to save network weights
        load_weights : str, optional
            If not None, load pretrained weights prior to training in path "load_weights"
        save_iterations : bool
            If True, weight of each epoch are save in save_dir
        weights_edge_loss : tuple(float, float) = (1, 0)
            Ratio between edge loss and area loss (not clear if useful)
        loss_func : str
            Loss function ('BCEDiceLoss', 'TverskyLoss', 'logcoshTverskyLoss')
        loss_params : tuple(float, float)
            Loss parameters, depending on loss function (for logcoshTverskyLoss/TverskyLoss = alpha, beta)
        """
        self.model = network(n_filter=n_filter).to(device)
        self.dataset = dataset
        self.n_filter = n_filter
        self.num_epochs = num_epochs
        self.mode = mode
        self.loss_func = loss_func
        self.loss_params = loss_params
        self.weights_edge_loss = weights_edge_loss
        self.save_iterations = save_iterations
        self.n_slices = n_slices
        self.batch_size = batch_size
        self.lr = lr
        self.best_loss = torch.tensor(float('inf'))
        # split training and validation data
        num_val = int(len(dataset) * val_split)
        num_train = len(dataset) - num_val
        self.dim = dataset.dim_out
        train_data, val_data = random_split(dataset, [num_train, num_val])
        self.train_loader = DataLoader(train_data, batch_size=self.batch_size, pin_memory=True, drop_last=True)
        self.val_loader = DataLoader(val_data, batch_size=self.batch_size, pin_memory=True, drop_last=True)
        if self.loss_func == 'BCEDiceLoss':
            self.criterion = BCEDiceLoss(self.loss_params[0], self.loss_params[1])
        elif self.loss_func == 'TverskyLoss':
            self.criterion = TverskyLoss(self.loss_params[0], self.loss_params[1])
        elif self.loss_func == 'logcoshTverskyLoss':
            self.criterion = logcoshTverskyLoss(self.loss_params[0], self.loss_params[1])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=5)
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.loss = np.zeros(num_epochs) * np.nan
        if load_weights is not None:
            self.state = torch.load(load_weights)
            self.model.load_state_dict(self.state['state_dict'])
            self.epoch = self.state['epoch']
        else:
            self.epoch = 0

    def __iterate(self, mode):
        if mode == 'train':
            print('\nStarting training epoch %s ...' % self.epoch)
            for i, batch_i in enumerate(Bar(self.train_loader)):
                x_i = batch_i['image'].view(self.batch_size, 1, self.n_slices, self.dim[0], self.dim[1]).to(device)
                y_i = batch_i['mask'].view(self.batch_size, 1, self.n_slices, self.dim[0], self.dim[1]).to(device)
                mult = torch.mul(y_i, x_i)
                max_i = torch.max(mult, 2)[0]
                edge = kornia.sobel(y_i.view(self.batch_size, self.n_slices, self.dim[0], self.dim[1])).to(device)

                # Forward pass: Compute predicted y by passing x to the model
                y_pred, mask_pred, edge_pred = self.model(x_i)

                # Compute and print loss
                if self.mode == 'binary_mask':
                    loss = self.criterion(mask_pred, y_i)
                elif self.mode == 'max_projection':
                    loss = self.criterion(y_pred, max_i)
                elif self.mode == 'edge':
                    mask_pred = mask_pred.view(self.batch_size, self.n_slices, self.dim[0], self.dim[1])
                    y_i = y_i.view(self.batch_size, self.n_slices, self.dim[0], self.dim[1])
                    edge_pred = edge_pred.view(self.batch_size, self.n_slices, self.dim[0], self.dim[1])
                    edge = edge.view(self.batch_size, self.n_slices, self.dim[0], self.dim[1])
                    loss = self.weights_edge_loss[0] * self.criterion(mask_pred, y_i) + \
                           self.weights_edge_loss[1] * self.criterion(edge_pred, edge)
                else:
                    raise ValueError(f'Unknown mode {self.mode}.')

                # Zero gradients, perform a backward pass, and update the weights.
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        elif mode == 'val':
            loss_list = []
            print('\nStarting validation epoch %s ...' % self.epoch)
            with torch.no_grad():
                for i, batch_i in enumerate(Bar(self.val_loader)):
                    x_i = batch_i['image'].view(self.batch_size, 1, self.n_slices, self.dim[0], self.dim[1]).to(device)
                    y_i = batch_i['mask'].view(self.batch_size, 1, self.n_slices, self.dim[0], self.dim[1]).to(device)
                    mult = torch.mul(y_i, x_i)
                    max_i = torch.max(mult, 2)[0]
                    edge = kornia.sobel(y_i.view(self.batch_size, self.n_slices, self.dim[0], self.dim[1])).to(device)

                    # Forward pass: Compute predicted y by passing x to the model
                    y_pred, mask_pred, edge_pred = self.model(x_i)

                    # Compute and print loss
                    if self.mode == 'binary_mask':
                        loss = self.criterion(mask_pred, y_i)
                    elif self.mode == 'max_projection':
                        loss = self.criterion(y_pred, max_i)
                    elif self.mode == 'edge':
                        mask_pred = mask_pred.view(self.batch_size, self.n_slices, self.dim[0], self.dim[1])
                        y_i = y_i.view(self.batch_size, self.n_slices, self.dim[0], self.dim[1])
                        edge_pred = edge_pred.view(self.batch_size, self.n_slices, self.dim[0], self.dim[1])
                        edge = edge.view(self.batch_size, self.n_slices, self.dim[0], self.dim[1])
                        loss = self.weights_edge_loss[0] * self.criterion(mask_pred, y_i) + \
                               self.weights_edge_loss[1] * self.criterion(edge_pred, edge)

                    loss_list.append(loss.to('cpu'))
            val_loss = torch.stack(loss_list).mean()
            return val_loss

        torch.cuda.empty_cache()

    def start(self):
        """
        Start training of DeepProjection
        """
        for epoch in range(self.num_epochs):
            self.__iterate('train')
            self.state = {
                'epoch': self.epoch,
                'loss': self.loss,
                'best_loss': self.best_loss,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'loss_func': self.loss_func,
                'loss_params': self.loss_params,
                'n_filter': self.n_filter,
                'datetime': datetime.now(),
                'lr': self.lr,
                'weight_edge_loss': self.weights_edge_loss,
                'aug_factor': self.dataset.aug_factor,
                'brightness_contrast': self.dataset.brightness_contrast,
                'noise_amp': self.dataset.noise_amp,
                'random_flip_z': self.dataset.random_flip_z,
                'padding_mode': self.dataset.padding_mode,
                'clip_thrs': self.dataset.clip_thrs,
                'n_slices': self.dataset.n_slices,
                'mode': self.dataset.mode,
            }
            val_loss = self.__iterate('val')
            self.state['loss'][epoch] = val_loss.cpu().numpy()
            self.scheduler.step(val_loss)
            if val_loss < self.best_loss:
                print('\nValidation loss improved from %s to %s - saving model state' % (
                    round(self.best_loss.item(), 5), round(val_loss.item(), 5)))
                self.state['best_loss'] = self.best_loss = val_loss
                torch.save(self.state, self.save_dir + f'/model_best.pth')
            if self.save_iterations:
                torch.save(self.state, self.save_dir + f'/model_e{self.epoch}.pth')
            self.epoch += 1
