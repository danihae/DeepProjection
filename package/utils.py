import glob
import os
import re
import shutil

import numpy as np
import tifffile
import torch
from torch import nn as nn
from tqdm import tqdm


def save_as_tif(imgs, filename):
    imgs = imgs - np.nanmin(imgs)
    imgs /= np.nanmax(imgs)
    imgs *= 255
    imgs = imgs.astype('uint8')
    if len(imgs.shape) == 3:  # if stack
        with tifffile.TiffWriter(filename) as tiff:
            for img in imgs:
                tiff.save(img)
    elif len(imgs.shape) == 2:  # if single image
        tifffile.imsave(filename, imgs)
    print('Saving prediction results as %s' % filename)


def get_stack_directories(base_folder, signature='.tif', n_min=40):
    """ Get paths of directories containing {signature} """
    paths = []
    for root, dirs, files in os.walk(base_folder, topdown=False):
        for name in files:
            if signature in name:
                if len(files) >= n_min:
                    paths.append(root + '/')
    return paths


def convert_to_stack(folder_in, folder_out, invert_order=False):
    """ Parse single image data from 3D stacks and convert to 2D stacks for each time point.
        If data is already in stacks, data is only copied to folder_out """
    os.makedirs(folder_out, exist_ok=True)
    files = np.asarray([file for file in glob.glob(folder_in + '*') if 'txt' not in file])
    shape = tifffile.imread(files[0]).shape
    # if len(shape) == 4:  # two color stack

    if len(shape) == 3:  # if files are stacks
        for file_i in files:
            t = int(re.findall('[0-9]+.TIF', file_i)[0][:-4]) - 1
            shutil.copy(file_i, folder_out + f'stack_{t}.tif')
        n_frames, n_slices, n_pixel = len(files), shape[0], shape[1:]
    elif len(shape) == 2:  # if files are single images
        # get # frames and stacks and create array
        ts, zs = [], []
        for file_i in files:
            if 'Default' in file_i:
                t = int(re.findall('img_[0-9]+', file_i)[0][4:])
                z = int(re.findall('Default_[0-9]+', file_i)[0][8:])
            else:
                try:
                    t = int(re.findall('t[0-9]+', file_i)[0][1:])
                except:
                    t = int(re.findall('time[0-9]+', file_i)[0][4:])
                z = int(re.findall('z[0-9]+', file_i)[0][1:])
            ts.append(t)
            zs.append(z)
        ts, zs = np.asarray(ts), np.asarray(zs)
        # if counting starts not from zero, subtract offset
        if 0 not in ts:
            ts -= np.min(ts)
        if 0 not in zs:
            zs -= np.min(zs)
        n_frames, n_slices, n_pixel = len(np.unique(ts)), len(np.unique(zs)), shape
        # load data and save as stacks
        for t in range(n_frames):
            stack_t = np.zeros((len(np.unique(zs)), *shape), dtype='uint16')
            files_t = files[ts == t]
            z_t = zs[ts == t]
            for z in range(n_slices):
                file_z = files_t[z_t == z]
                stack_t[z] = tifffile.imread(file_z)
            if invert_order:
                stack_t = stack_t[::-1]
            tifffile.imsave(folder_out + f'stack_{t}.tif', stack_t)
    else:
        raise FileNotFoundError('Data structure unknown!')

    return n_frames, n_slices, n_pixel


class MaxProjection:
    """maximum intensity projection w/o normalization"""

    def __init__(self, folder, filename_output=None):
        self.files = np.asarray([file for file in glob.glob(folder + '*') if 'txt' not in file])
        self.shape = tifffile.imread(self.files[0]).shape
        self.get_data_structure()
        self.max_projection()
        # save movie
        if filename_output is None:
            filename_output = folder[:-1] + '_max_int'
        if self.data_mode == 'two_color':
            tifffile.imsave(filename_output + '_c0.tif', self.arr_max_proj_c0)
            tifffile.imsave(filename_output + '_c1.tif', self.arr_max_proj_c1)
            del self.arr_max_proj_c0, self.arr_max_proj_c1
        else:
            tifffile.imsave(filename_output + '.tif', self.arr_max_proj)
            del self.arr_max_proj

    def get_data_structure(self):
        if len(self.shape) == 3:  # if files are stacks
            self.data_mode = 'stacks'
            self.n_frames, self.n_slices, self.n_pixel = len(self.files), self.shape[0], self.shape[1:]
        elif len(self.shape) == 2:  # if files are single images
            self.data_mode = 'images'
            # get # frames and stacks and create array
            ts, zs, cs = [], [], []
            for file_i in self.files:
                try:
                    t = int(re.findall('t[0-9]+', file_i)[0][1:])
                except:
                    t = int(re.findall('time[0-9]+', file_i)[0][4:])
                z = int(re.findall('z[0-9]+', file_i)[0][1:])
                c = int(re.findall('channel[0-9]+', file_i)[0][7:])
                ts.append(t)
                zs.append(z)
                cs.append(c)
            self.ts, self.zs, self.cs = np.asarray(ts), np.asarray(zs), np.asarray(cs)
            # if counting starts not from zero, subtract offset
            if 0 not in self.ts:
                self.ts -= np.min(self.ts)
            if 0 not in self.zs:
                self.zs -= np.min(self.zs)
            self.n_frames = len(np.unique(self.ts))
            self.n_slices = len(np.unique(self.zs))
            self.n_channels = len(np.unique(self.cs))
            self.n_pixel = self.shape
            # check if two-color
            if self.n_channels == 2:
                self.data_mode = 'two_color'
        else:
            raise FileNotFoundError('Data structure unknown!')

    def max_projection(self):
        # iterate through files and max projection
        if self.data_mode == 'stacks':
            self.arr_max_proj = np.zeros((self.n_frames, self.n_pixel[0], self.n_pixel[1]), dtype='uint16')
            for file_i in tqdm(self.files):
                t = int(re.findall('[0-9]+.TIF', file_i)[0][:-4]) - 1
                self.arr_max_proj[t] = np.max(tifffile.imread(file_i), axis=0)
        if self.data_mode == 'images':
            self.arr_max_proj = np.zeros((self.n_frames, self.n_pixel[0], self.n_pixel[1]), dtype='uint16')
            for t in tqdm(range(self.n_frames)):
                stack_t = np.zeros((self.n_slices, *self.n_pixel), dtype='uint16')
                files_t = self.files[self.ts == t]
                z_t = self.zs[self.ts == t]
                for z in range(self.n_slices):
                    file_z = files_t[z_t == z]
                    stack_t[z] = tifffile.imread(file_z)
                self.arr_max_proj[t] = np.max(stack_t, axis=0)
        if self.data_mode == 'two_color':
            self.arr_max_proj_c0 = np.zeros((self.n_frames, self.n_pixel[0], self.n_pixel[1]), dtype='uint16')
            self.arr_max_proj_c1 = np.zeros((self.n_frames, self.n_pixel[0], self.n_pixel[1]), dtype='uint16')
            for t in tqdm(range(self.n_frames)):
                stack_c0_t = np.zeros((self.n_slices, *self.n_pixel), dtype='uint16')
                stack_c1_t = np.zeros((self.n_slices, *self.n_pixel), dtype='uint16')
                files_t = self.files[self.ts == t]
                z_t = self.zs[self.ts == t]
                c_t = self.cs[self.ts == t]
                for z in range(self.n_slices):
                    file_c0_z = files_t[(z_t == z) & (c_t == 0)]
                    file_c1_z = files_t[(z_t == z) & (c_t == 1)]
                    stack_c0_t[z] = tifffile.imread(file_c0_z)
                    stack_c1_t[z] = tifffile.imread(file_c1_z)
                self.arr_max_proj_c0[t] = np.max(stack_c0_t, axis=0)
                self.arr_max_proj_c1[t] = np.max(stack_c1_t, axis=0)


class BCEDiceLoss(nn.Module):
    def __init__(self, alpha, beta):
        super(BCEDiceLoss, self).__init__()
        self.bce = BCELoss2d()
        self.dice = SoftDiceLoss()
        self.alpha = alpha
        self.beta = beta

    def forward(self, inputs, targets):
        return self.alpha * self.bce(inputs, targets) + self.beta * self.dice(inputs, targets)


class BCELoss2d(nn.Module):
    def __init__(self, weight=None, reduction='mean', **kwargs):
        super(BCELoss2d, self).__init__()
        self.bce_loss = nn.BCELoss(weight, reduction=reduction)

    def forward(self, inputs, targets):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        return self.bce_loss(inputs, targets)


class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=0.5):
        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()

        dice = 2. * (intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        return 1 - dice


class TverskyLoss(nn.Module):
    """
    Salehi, S. S. M., Erdogmus, D. & Gholipour, A. Tversky loss function for image segmentation using 3D fully
    convolutional deep networks. Arxiv (2017).
    """
    def __init__(self, alpha=0.5, beta=0.5, smooth=1):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, inputs, targets):
        #  inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()

        Tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)

        return 1 - Tversky


class logcoshTverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, smooth=1):
        super(logcoshTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, inputs, targets):
        #  inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()

        Tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)

        return torch.log(torch.cosh(1 - Tversky))
