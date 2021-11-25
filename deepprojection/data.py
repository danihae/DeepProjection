import glob
import os
import shutil
from random import choice

import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset
from albumentations import (Compose, Flip, GaussNoise, RandomBrightnessContrast, RandomRotate90)

torch.set_default_dtype(torch.float32)
torch.cuda.empty_cache()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class DataProcess(Dataset):
    """
    Class for data preparation and pre-processing for network training
    """
    def __init__(self, source_dirs, dim_out=(128, 128), n_slices=20, aug_factor=10, noise_amp=10,
                 brightness_contrast=(0.15, 0.15), random_flip_z=False, padding_mode='edge', clip_thrs=(0.0, 99.95),
                 mode='binary_mask', data_path='./data/', create=False):
        """
        Modifies and augments training data, and then a training data object for trainer

        Parameters
        ----------
        source_dirs : tuple(str, str)
            Training data directories [raw_stacks_dir, masked_stacks_dir]
        dim_out : tuple(int, int)
            Resize dimensions (has to be divisable by 8)
        n_slices : int
            Number of slices per stack (needs to be larger than maximal slice number in data set, stacks smaller are
            zero-padded)
        aug_factor : int
            Factor by which data is augmented
        noise_amp : float
            Amplitude of noise for augmentation
        brightness_contrast : tuple(float, float)
            Factors for augmentation of [brightness, contrast]
        random_flip_z : bool
            If True, the stacks are randomly flipped in z-direction
        padding_mode : str, default 'edge'
            Padding mode for z-padding ('edge', 'constant', 'mirror')
        clip_thrs : tuple(float, float)
            Clip thresholds for intensity normalization [lower, upper] in percentiles
        mode : str
            Training mode for loss calculation ('binary_mask' or 'max_projection')
        data_path : str
            Path for training data tree (modified data with augmentations etc)
        create : bool
            If False, training data set (tiling, augmentation) is not created, the existing set in data_path is used.
        """
        self.data_path = data_path

        self.aug_mask_path = self.data_path + '/augmentation/aug_mask/'
        self.aug_input_path = self.data_path + '/augmentation/aug_input/'
        self.split_mask_path = self.data_path + '/split/mask/'
        self.split_input_path = self.data_path + '/split/input/'
        self.split_merge_path = self.data_path + '/split/merge/'
        self.merge_path = self.data_path + '/merge/'
        self.mask_path = self.data_path + '/mask/'
        self.input_path = self.data_path + '/input/'
        self.source_dirs = source_dirs
        self.dim_out = dim_out
        # check dim_out
        if np.count_nonzero(np.mod(dim_out, 8))>0:
            raise ValueError(f'dim_out {dim_out} has to be divisible by 8.')
        self.aug_factor = aug_factor
        self.brightness_contrast = brightness_contrast
        self.noise_amp = noise_amp
        self.random_flip_z = random_flip_z
        self.padding_mode = padding_mode
        self.clip_thrs = clip_thrs
        self.n_slices = n_slices
        self.mode = mode

        if create:
            self.make_dirs()
            self.move_and_edit()
            self.merge_images()
            self.split()
            self.augment()

    def make_dirs(self):
        """
        Create directories for training data
        """
        # delete old files
        if os.path.exists('./data/'):
            shutil.rmtree('./data/')
        # make folders
        os.makedirs(self.input_path, exist_ok=True)
        os.makedirs(self.mask_path, exist_ok=True)
        os.makedirs(self.merge_path, exist_ok=True)
        os.makedirs(self.split_merge_path, exist_ok=True)
        os.makedirs(self.split_input_path, exist_ok=True)
        os.makedirs(self.split_mask_path, exist_ok=True)
        os.makedirs(self.aug_input_path, exist_ok=True)
        os.makedirs(self.aug_mask_path, exist_ok=True)

    def move_and_edit(self):
        """
        Move and edit training images and labels
        """
        # create input data
        files_input = glob.glob(self.source_dirs[0] + '*')
        for file_i in files_input:
            stack_i = tifffile.imread(file_i)
            self.stack_shape = stack_i.shape
            # clip and normalize (0,255)
            stack_i = np.clip(stack_i, a_min=np.percentile(stack_i, self.clip_thrs[0]),
                              a_max=np.percentile(stack_i, self.clip_thrs[1]))
            stack_i = stack_i - np.min(stack_i)
            stack_i = stack_i / np.max(stack_i) * 255
            stack_i = stack_i.astype('uint8')
            shape_i = stack_i.shape
            if self.n_slices<shape_i[0]:
                raise ValueError(f'n_slices needs to be larger than number of slices. {file_i}: {shape_i[0]} slices.')
            stack_i = np.pad(stack_i, ((0, self.n_slices - shape_i[0]), (0, 0), (0, 0)), mode=self.padding_mode)
            save_i = os.path.splitext(os.path.basename(file_i))[0]
            save_i = save_i.replace(' ', '_')
            tifffile.imsave(self.input_path + save_i + '.tif', stack_i)

        # create masks
        files_mask = glob.glob(self.source_dirs[1] + '*')
        print('%s files found' % len(files_mask))
        for file_i in files_mask:
            stack_i = tifffile.imread(file_i).astype('float32')
            save_i = os.path.splitext(os.path.basename(file_i))[0]
            save_i = save_i.replace(' ', '_')
            if self.mode == 'max_projection':
                # make maximum intensity projection
                max_int_i = np.max(stack_i, axis=0)
                # clip and normalize (0,255)
                max_int_i = np.clip(max_int_i, a_min=np.percentile(max_int_i, self.clip_thrs[0]),
                                    a_max=np.percentile(max_int_i, self.clip_thrs[1]))
                max_int_i = max_int_i - np.min(max_int_i)
                max_int_i = max_int_i / np.max(max_int_i) * 255
                self.mask_shape = [1, max_int_i.shape[0], max_int_i.shape[1]]
                tifffile.imsave(self.mask_path + save_i + '.tif', max_int_i.astype('int8'))
            elif self.mode == 'binary_mask':
                # make binary stack
                stack_i[stack_i > 0] = 255
                shape_i = stack_i.shape
                stack_i = np.pad(stack_i, ((0, self.n_slices - shape_i[0]), (0, 0), (0, 0)), mode=self.padding_mode)
                tifffile.imsave(self.mask_path + save_i + '.tif', stack_i.astype('int8'))
                self.mask_shape = stack_i.shape

    def merge_images(self):
        """
        Merge images and labels
        """
        mask_files = glob.glob(self.data_path + '/mask/*.tif')
        input_files = glob.glob(self.data_path + '/input/*.tif')

        if len(mask_files) != len(input_files):
            raise ValueError('Number of ground truth does not match number of input stacks')

        for i, file_i in enumerate(mask_files):
            basename_i = os.path.basename(file_i)
            mask_i = tifffile.imread(self.data_path + '/mask/' + basename_i)
            if self.mode == 'max_projection':
                mask_i = mask_i.reshape((1, mask_i.shape[0], mask_i.shape[1]))
            input_i = tifffile.imread(self.data_path + '/input/' + basename_i)
            merge = np.zeros((mask_i.shape[0] + input_i.shape[0], mask_i.shape[1], mask_i.shape[2]))
            merge[:mask_i.shape[0], :, :] = mask_i
            merge[mask_i.shape[0]:, :, :] = input_i
            merge = merge.astype('uint8')
            tifffile.imsave(self.merge_path + str(i) + '.tif', merge)

    def split(self):
        """
        Split merged images and labels into patches of size dim_out
        """
        merges = glob.glob(self.merge_path + '*.tif')
        for i in range(len(merges)):
            merge = tifffile.imread(self.merge_path + str(i) + '.tif')
            dim_in = merge.shape
            # padding if dim_in < dim_out
            x_gap = max(0, self.dim_out[0] - dim_in[1])
            y_gap = max(0, self.dim_out[1] - dim_in[2])
            merge = np.pad(merge, ((0, 0), (0, x_gap), (0, y_gap)))
            # number of patches in x and y
            dim_in = merge.shape
            n_x = int(np.ceil(dim_in[1] / self.dim_out[0]))
            n_y = int(np.ceil(dim_in[2] / self.dim_out[1]))
            # starting indices of patches
            x_start = np.linspace(0, dim_in[1] - self.dim_out[0], n_x).astype('int16')
            y_start = np.linspace(0, dim_in[2] - self.dim_out[1], n_y).astype('int16')
            for j in range(n_x):
                for k in range(n_y):
                    patch_ij = merge[:, x_start[j]:x_start[j] + self.dim_out[0],
                                    y_start[k]:y_start[k] + self.dim_out[1]]
                    tifffile.imsave(self.split_merge_path + '%s_%s_%s.tif' % (i, j, k), patch_ij)
                    tifffile.imsave(self.split_input_path + '%s_%s_%s.tif' % (i, j, k),
                                    patch_ij[self.mask_shape[0]:, :, :])
                    tifffile.imsave(self.split_mask_path + '%s_%s_%s.tif' % (i, j, k),
                                    patch_ij[:self.mask_shape[0], :, :])

    def augment(self, p=0.8):
        """
        Augment training images
        :param p: ratio of augmented images
        """
        aug_pipeline = Compose([
            Flip(),
            RandomRotate90(p=1.0),
            GaussNoise(var_limit=(self.noise_amp, self.noise_amp), p=0.5),
            RandomBrightnessContrast(brightness_limit=self.brightness_contrast[0],
                                     contrast_limit=self.brightness_contrast[1], p=0.5)
        ],
            p=p)

        patches_image = glob.glob(self.split_input_path + '*.tif')
        patches_mask = glob.glob(self.split_mask_path + '*.tif')
        n_patches = len(patches_image)
        k = 0
        for i in range(n_patches):
            image_i = tifffile.imread(patches_image[i])
            mask_i = tifffile.imread(patches_mask[i])
            # swap axis (1->3)
            image_i = np.swapaxes(image_i, 0, 2)
            mask_i = np.swapaxes(mask_i, 0, 2)

            data_i = {'image': image_i, 'mask': mask_i}
            data_aug_i = np.asarray([aug_pipeline(**data_i) for _ in range(self.aug_factor)])
            imgs_aug_i = np.asarray([data_aug_i[j]['image'] for j in range(self.aug_factor)])
            masks_aug_i = np.asarray([data_aug_i[j]['mask'] for j in range(self.aug_factor)])

            # swap axis back
            imgs_aug_i = np.swapaxes(imgs_aug_i, 1, 3)
            masks_aug_i = np.swapaxes(masks_aug_i, 1, 3)

            # random flip z axis
            if self.random_flip_z:
                if choice([True, False]):
                    imgs_aug_i = imgs_aug_i[:, ::-1]
                    masks_aug_i = masks_aug_i[:, ::-1]

            for j in range(self.aug_factor):
                tifffile.imsave(self.aug_input_path + '%s.tif' % k, imgs_aug_i[j])
                tifffile.imsave(self.aug_mask_path + '%s.tif' % k, masks_aug_i[j])
                k += 1
        print('Number of training images: %s' % k)

    def __len__(self):
        return len(os.listdir(self.aug_input_path))

    def __getitem__(self, idx):
        img_name = str(idx) + '.tif'
        mid_name = os.path.basename(img_name)
        image_0 = tifffile.imread(self.aug_input_path + mid_name).astype('float32') / 255
        mask_0 = tifffile.imread(self.aug_mask_path + mid_name).astype('float32') / 255
        image = torch.from_numpy(image_0)
        mask = torch.from_numpy(mask_0)
        del image_0, mask_0
        sample = {'image': image, 'mask': mask}
        return sample
