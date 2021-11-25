import gc

import numpy as np

from .utils import *
from .ProjNet import ProjNet

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class PredictMovie:
    """
    Class for prediction of movies
    """

    def __init__(self, folder, weights, model=ProjNet, filename_output=None, resize_dim=(512, 1024),
                 clip_thrs=(0, 99.9), n_filter=8, mask_thrs=None, folder_color=None, normalization_mode='movie',
                 export_masks=False, invert_slices=False, temp_folder='../temp/', bigtiff=False):
        """

        Parameters
        ----------
        folder : str
            Folder containing stacks (filesnames need to have time at end)
        weights : str
            Trained model weights
        model
            Network model
        filename_output : str
            If not None, output is save to filename_output. If None, it is saved in the parent directory of the input
        resize_dim : tuple(int, int)
            Resize dimensions (has to be divisible by 8)
        clip_thrs : tuple(float, float)
            Lower and higher percentile for intensity clipping
        n_filter : int
            Number of convolutional filters
        mask_thrs : float = None
            If not None, additional binary thresholding of predicted masks
        folder_color
            If not None, masks are applied to stacks of second fluorescent channel in folder_color
        normalization_mode : str
            If 'movie', the intensity is normalized based on a cummulative histogram of all stacks, if 'stack',
            the intensities are normalized individually for each stack, if 'first', only the histogram of the first
            frame is used.
        export_masks : bool
            If True, the predicted masks are stored
        invert_slices :
            If True, z order of stacks is inverted prior to prediction
        bigtiff : bool
            If True, bigtiff format is used (file size >4GB)
        """
        print(f'Predicting {folder} ...')

        self.folder = folder
        self.folder_color = folder_color
        self.filename_output = filename_output
        self.temp_folder = temp_folder

        # params
        self.resize_dim = resize_dim
        # check resize_dim
        if np.count_nonzero(np.mod(resize_dim, 8)) > 0:
            raise ValueError(f'resize_dim {resize_dim} has to be divisible by 8.')
        self.n_filter = n_filter
        self.mask_thrs = mask_thrs
        self.export_masks = export_masks
        self.info = {'model': weights, 'mask_thrs': mask_thrs, 'clip_thrs': clip_thrs,
                     'normalization_mode': normalization_mode}
        # temp folder
        if os.path.exists(self.temp_folder):
            shutil.rmtree(self.temp_folder)
        os.makedirs(self.temp_folder, exist_ok=True)

        # read and preprocess data
        print('Copying stacks to temp folder...')
        self.n_frames, self.n_slices, self.n_pixel = convert_to_stack(self.folder, self.temp_folder,
                                                                      invert_order=invert_slices)

        # normalization modes
        print('Normalizing movie...')
        if normalization_mode == 'stack':
            clip_values = None
        elif normalization_mode == 'first':
            first_frame = tifffile.imread(self.temp_folder + 'stack_0.tif')
            clip_values = (np.percentile(first_frame, clip_thrs[0]), np.percentile(first_frame, clip_thrs[1]))
        elif normalization_mode == 'movie':
            # get percentiles of all stacks
            percentiles = np.zeros((self.n_frames, 2))
            for t in tqdm(range(self.n_frames)):
                stack_t = tifffile.imread(self.temp_folder + f'/stack_{t}.tif')
                percentiles[t] = (np.percentile(stack_t, clip_thrs[0]),
                                  np.percentile(stack_t, clip_thrs[1]))
            clip_values = np.mean(percentiles, axis=0)
        else:
            raise ValueError('Specify correct normalization mode (stack, first or movie)')

        if filename_output is None:
            filename_output = self.folder[:-1] + '.tif'

        # predict stacks and write in tiff file
        print('Predicting stacks...')
        with tifffile.TiffWriter(filename_output, bigtiff=bigtiff) as tif:
            for t in tqdm(range(self.n_frames)):
                stack_t = PredictStack(self.temp_folder + f'/stack_{t}.tif', filename_output=None, weights=weights,
                                       model=model, resize_dim=resize_dim, clip_thrs=clip_thrs,
                                       clip_values=clip_values, n_filter=n_filter, mask_thrs=mask_thrs,
                                       export_masks=False, invert_slices=invert_slices)
                tif.write(stack_t.result, metadata=self.info, contiguous=True)

        print(f'Result saved to {filename_output}.')
        # delete temp folder
        shutil.rmtree(self.temp_folder)


class PredictStack:
    """
    Class for prediction of single stacks
    """

    def __init__(self, filename, filename_output, weights, model=ProjNet, resize_dim=(512, 1024), clip_thrs=(0, 99.98),
                 clip_values=None, n_filter=8, mask_thrs=None, export_masks=True, add_tile=0, invert_slices=False):
        """

        Parameters
        ----------
        filename : str
            Filename of stack
        filename_output :
            Filename of output. If None, result is not saved as tif file.
        weights : str
            Trained model weights
        model
            Network model
        resize_dim : tuple(int, int)
            Resize dimensions (2**n, 2**n) with n>4
        clip_thrs : tuple(float, float)
            Lower and higher percentile for intensity clipping
        clip_values : tuple
            If not None, clip_values are applied for normalization and clip_thrs is ignored.
        n_filter : int
            Number of convolutional filters
        mask_thrs : float = None
            If not None, additional binary thresholding of predicted masks
        export_masks : bool
            If True, the predicted masks are stored
        invert_slices : book
            If True, z order of stacks is inverted prior to prediction
        """
        self.filename = filename
        self.filename_output = filename_output
        self.resize_dim = resize_dim
        self.n_filter = n_filter
        self.mask_thrs = mask_thrs
        self.export_masks = export_masks
        self.add_tile = add_tile
        # read and preprocess data
        self.stack = tifffile.imread(self.filename).astype('float32')
        if invert_slices:
            self.stack = self.stack[::-1]
        if len(self.stack.shape) == 4:  # if two color stack
            self.stack = self.stack[:, 0]
        self.n_slices, self.n_pixel = self.stack.shape[0], self.stack.shape[1:]
        if clip_values is None:
            clip_values = (np.percentile(self.stack, clip_thrs[0]), np.percentile(self.stack, clip_thrs[1]))
        self.stack = np.clip(self.stack, a_min=clip_values[0], a_max=clip_values[1])
        self.stack = self.stack - np.min(self.stack)
        self.stack = self.stack / np.max(self.stack)
        # split stacks in patches
        self.split()
        # load model and predict data
        self.model = model(n_filter=n_filter).to(device)
        self.model.load_state_dict(torch.load(weights)['state_dict'])
        self.model.eval()
        self.predict()
        # stitch patches back together
        self.stitch()
        # save result
        if self.filename_output is not None:
            tifffile.imsave(self.filename_output + '_result.tif', self.result.astype('uint8'))
            if self.export_masks:
                self.masks = self.masks[:, :self.n_pixel[0], :self.n_pixel[1]]
                tifffile.imsave(self.filename_output + '_masks.tif', self.masks.astype('uint8'))

    def split(self):
        # number of patches in x and y
        self.n_x = int(np.ceil(self.n_pixel[1] / self.resize_dim[1])) + self.add_tile
        self.n_y = int(np.ceil(self.n_pixel[0] / self.resize_dim[0])) + self.add_tile
        self.n = self.n_x * self.n_y  # number of patches
        # starting indices of patches
        self.x_start = np.linspace(0, self.n_pixel[1] - self.resize_dim[1], self.n_x).astype('uint16')
        self.y_start = np.linspace(0, self.n_pixel[0] - self.resize_dim[0], self.n_y).astype('uint16')
        self.patches = np.zeros((self.n, self.n_slices, self.resize_dim[0], self.resize_dim[1]), dtype='float32')
        # padding if image dim smaller than resize_dim
        self.stack = np.pad(self.stack, ((0, 0), (0, np.max((0, self.resize_dim[0] - self.n_pixel[0]))),
                                         (0, np.max((0, self.resize_dim[1] - self.n_pixel[1])))), constant_values=0)
        # split in resize_dim
        n = 0
        for j in range(self.n_y):
            for k in range(self.n_x):
                self.patches[n, :, :, :] = self.stack[:, self.y_start[j]:self.y_start[j] + self.resize_dim[0],
                                           self.x_start[k]:self.x_start[k] + self.resize_dim[1]]
                n += 1

    def predict(self):
        with torch.no_grad():
            self.patches_result = np.zeros((self.n, self.resize_dim[0], self.resize_dim[1]), dtype='float32')
            self.patches_masks = np.zeros_like(self.patches, dtype='float32')
            for j, patch_j in enumerate(self.patches):
                patch_j_torch = torch.from_numpy(patch_j).to(device)
                patch_j_torch = patch_j_torch.view((1, 1, self.n_slices, self.resize_dim[0], self.resize_dim[1]))
                res_j, mask_j, edge_j = self.model(patch_j_torch)
                mask_j = mask_j.view((self.n_slices, self.resize_dim[0], self.resize_dim[1])).detach().cpu().numpy()
                res_j = res_j.view((self.resize_dim[0], self.resize_dim[1])).detach().cpu().numpy()
                # threshold mask
                if self.mask_thrs is not None:
                    mask_j_thrs = np.copy(mask_j)
                    mask_j_thrs[mask_j < self.mask_thrs] = 0
                    mask_j_thrs[mask_j >= self.mask_thrs] = 1
                    res_j = np.max(patch_j * mask_j_thrs, axis=0)
                    mask_j = mask_j_thrs
                # write in array
                self.patches_result[j] = res_j * 255
                self.patches_masks[j] = mask_j * 255
            gc.collect()
            torch.cuda.empty_cache()

    def stitch(self):
        # create array
        result_temp = np.zeros((self.n, np.max((self.resize_dim[0], self.n_pixel[0])),
                                np.max((self.resize_dim[1], self.n_pixel[1]))), dtype='uint8')
        if self.export_masks:
            masks_temp = np.zeros((self.n, self.n_slices, np.max((self.resize_dim[0], self.n_pixel[0])),
                                   np.max((self.resize_dim[1], self.n_pixel[1]))), dtype='uint8')
        n = 0
        for j in range(self.n_y):
            for k in range(self.n_x):
                result_temp[n, self.y_start[j]:self.y_start[j] + self.resize_dim[0],
                self.x_start[k]:self.x_start[k] + self.resize_dim[1]] = self.patches_result[n, :, :]
                if self.export_masks:
                    masks_temp[n, :, self.y_start[j]:self.y_start[j] + self.resize_dim[0],
                    self.x_start[k]:self.x_start[k] + self.resize_dim[1]] = self.patches_masks[n, :, :, :]
                n += 1
        # maximum of overlapping regions
        self.result = np.max(result_temp, axis=0)
        if self.export_masks:
            self.masks = np.max(masks_temp, axis=0)
        # change to input size (if zero padding) and save results
        self.result = self.result[:self.n_pixel[0], :self.n_pixel[1]]
