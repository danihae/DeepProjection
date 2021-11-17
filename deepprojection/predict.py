import gc
import json

from .utils import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class PredictMovie:
    """
    Class for prediction of movies
    folder: path to folder with individual tif files
    resize_dim: dimension for resizing prior to prediction
    clip_thrs: clip threshold for stack normalization
    n_filter: number of filter kernels
    weights: path to model weights
    """

    def __init__(self, folder, model, weights, resize_dim=(512, 1024), clip_thrs=99.9, n_filter=8,
                 mask_thrs=None, folder_color=None, normalization_mode='movie', export_masks=False,
                 invert_slices=False):
        """

        Parameters
        ----------
        folder : str
            Folder containing stacks (filesnames need to have time at end)
        model
            Network model
        weights : str
            Trained model weights
        resize_dim : tuple(int, int)
            Resize dimensions (2**n, 2**n) with n>4
        clip_thrs : float
            Higher percentile for intensity clipping
        n_filter : int
            Number of convolutional filters
        mask_thrs : float = None
            If not None, additional binary thresholding of predicted masks
        folder_color : str
            If not None, masks are applied to stacks of second fluorescent channel in folder_color
        normalization_mode : str
            If 'movie', the intensity is normalized based on a cummulative histogram of all stacks, if 'stack',
            the intensities are normalized individually for each stack
        export_masks : bool
            If True, the predicted masks are stored
        invert_slices :
            If True, z order of stacks is inverted prior to prediction
        """
        self.folder = folder
        self.folder_color = folder_color

        # params
        self.resize_dim = resize_dim
        self.n_filter = n_filter
        self.mask_thrs = mask_thrs
        self.export_masks = export_masks
        self.info = {'model': weights, 'mask_thrs': mask_thrs, 'clip_thrs': clip_thrs,
                     'normalization_mode': normalization_mode}
        # save metadata dictionary
        with open(self.folder + '/info.txt', 'w') as file:
            file.write(json.dumps(self.info))
        # temp folder
        self.temp_folder = './temp/'
        if os.path.exists(self.temp_folder):
            shutil.rmtree(self.temp_folder)
        os.makedirs(self.temp_folder, exist_ok=True)

        # read and preprocess data
        self.n_frames, self.n_slices, self.n_pixel = convert_to_stack(self.folder, self.temp_folder + '/stacks/',
                                                                      invert_order=invert_slices)
        print('Data loaded!')
        self.preprocess(normalization_mode, clip_thrs)
        print('Data pre-processed!')

        # split stacks in patches
        self.split()
        # load model and predict data
        self.model = model(n_filter=n_filter).to(device)
        self.model.load_state_dict(torch.load(weights, map_location=device)['state_dict'])
        self.model.eval()
        self.predict()
        print('Data predicted!')
        # stitch patches back together
        self.stitch()
        print('Images stitched back together!')
        # remove temp data
        shutil.rmtree(self.temp_folder)

    def preprocess(self, normalization_mode, clip_thrs):
        """
        1. Normalization (clipping and rescaling grey values to 8bit)
        2. padding of array for neural network (2**n + 2**m (+ 2**p + ...) pixels with (n, m, ... ) > 5
        """
        if normalization_mode == 'movie':
            # get percentiles of all stacks
            percentiles = np.zeros((self.n_frames, 2))
            for t in range(self.n_frames):
                stack_t = tifffile.imread(self.temp_folder + f'/stacks/stack_{t}.tif')
                percentiles[t] = (np.percentile(stack_t, 0.),
                                  np.percentile(stack_t, clip_thrs))
            percentiles = np.mean(percentiles, axis=0)

        for t in range(self.n_frames):
            stack_t = tifffile.imread(self.temp_folder + f'/stacks/stack_{t}.tif')
            if normalization_mode == 'movie':
                stack_t = np.clip(stack_t, a_min=percentiles[0],
                                  a_max=percentiles[1])
            elif normalization_mode == 'stack':
                stack_t = np.clip(stack_t, a_min=np.percentile(stack_t, 0.),
                                  a_max=np.percentile(stack_t, clip_thrs))
            else:
                raise AttributeError('Specify normalization mode! ("movie" or "stack")')

            stack_t = stack_t - np.min(stack_t)
            stack_t = stack_t / np.max(stack_t) * 255
            stack_t = stack_t.astype('uint8')
            tifffile.imsave(self.temp_folder + f'/stacks/stack_{t}.tif', stack_t)

    def split(self):
        # number of patches in x and y
        self.n_x = int(np.ceil(self.n_pixel[1] / self.resize_dim[1]))
        self.n_y = int(np.ceil(self.n_pixel[0] / self.resize_dim[0]))
        self.n = self.n_x * self.n_y  # number of patches
        # starting indices of patches
        self.x_start = np.linspace(0, self.n_pixel[1] - self.resize_dim[1], self.n_x).astype('uint16')
        self.y_start = np.linspace(0, self.n_pixel[0] - self.resize_dim[0], self.n_y).astype('uint16')
        print(f'Resizing into each {self.n} patches ...')
        folder_patches = self.temp_folder + '/patches/'
        os.makedirs(folder_patches, exist_ok=True)
        for i in range(self.n_frames):
            stack_i = tifffile.imread(self.temp_folder + f'/stacks/stack_{i}.tif')
            patches_i = np.zeros((self.n, self.n_slices, self.resize_dim[0], self.resize_dim[1]), dtype='uint8')
            # padding if image dim smaller than resize_dim
            stack_i = np.pad(stack_i, ((0, 0), (0, np.max((0, self.resize_dim[0] - self.n_pixel[0]))),
                                       (0, np.max((0, self.resize_dim[1] - self.n_pixel[1])))), constant_values=0)
            # split in resize_dim
            n = 0
            for j in range(self.n_y):
                for k in range(self.n_x):
                    patches_i[n, :, :, :] = stack_i[:, self.y_start[j]:self.y_start[j] + self.resize_dim[0],
                                            self.x_start[k]:self.x_start[k] + self.resize_dim[1]]
                    n += 1
            # save patches in temp folder and delete stacks
            tifffile.imsave(folder_patches + f'patches_{i}.tif', patches_i)
            # os.remove(self.temp_folder + f'/stacks/stack_{i}.tif')

    def predict(self):
        with torch.no_grad():
            result_folder = self.temp_folder + '/results/'
            os.makedirs(result_folder, exist_ok=True)
            for i in tqdm(range(self.n_frames)):
                patches_i = tifffile.imread(self.temp_folder + f'/patches/patches_{i}.tif').astype('float32') / 255
                res_i = np.zeros((self.n, self.resize_dim[0], self.resize_dim[1]))
                mask_i = np.zeros_like(patches_i)
                for j, patch_j in enumerate(patches_i):
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
                    res_i[j] = res_j * 255
                    mask_i[j] = mask_j * 255
                # save results as tif
                tifffile.imsave(result_folder + f'res_patches_{i}.tif', res_i.astype('uint8'))
                if self.export_masks:
                    tifffile.imsave(result_folder + f'masks_patches_{i}.tif', mask_i.astype('uint8'))
                os.remove(self.temp_folder + f'/patches/patches_{i}.tif')
                gc.collect()
                torch.cuda.empty_cache()

    def stitch(self):
        print('Stitching patches back together ...')
        # create array
        self.results = np.zeros((self.n_frames, np.max((self.resize_dim[0], self.n_pixel[0])),
                                 np.max((self.resize_dim[1], self.n_pixel[1]))), dtype='uint8')
        if self.export_masks:
            self.masks = np.zeros((self.n_frames, self.n_slices, np.max((self.resize_dim[0], self.n_pixel[0])),
                                   np.max((self.resize_dim[1], self.n_pixel[1]))), dtype='uint8')
        for i in tqdm(range(self.n_frames)):
            patches_i = tifffile.imread(self.temp_folder + f'/results/res_patches_{i}.tif')
            result_i = np.zeros((self.n, np.max((self.resize_dim[0], self.n_pixel[0])),
                                 np.max((self.resize_dim[1], self.n_pixel[1]))), dtype='uint8')
            if self.export_masks:
                masks_patches_i = tifffile.imread(self.temp_folder + f'/results/masks_patches_{i}.tif')
                masks_i = np.zeros((self.n, self.n_slices, np.max((self.resize_dim[0], self.n_pixel[0])),
                                    np.max((self.resize_dim[1], self.n_pixel[1]))), dtype='uint8')
            n = 0
            for j in range(self.n_y):
                for k in range(self.n_x):
                    result_i[n, self.y_start[j]:self.y_start[j] + self.resize_dim[0],
                    self.x_start[k]:self.x_start[k] + self.resize_dim[1]] = patches_i[n, :, :]
                    if self.export_masks:
                        masks_i[n, :, self.y_start[j]:self.y_start[j] + self.resize_dim[0],
                        self.x_start[k]:self.x_start[k] + self.resize_dim[1]] = masks_patches_i[n, :, :, :]
                    n += 1
            # maximum of overlapping regions
            self.results[i] = np.max(result_i, axis=0)
            if self.export_masks:
                self.masks[i] = np.max(masks_i, axis=0)
        # change to input size (if zero padding) and save results
        self.results = self.results[:, :self.n_pixel[0], :self.n_pixel[1]]
        tifffile.imsave(self.folder[:-1] + '.tif', self.results, metadata=self.info)
        del self.results
        os.remove(self.temp_folder + f'/results/res_patches_{i}.tif')
        if self.export_masks:
            self.masks = self.masks[:, :, :self.n_pixel[0], :self.n_pixel[1]]
            del self.masks
            os.remove(self.temp_folder + f'/results/masks_patches_{i}.tif')
            # here save masks?


class PredictStack:
    """
    Class for prediction of single stacks
    """

    def __init__(self, filename, filename_output, model, weights, resize_dim=(512, 1024), clip_thrs=99.98,
                 n_filter=8, mask_thrs=None, export_masks=True, add_tile=0, invert_slices=False):
        """

        Parameters
        ----------
        filename : str
            Filename of stack
        filename_output : str
            Filename of output
        model
            Network model
        weights : str
            Trained model weights
        resize_dim : tuple(int, int)
            Resize dimensions (2**n, 2**n) with n>4
        clip_thrs : float
            Higher percentile for intensity clipping
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
        print('Data loaded!')
        self.stack = np.clip(self.stack, a_min=np.percentile(self.stack, 0),
                             a_max=np.percentile(self.stack, clip_thrs))
        self.stack = self.stack - np.min(self.stack)
        self.stack = self.stack / np.max(self.stack)
        print('Data pre-processed!')
        # split stacks in patches
        self.split()
        # load model and predict data
        self.model = model(n_filter=n_filter).to(device)
        self.model.load_state_dict(torch.load(weights)['state_dict'])
        self.model.eval()
        self.predict()
        print('Data predicted!')
        # stitch patches back together
        self.stitch()
        print('Images stitched back together!')
        # remove temp data
        # os.remove(self.temp_folder)

    def split(self):
        # number of patches in x and y
        self.n_x = int(np.ceil(self.n_pixel[1] / self.resize_dim[1])) + self.add_tile
        self.n_y = int(np.ceil(self.n_pixel[0] / self.resize_dim[0])) + self.add_tile
        self.n = self.n_x * self.n_y  # number of patches
        # starting indices of patches
        self.x_start = np.linspace(0, self.n_pixel[1] - self.resize_dim[1], self.n_x).astype('uint16')
        self.y_start = np.linspace(0, self.n_pixel[0] - self.resize_dim[0], self.n_y).astype('uint16')
        print(f'Resizing into each {self.n} patches ...')
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
        print('Stitching patches back together ...')
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
        tifffile.imsave(self.filename_output + '_result.tif', self.result.astype('uint8'))
        if self.export_masks:
            self.masks = self.masks[:, :self.n_pixel[0], :self.n_pixel[1]]
            tifffile.imsave(self.filename_output + '_masks.tif', self.masks.astype('uint8'))
