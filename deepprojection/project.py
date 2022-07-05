import shutil

import tifffile

from .ProjNet import ProjNet
from .postprocess import *
from .predict import PredictStack
from .utils import *


class Project:
    """
    Main class for prediction of post-processing of stacks
    """

    def __init__(self, input_path, data_format, weights, mode='mip', mask_thrs=None, filter_time=0, filter_size=0, offset=0,
                 channel=None, model=ProjNet, filename_output=None, filename_masks=None, resize_dim=(1024, 1024),
                 clip_thrs=(0, 99.95), normalization_mode='movie', invert_slices=False, temp_folder='../temp/',
                 bigtiff=False):
        """
        Parameters
        ----------
        input_path : str
            For individual files: folder containing images/stacks (names must have timepoint "_time**_"/"_t**_").
            For single hyperstack: filename of tif-file
        data_format : str
            Data format ('Z-XY': single slices of stack, 'ZXY': single stack, 'T-Z-XY': single slices of movie,
            'T-ZXY': single stacks of movie, 'TZXY': hyperstack of movie, 'ZTCXY': hyperstack of multi-color movie,
            'ZCXY': hyperstack of multi-color stack)
        weights : str
            Trained model weights (*.pth)
        model
            Convolutional neural network model (default: ProjNet)
        mode : str
            Projection mode for masked stack ('mip', 'mean', 'median', 'max', 'min'). For all modes except 'mip' a
            unique z-map is created (self.z_map).
        mask_thrs : float = None
            If not None, additional binary thresholding of predicted masks
        filter_time : int
            Filtering window of masks over time (maximum filter, needs to be uneven integer)
        filter_size : int
            Filter size for smoothing z-map (not for mode='mip')
        offset : list
            z-offset of masks. Can be single value, or list of values (not for 'mip'), e.g. [0, 1, 2].
            For all modes except 'mip' the z-map shifted by offset value, or for a list of values, multiple z-positions
            are projected.
        filename_output : str
            If not None, output is saved to filename_output. If None, it is saved in the parent directory of the
            input_path
        filename_masks : str
            If not None, predicted masks are saved to filename_masks. If None, they are saved in the parent directory
            of the input_path
        resize_dim : tuple(int, int)
            Resize dimensions (has to be divisible by 8)
        clip_thrs : tuple(float, float)
            Lower and higher percentile for intensity clipping prior to prediction.
        normalization_mode : str
            If 'movie', the intensity is normalized based on a cumulative histogram of all stacks, if 'stack',
            the intensities are normalized individually for each stack, if 'first', only the histogram of the first
            frame is used.
        invert_slices :
            If True, z order of stacks is inverted prior to prediction
        bigtiff : bool
            If True, bigtiff format is used (file size >4GB)
        """
        print(f'Predicting {input_path} ...')

        self.input = input_path
        self.filename_output = filename_output
        self.filename_masks = filename_masks
        self.temp_folder = temp_folder
        self.temp_file = temp_folder + 'data.tif'

        # params
        # check resize_dim
        if np.count_nonzero(np.mod(resize_dim, 8)) > 0:
            raise ValueError(f'resize_dim {resize_dim} has to be divisible by 8.')
        info = {'mode': mode, 'mask_thrs': mask_thrs, 'clip_thrs': clip_thrs,
                'normalization_mode': normalization_mode, 'weights': weights, 'time_average': filter_time,
                'filter_size': filter_size, 'offset': offset}
        # temp input_path
        if os.path.exists(self.temp_folder):
            shutil.rmtree(self.temp_folder)
        os.makedirs(self.temp_folder, exist_ok=True)

        # read and preprocess data
        print('Copying stacks to temp input_path...')
        n_frames, n_slices, n_pixel = convert_to_hyperstack(self.input, self.temp_file, data_format=data_format,
                                                            invert_order=invert_slices, bigtiff=True)

        # normalization modes
        print('Normalizing movie...')
        if normalization_mode == 'stack':
            clip_values = None
        elif normalization_mode == 'first':
            first_frame = tifffile.imread(self.temp_file, key=0)
            clip_values = (np.percentile(first_frame, clip_thrs[0]), np.percentile(first_frame, clip_thrs[1]))
        elif normalization_mode == 'movie':
            # get percentiles of all stacks
            percentiles = np.zeros((n_frames, 2))
            for t in tqdm(range(n_frames)):
                stack_t = tifffile.imread(self.temp_file, key=t)
                percentiles[t] = (np.percentile(stack_t, clip_thrs[0]),
                                  np.percentile(stack_t, clip_thrs[1]))
            clip_values = np.mean(percentiles, axis=0)
        else:
            raise ValueError('Specify correct normalization mode (stack, first or movie)')

        # set paths to store results and masks
        if self.filename_output is None:
            if '.tif' in self.input or '.TIF' in self.input:
                self.filename_output = self.input[:-4] + '_result.tif'
            else:
                self.filename_output = self.input[:-1] + '.tif'
        if self.filename_masks is None:
            self.filename_masks = self.temp_folder + 'masks.tif'

        # predict stacks and write in tiff file
        print('Predicting stacks...')
        with tifffile.TiffWriter(self.filename_output, bigtiff=bigtiff) as tif:
            with tifffile.TiffWriter(self.filename_masks, bigtiff=True) as tif_masks:
                for t in tqdm(range(n_frames)):
                    stack_t = tifffile.imread(self.temp_file, key=range(t*n_slices, (t+1)*n_slices))
                    predict_t = PredictStack(stack_t, weights=weights,
                                             model=model, resize_dim=resize_dim, clip_thrs=clip_thrs,
                                             clip_values=clip_values, mask_thrs=mask_thrs, invert_slices=invert_slices)
                    tif.write(predict_t.result, metadata=info, contiguous=True)
                    masks_t = predict_t.masks * 255
                    tif_masks.write(masks_t.astype('uint8'), metadata=info, contiguous=True)

        # postprocessing
        if mode != 'mip' or filter_time != 0 or offset != 0:
            # load masks and input stacks
            masks = tifffile.imread(self.filename_masks)
            stacks = tifffile.imread(self.temp_file)
            # postprocessing (see postprocess.py)
            process = PostProcess(stacks, masks, channel=channel)
            process.process(mode=mode, filter_time=filter_time, mask_thrs=mask_thrs, filter_size=filter_size,
                            offset=offset)
            # remove empty dimensions
            process.result = np.squeeze(process.result)
            process.masks_edit = np.squeeze(process.masks_edit)
            # remove empty dimensions and save tif file
            tifffile.imwrite(self.filename_output, process.result, metadata=info)
            if filename_masks is not None:
                tifffile.imwrite(filename_masks, process.masks_edit)

        print(f'Result saved to {self.filename_output}, \n masks saved to {self.filename_masks}.')
        # delete temp input_path
        shutil.rmtree(self.temp_folder)
