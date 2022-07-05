import numpy as np
import tifffile
from tqdm import tqdm
import torch
from scipy.ndimage import generic_filter, geometric_transform
from torch import nn

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class PostProcess:
    """Class for post-processing of predicted masks"""

    def __init__(self, stacks, masks, channel=None):
        """
        Parameters
        ----------
        stacks : ndarray
            Numpy array of 3D (ZXY) or 4D (TZXY) hyperstack with input microscopy data
        masks : ndarray
            Numpy array of 3D (ZXY) or 4D (TZXY) hyperstack of binary masks. Binary masks can be saved with parameter
            filename_mask in Project class.
        channel : int
            Color channel (hyperstack needs to have ZCXY or TZCXY format)
        """
        self.stacks, self.masks = stacks, masks

        # expand dimensions of only single stack
        if len(self.masks.shape) == 3:
            self.masks = np.expand_dims(self.masks, 0)
        if len(self.stacks.shape) == 3:
            self.stacks = np.expand_dims(self.stacks, 0)

        # select color channel
        if channel is not None:
            self.stacks = self.stacks[:, :, channel]

        # check whether dimensions of stacks and masks match
        if not self.stacks.shape == self.masks.shape:
            raise ValueError('Stacks and masks need to have same shape')

        self.masks_edit, self.result, self.z_map, self.flattened = None, None, None, None

    def process(self, mode='mean', filter_time=0, mask_thrs=0.25, filter_size=0, offset=0):
        """
        Post-processing of predicted masks

        Parameters
        ----------
        mode : str
            Projection mode for masked stack ('mip', 'mean', 'median', 'max', 'min'). For all modes except 'mip' a
            unique z-map is created (self.z_map).
        filter_time : int
            Kernel size for temporal max. filter of masks (needs to be uneven integer)
        mask_thrs : float = None
            If not None, additional binary thresholding of predicted masks.
        filter_size : int
            Filter size for smoothing z-map (not for mode='mip')
        offset : list
            z-offset of masks. Can be single value, or list of values (not for 'mip'), e.g. [0, 1, 2].
            For all modes except 'mip' the z-map shifted by offset value, or for a list of values, multiple z-positions
            are projected.
        """
        if mode not in ['mip', 'mean', 'median', 'max', 'min']:
            raise ValueError(f'projection_mode {mode} not valid.')

        # time average masks
        if filter_time > 1:
            masks_temp = filter_masks_time(self.masks, filter_time, mode='mean')
        else:
            masks_temp = self.masks.copy()

        # thresholding of masks
        if mask_thrs is None:
            raise ValueError('mask_thres has to be float [0-1]')
        masks_temp[masks_temp < 255 * mask_thrs] = 0
        masks_temp[masks_temp >= 255 * mask_thrs] = 1
        masks_temp = masks_temp.astype('bool')

        self.masks_edit = np.zeros_like(self.masks, dtype='bool')

        if mode == 'mip':
            if isinstance(offset, int):
                # z offset masks
                self.masks_edit = np.roll(masks_temp, shift=-offset, axis=1)
                # set rolled edge slices 0
                if offset > 0:
                    self.masks_edit[:, :-offset] = 0
                elif offset < 0:
                    self.masks_edit[:, -offset:] = 0
            elif isinstance(offset, list):
                raise ValueError('offset has to be integer if mode is mip')
            # mask input stack
            masked_stack = self.stacks * self.masks_edit
            # z_map argmax of masks
            self.z_map = np.nanargmax(masked_stack, axis=1).astype('float32')
            # set to nan where all 0
            self.z_map[np.nansum(self.masks_edit, axis=1) == 0] = np.nan
            # interpolate z-map for smoothing
            if filter_size > 1:
                self.z_map = filter_avg_2d(self.z_map, filter_size)

        elif mode in ['max', 'min', 'mean', 'median']:
            # get array with z indices
            masks_idxs = np.zeros_like(self.masks, dtype='float32')
            temp = np.swapaxes(masks_idxs, 1, 3)
            temp[:, :, :] = np.arange(0, self.masks.shape[1])
            masks_idxs = np.swapaxes(temp, 3, 1)
            # set masks_idxs to nan if masks is zero
            masks_idxs[masks_temp == 0] = np.nan

            if mode == 'max':
                self.z_map = np.nanmax(masks_idxs, axis=1)
            elif mode == 'min':
                self.z_map = np.nanmin(masks_idxs, axis=1)
            elif mode == 'mean':
                self.z_map = np.nanmean(masks_idxs, axis=1)
            elif mode == 'median':
                self.z_map = np.nanmedian(masks_idxs, axis=1)

            # interpolate z-map for smoothing
            if filter_size > 1:
                self.z_map = filter_avg_2d(self.z_map, filter_size)

            # if single offset value convert to list
            if isinstance(offset, int):
                offset = [offset]

            # iterate list of offsets
            if isinstance(offset, list):
                for o in offset:
                    z_map_temp = self.z_map.copy()
                    z_map_temp += o
                    z_map_temp[(z_map_temp < 0) | (z_map_temp > self.stacks.shape[1] - 1)] = np.nan
                    # round and convert to integer
                    z_map_int = np.round(z_map_temp, 0).astype('uint8')
                    # convert to binary mask
                    idxs = np.where(~np.isnan(self.z_map))
                    self.masks_edit[idxs[0], z_map_int[idxs], idxs[1], idxs[2]] = True
            self.masks_edit *= masks_temp

            # mask input stack
            masked_stack = self.stacks * self.masks_edit

        else:
            raise ValueError(f'mode {mode} not valid.')

        # projection of masked stack
        self.result = np.max(masked_stack, axis=1)

    def flatten(self, axis, ref, xyz_ratio, padding=(0, 0)):
        """
        Flatten curved tissue

        Parameters
        ----------
        axis : int
            Axis along which the tissue is to be flattened (for arbitrary axis, use rotate_hyperstack function to rotate
            stack and masks prior to flattening
        ref : int
            Reference line for flattening
        xyz_ratio : float
            Ratio between xy- to z pixel size
        padding : int
            Padding of image prior to flattening, if flattened image is larger than original image
        """
        # compute gradient from z-map
        gradient = np.gradient(self.z_map, axis=(1, 2))

        # split regions above / below or left / right of reference line and unroll distance from reference along
        # z-map contour
        if axis == 0:
            r1 = gradient[axis][:, ref:]
            r2 = gradient[axis][:, :ref]
            delta_r1 = np.nancumsum(np.sqrt(1 + (r1 * xyz_ratio) ** 2), axis=axis + 1)
            delta_r2 = np.nancumsum(np.sqrt(1 + (r2[:, ::-1] * xyz_ratio) ** 2), axis=axis + 1)[:, ::-1]
            delta = np.concatenate((-delta_r2, delta_r1), axis=axis + 1) + ref
        elif axis == 1:
            r1 = gradient[axis][:, :, ref:]
            r2 = gradient[axis][:, :, :ref]
            delta_r1 = np.nancumsum(np.sqrt(1 + (r1 * xyz_ratio) ** 2), axis=axis + 1)
            delta_r2 = np.nancumsum(np.sqrt(1 + (r2[:, :, ::-1] * xyz_ratio) ** 2), axis=axis + 1)[:, :, ::-1]
            delta = np.concatenate((-delta_r2, delta_r1), axis=axis + 1) + ref
        else:
            raise ValueError(f'axis={axis} not valid, needs to be 0 or 1.')

        # create inverted map
        map_ = np.zeros_like(delta) * np.nan

        # padding
        pad_width = ((0, 0), (padding[0], padding[0]), (padding[1], padding[1]))
        delta_ = np.pad(delta, pad_width, mode='edge')
        map_ = np.pad(map_, pad_width, mode='constant', constant_values=np.nan)
        res_ = np.pad(self.result, pad_width, mode='constant', constant_values=np.nan)

        # invert map by iterating all pixels
        print('Creating transformation map...')
        for t, delta_t in enumerate(tqdm(delta_)):
            for i, column_i in enumerate(delta_t.T):
                for j, k in enumerate(column_i):
                    if axis == 0:
                        map_[t, int(k) + padding[axis], i] = j
                    elif axis == 1:
                        map_[t, j, int(k) + padding[axis]] = i

        # interpolate map with mean filter
        def nanmean_2d(x):
            return np.nanmean(x)

        map_ = generic_filter(map_, nanmean_2d, (1, 3, 3))

        # transform image with map
        print('Transforming images...')
        self.flattened = np.zeros_like(res_)
        for t, res_t in enumerate(tqdm(res_)):
            def map_array(x):
                if axis == 0:
                    return map_[t, x[0], x[1]], x[1 - axis]
                elif axis == 1:
                    return x[1 - axis], map_[t, x[0], x[1]]

            self.flattened[t] = geometric_transform(res_t, map_array, order=2)

    def save_result(self, filename, bigtiff=False):
        tifffile.imwrite(filename, self.result, bigtiff=bigtiff)

    def save_flattened(self, filename, bigtiff=False):
        tifffile.imwrite(filename, self.flattened, bigtiff=bigtiff)


def filter_masks_time(masks, size=3, mode='max', device=torch.device('cpu')):
    """Smoothing of masks over time (max pooling, min pooling, average pooling)"""
    # convert to torch tensor
    masks = torch.from_numpy(masks).type(torch.float).to(device)
    # swap axis
    masks = torch.swapdims(masks, 0, 2)
    # filtering
    if mode == 'max':
        m = nn.MaxPool2d((size, 1), stride=(1, 1)).to(device)
        out = m(masks)
    elif mode == 'min':
        m = nn.MaxPool2d((size, 1), stride=(1, 1)).to(device)
        out = - m(-masks)
    elif mode == 'mean':
        m = nn.AvgPool2d((size, 1), stride=(1, 1)).to(device)
        out = m(masks)
    else:
        raise ValueError(f'mode {mode} not valid.')
    # pad (reflection)
    padding = nn.ReflectionPad2d((0, 0, (size - 1) // 2, (size - 1) // 2)).to(device)
    out = padding(out)
    # swap axis
    out = torch.swapdims(out, 2, 0)
    return out.cpu().numpy()


def filter_avg_2d(z_map, size=3):
    """Smooth z-map with average filter"""
    # convert to torch tensor
    z_map = torch.from_numpy(z_map).type(torch.float).to(device)
    # add dim
    z_map = torch.unsqueeze(z_map, 0)
    # padding with nan before filtering
    padding = nn.ConstantPad2d(padding=(size - 1) // 2, value=float('nan')).to(device)
    out = padding(z_map)
    # create filter
    m = nn.AvgPool2d(size, stride=1).to(device)
    # filter z_map
    out = m(out)
    # set to nan where initial z-map nan
    out[torch.isnan(z_map)] = float('nan')
    # remove dim and return
    return torch.squeeze(out, 0).cpu().numpy()
