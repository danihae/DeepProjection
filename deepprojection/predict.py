from .ProjNet import ProjNet
from .postprocess import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class PredictStack:
    """
    Class for prediction of single stacks
    """

    def __init__(self, stack, weights, model=ProjNet, resize_dim=(1024, 1024), clip_thrs=(0, 99.95),
                 clip_values=None, mask_thrs=None, add_tile=0, invert_slices=False):
        """
        Parameters
        ----------
        stack : ndarray
            ZXY stack numpy array
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
        mask_thrs : float = None
            If not None, additional binary thresholding of predicted masks
        invert_slices : book
            If True, z order of stacks is inverted prior to prediction
        """
        self.resize_dim = resize_dim
        self.mask_thrs = mask_thrs
        self.add_tile = add_tile
        # preprocess data
        self.stack_input = stack.copy()
        self.stack = stack.astype('float32')
        if invert_slices:
            self.stack = self.stack[::-1]
        self.n_slices, self.n_pixel = self.stack.shape[0], self.stack.shape[1:]
        if clip_values is None:
            clip_values = (np.percentile(self.stack, clip_thrs[0]), np.percentile(self.stack, clip_thrs[1]))
        self.stack = np.clip(self.stack, a_min=clip_values[0], a_max=clip_values[1])
        self.stack = self.stack - np.min(self.stack)
        self.stack = self.stack / np.max(self.stack)
        # split stacks in patches
        self.__split()
        # load model and predict data
        weights_dict = torch.load(weights)
        self.model = model(n_filter=weights_dict['n_filter']).to(device)
        self.model.load_state_dict(weights_dict['state_dict'])
        self.model.eval()
        self.__predict()
        # stitch patches back together
        self.__stitch()

    def __split(self):
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

    def __predict(self):
        with torch.no_grad():
            self.patches_masks = np.zeros_like(self.patches, dtype='float16')
            for j, patch_j in enumerate(self.patches):
                patch_j_torch = torch.from_numpy(patch_j).to(device)
                patch_j_torch = patch_j_torch.view((1, 1, self.n_slices, self.resize_dim[0], self.resize_dim[1]))
                res_j, mask_j = self.model(patch_j_torch)
                mask_j = mask_j.view((self.n_slices, self.resize_dim[0], self.resize_dim[1])).detach().cpu().numpy()
                # threshold mask
                if self.mask_thrs is not None:
                    mask_j_thrs = np.copy(mask_j)
                    mask_j_thrs[mask_j < self.mask_thrs] = 0
                    mask_j_thrs[mask_j >= self.mask_thrs] = 1
                    mask_j = mask_j_thrs
                # write in array
                self.patches_masks[j] = mask_j

    def __stitch(self):
        # create array
        self.masks = np.zeros((self.n, self.n_slices, np.max((self.resize_dim[0], self.n_pixel[0])),
                               np.max((self.resize_dim[1], self.n_pixel[1]))), dtype='float16')
        n = 0
        for j in range(self.n_y):
            for k in range(self.n_x):
                self.masks[n, :, self.y_start[j]:self.y_start[j] + self.resize_dim[0],
                self.x_start[k]:self.x_start[k] + self.resize_dim[1]] = self.patches_masks[n, :, :, :]
                n += 1
        # max of masks in overlapping patches and resize masks
        self.masks = np.max(self.masks, axis=0)[:, :self.n_pixel[0], : self.n_pixel[1]]
        self.result = np.max(self.stack_input * self.masks, axis=0).astype(self.stack_input.dtype)
