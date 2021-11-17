### DeepProjection: Trainable, structure-specific, 2D projections of 3D epithelial tissue stacks

Reference: 

**Abstract:** The efficient extraction of local high-resolution content from massive amounts of imaging data remains a serious and unsolved problem in studies of complex biological tissues. Here we present DeepProjection, a trainable projection algorithm based on deep learning. This algorithm rapidly and robustly extracts image content contained in curved manifolds from time-lapse recorded 3D image stacks by binary masking of background content, stack by stack. The masks calculated for a given movie, when predicted, e.g., on fluorescent cell boundaries on one channel, can subsequently be applied to project other fluorescent channels from the same manifold. We apply DeepProjection to follow the dynamic movements of 2D-tissue sheets in embryonic development. We show that we can selectively project the amnioserosa cell sheet during dorsal closure in Drosophila melanogaster embryos and the periderm layer in the elongating zebrafish embryo while masking highly fluorescent out-of-plane artifacts.


![Fig1](https://user-images.githubusercontent.com/36985758/142215302-88e8748e-2af7-46ce-8ac0-84f52cf51203.png)
**Comparison of DeepProjection with maximum intensity projection.** A: Maximum intensity projection (MIP) of a single stack (8 slices, 1 µm z-distance) of images of the dorsal opening of a Drosophila embryo during dorsal closure, cell boundaries labeled with Cadherin-GFP. B: MIP of as single stack (53 slices, 2 µm z-distance) of images of a zebrafish periderm labeled with labeled krt4-lyn-GFP. A’-D’: y-z cuts of 3D image stacks at red dashed line in A-B. C’ and D’ show the masked stack with the manifolds predicted by DP. C, D: DeepProjection (DP) results from the same stacks of Drosophila and zebrafish embryo. E, F: Zoom into amnioserosal tissue in Drosophila comparing MIP (E) and DP (F), showing successful masking of yolk granules and gut tissue underneath the amnioserosal tissue. G, H: Zoom into zebrafish embryo comparing MIP (G) and DP (H), showing masking of underlying epithelial tissue layer. Scale bars: A, C 50 µm; B, D 100 µm; E-F 10 µm; G, H: 50 µm.

# Installation
### Install CUDA toolkit
IMPORTANT: Convolutional neural networks run much faster on NVIDIA-GPUs than on CPUs. To enable training and prediction on GPUs, users need to install [CUDA toolkit](https://developer.nvidia.com/cuda-toolkit) and the corresponding version of PyTorch from the [official PyTorch website](https://pytorch.org/get-started/locally/). Select the correct version of CUDA on this webpage and run the command in your terminal. DeepProjection was tested with CUDA 11.0 and Pytorch 1.7.1.

### Python Package Index (PyPI)
To be added soon (Dec 21)

### Manual installation
1. Download the code using the GitHub [.zip](https://github.com/danihae/DeepProjection/archive/refs/heads/main.zip) download option or clone repository.
2. Install all Python packages listed in [requirements.txt](requirements.txt)

# Usage
### Import package 
`import deepprojection as dp`

### Training and prediction of stacks
A detailed instruction on how to train and use DeepProjection can by found [here](Quickstart_training_and_prediction.ipynb).

### Graphical User Interface
Run GUI in terminal by `python your_path/deepprojection/GUI.py`. Make sure to use an environment with all necessary packages installed.

# License

This work is licensed under the [GPL3 License](LICENSE).
