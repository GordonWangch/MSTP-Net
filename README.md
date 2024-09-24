# EEG Noise Reduction Based on the Multi-Scale Temporal Propagation Network
---
## Environment and Install
The following setup has been used to reproduce this work:
- CUDA toolkit 11.3.1 and CuDNN 8.2.1
- Python == 3.8.18
- pytorch == 1.11.0
- Tensorflow == 2.5.0
- Matplotlib == 3.7.3
- Numpy == 1.24.4
- Scipy == 1.10.1

All the specific required packages are in the `install.txt` above
## Prepare Dataset
We evaluate our model in EEGDenoiseNet dataset and SS2016 dataset.
The EEGDenoiseNet dataset is from article [EEGdenoiseNet: A benchmark dataset for end-to-end deep learning solutions of EEG denoising].(https://github.com/ncclabsustech/EEGdenoiseNet)
The SS2016 dataset is from article [EEGDiR: Electroencephalogram denoising network for temporalinformation storage and global modeling through Retentive Network].(https://github.com/woldier/EEGDiR)

## Training
