# EEG Noise Reduction Based on the Multi-Scale Temporal Propagation Network
---
## Environment and Install
---
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
---
We evaluate our model in EEGDenoiseNet dataset and SS2016 dataset.

The EEGDenoiseNet dataset is from article [EEGdenoiseNet: A benchmark dataset for end-to-end deep learning solutions of EEG denoising](https://github.com/ncclabsustech/EEGdenoiseNet).

The SS2016 dataset is from article [EEGDiR: Electroencephalogram denoising network for temporal information storage and global modeling through Retentive Network](https://github.com/woldier/EEGDiR).

Any information about data processing can be viewed in the `get_datasets` function in the `eeg_set.py` file above.

The processed data can be viewed [here](https://gin.g-node.org/gordon-won/MSTP-Net_pre-trained_model/), under the `processed_data` folder. There are 12 files in total, 6 for EEGDenoiseNet dataset and 6 for SS2016 dataset. These 12 files are placed under the `data` folder.

The file structure is as follows:

```python
|-MSTP-Net
	|-configs
	|-models
	|-eeg_metric.py
	|-eeg_set.py
	|-eeg_utils.py
	|-main_train.py
	|-script_draw_graphs.py
	|-script_draw_saliency_map.py
|-data
	|-SS2016_EOG
		|-train
			|-data-00000-of-00001.arrow
			|-data-00000-of-00002.arrow
			|-data-00001-of-00002.arrow
			|-dataset_info.json
			|-state.json
		|-test
			|-data-00000-of-00001.arrow
			|-dataset_info.json
			|-state.json
	|-EEG_all_epochs.npy
	|-EOG_all_epochs.npy
	|-EMG_all_epochs.npy
	|-EEG_all_epochs_512hz.npy
	|-EMG_all_epochs_512hz.npy
```

## Training
---
```bash
cd MSTP-Net
python main_train.py -c path_to_config
```

Make sure that in the `config` file: 

1. `mode: ‘train’`. Or we can add -m train to the end of the above instruction.

## Evaluation
---
```bash
python main_train.py -c path_to_config
```

Make sure that in the config file:

1. `mode: ‘eval’`. Or we can add `-m eval` to the end of the above instruction.
2. `model_name` is the model you want to use.
3. `evaluation.model_path` is the path of the ckpt file, and preferably an absolute path.

The pre-trained models corresponding to the four artifacts are available [here.](https://gin.g-node.org/gordon-won/MSTP-Net_pre-trained_model)
## Reference
---
[Conv-TasNet code](https://github.com/JusperLee/Conv-TasNet) && [Dual-RNN code](https://github.com/JusperLee/Dual-Path-RNN-Pytorch)
