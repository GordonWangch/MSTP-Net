import os
import torch
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import pickle
import random
import itertools
import math

from datasets import Dataset



## EEGDenoiseNet traditional Processing
class EEGSET(torch.utils.data.Dataset):

	def __init__(
			self,
			eeg: np.ndarray,
			noise: np.ndarray,
			noise_type=None,
			name=None,
			data_dir=None,
	):
		self.eeg = eeg
		self.noise = noise

		self.noise_type = noise_type
		self.name = name
		self.data_dir = data_dir

		self.snr_db = [-7, 4] if self.noise_type == 'EMG' else [-7, 2]
		self.snr = 10 ** (0.1 * np.random.uniform(self.snr_db[0], self.snr_db[1], self.eeg.shape[0]))

		dataset_name = f"{self.name}_{self.noise_type}_{len(self)}.pkl"
		# self.save(os.path.join(self.data_dir, f"{dataset_name}"))


	def __getitem__(self, index):
		eeg = self.eeg[index]
		noise = self.noise[index]
		snr = self.snr[index]

		coe = self._calculate_coe(eeg, noise, snr)
		noise_eeg = eeg + noise * coe

		# Normalization
		std_value = np.std(noise_eeg)

		feature = noise_eeg / std_value
		target = eeg / std_value

		feature = np.expand_dims(feature, axis=0).astype(np.float32)
		target = np.expand_dims(target, axis=0).astype(np.float32)

		if 'Test' in self.name:
			return feature, target, 10 * np.log10(snr)
		return feature, target


	def _get_data_with_fixed_snr(self, snr_db, batch_size):
		assert snr_db <= self.snr_db[1] and snr_db >= self.snr_db[0]
		snr = 10 ** (0.1 * snr_db)

		eeg = self.eeg
		noise = self.noise

		noise_eeg = [e + n * self._calculate_coe(e, n, snr) for e, n in zip(eeg, noise)]
		std_values = np.std(noise_eeg, axis=1)

		features = [ne / std for ne, std in zip(noise_eeg, std_values)]
		targets = [e / std for e, std in zip(eeg, std_values)]

		features = np.expand_dims(features, axis=1).astype(np.float32)
		targets = np.expand_dims(targets, axis=1).astype(np.float32)

		# Split
		length = len(features)	
		numbers = list(range(length))
		indexes = [numbers[i:i + batch_size] for i in range(0, length, batch_size)]

		return [[features[idx], targets[idx]] for idx in indexes]


	def __len__(self):
		return len(self.eeg)


	def _calculate_coe(self, signal, noise, snr):
		'''Calculate coe'''
		coe = self._get_rms(signal) / (self._get_rms(noise) * snr)

		return coe


	def _get_rms(self, records):
		return math.sqrt(sum([x ** 2 for x in records]) / len(records))


	def _check_data(self):
		pass


	def save(self, file_path):
		with open(file_path, "wb") as f:
			pickle.dump(self, f)
			print(f"Dataset have saved to {file_path}")


	@staticmethod
	def load(file_path):
		with open(file_path, "rb") as f:
			dataset = pickle.load(f)
		assert isinstance(dataset, EEGSET)
		print(f"Loading dataset: {file_path}")

		return dataset


class EEGSET_SEMI(torch.utils.data.Dataset):

	def __init__(
			self,
			eeg: np.ndarray,
			noise_eeg: np.ndarray,
			dataset_type='Semi-simulated',
			noise_type=None,
			name=None,
			data_dir=None,
	):
		self.eeg = eeg
		self.noise_eeg = noise_eeg

		self.dataset_type = dataset_type
		self.noise_type = noise_type
		self.name = name
		self.data_dir = data_dir

		dataset_name = f"{self.name}_{self.noise_type}_{len(self)}.pkl"
		# self.save(os.path.join(self.data_dir, f"{dataset_name}"))


	def __getitem__(self, index):
		target = self.eeg[index]
		feature = self.noise_eeg[index]

		feature = np.expand_dims(feature, axis=0).astype(np.float32)
		target = np.expand_dims(target, axis=0).astype(np.float32)

		if 'Test' in self.name:
			return feature, target, 1

		return feature, target


	def __len__(self):
		return len(self.eeg)


	def save(self, file_path):
		with open(file_path, "wb") as f:
			pickle.dump(self, f)
			print(f"Dataset have saved to {file_path}")


	@staticmethod
	def load(file_path):
		with open(file_path, "rb") as f:
			dataset = pickle.load(f)
		assert isinstance(dataset, EEGSET)
		print(f"Loading dataset: {file_path}")

		return dataset


def get_datasets(
		dataset_type,
		noise_type,
		data_dir,
		ratio_of_dataset,
		dataset_names=['TrainSet', 'ValidSet', 'TestSet'],
		combine_num=10,
):
	assert dataset_type in ['EEGDenoiseNet', 'Semi-simulated']
	assert noise_type in ['EMG', 'EOG', 'EMG_EOG', 'Semi']

	ratio = [int(r) for r in ratio_of_dataset.split(':')]

	if dataset_type == 'EEGDenoiseNet':
		# EEGDenoiseNet dataset
		signal, noise = _load_from_npy(dataset_type, noise_type, data_dir)
		train_eeg, train_noise, val_eeg, val_noise, test_eeg, test_noise = _signal_preproc_denoisenet(
			signal, noise, combine_num, ratio)

		trainset = EEGSET(train_eeg, train_noise, noise_type, dataset_names[0], data_dir)
		validset = EEGSET(val_eeg, val_noise, noise_type, dataset_names[1], data_dir)
		testset = EEGSET(test_eeg, test_noise, noise_type, dataset_names[2], data_dir)
	else:
		# EEGDiR SS2016 dataset
		train_noise_eeg, train_eeg, val_noise_eeg, val_eeg, test_noise_eeg, test_eeg = _get_semi_data_from_eegdir(data_dir)

		trainset = EEGSET_SEMI(train_eeg, train_noise_eeg, dataset_type, noise_type, dataset_names[0], data_dir)	
		validset = EEGSET_SEMI(val_eeg, val_noise_eeg, dataset_type, noise_type, dataset_names[1], data_dir)
		testset = EEGSET_SEMI(test_eeg, test_noise_eeg, dataset_type, noise_type, dataset_names[2], data_dir)

	return trainset, validset, testset 


def _load_from_npy(dataset_type, noise_type, data_dir):

	if dataset_type == 'EEGDenoiseNet':
		EEG_all_path = os.path.join(data_dir, f'{noise_type}(EEG).npy')
		noise_all_path = os.path.join(data_dir, f'{noise_type}(noise).npy')

		if os.path.exists(EEG_all_path) and os.path.exists(noise_all_path):
			EEG_all = np.load(EEG_all_path, allow_pickle=True)
			noise_all = np.load(noise_all_path, allow_pickle=True)
		else:
			if noise_type == 'EOG':
				EEG_all = np.load(os.path.join(data_dir, 'EEG_all_epochs.npy'), allow_pickle=True)
				noise_all = np.load(os.path.join(data_dir, 'EOG_all_epochs.npy'), allow_pickle=True)
			elif noise_type == 'EMG_EOG':
				EEG_all = np.load(os.path.join(data_dir, 'EEG_all_epochs.npy'), allow_pickle=True)
				noise_EOG = np.load(os.path.join(data_dir, 'EOG_all_epochs.npy'), allow_pickle=True)
				noise_EMG = np.load(os.path.join(data_dir, 'EMG_all_epochs.npy'), allow_pickle=True)[:noise_EOG.shape[0]]
				noise_all = noise_EOG + noise_EMG
			else:
				EEG_all = np.load(os.path.join(data_dir, 'EEG_all_epochs_512hz.npy'), allow_pickle=True)
				noise_all = np.load(os.path.join(data_dir, 'EMG_all_epochs_512hz.npy'), allow_pickle=True)

			EEG_all = np.squeeze(_random_signal(EEG_all, 1))
			noise_all = np.squeeze(_random_signal(noise_all, 1))

			if noise_type == 'EMG':
				reuse_num = noise_all.shape[0] - EEG_all.shape[0]
				reuse_index = random.sample(list(range(EEG_all.shape[0])), reuse_num)
				EEG_all = np.concatenate((EEG_all, EEG_all[reuse_index, :]), axis=0)
			else:
				EEG_all = EEG_all[:noise_all.shape[0]]

			assert EEG_all.shape[0] == noise_all.shape[0]

			np.save(EEG_all_path, EEG_all)
			np.save(noise_all_path, noise_all)

		return EEG_all, noise_all
	else:
		noise_data = np.load(os.path.join(data_dir, 'signal_Semi-simulated EOG.npy'), allow_pickle=True)
		clean_data = np.load(os.path.join(data_dir, 'reference_Semi-simulated EOG.npy'), allow_pickle=True)

		EEG_all = np.array(clean_data).reshape((-1, 540))
		noise_EEG_all = np.array(noise_data).reshape((-1, 540))

		return EEG_all, noise_EEG_all


def _random_split(total_size, ratio, rand=False):
	ratio = [r / sum(ratio) for r in ratio]
	sizes = [round(total_size * r) for r in ratio]
	if sum(sizes) != total_size:
		sizes[-1] = total_size - sum(sizes[:-1])

	index_all = list(range(total_size))
	if rand: random.shuffle(index_all)

	iter_index = iter(index_all)
	index_all = [list(itertools.islice(iter_index, s)) for s in sizes]

	return index_all


def _random_signal(signal, combin_num):
	# Random disturb and augment signal
	random_result=[]

	for i in range(combin_num):
		random_num = np.random.permutation(signal.shape[0])
		shuffled_dataset = signal[random_num, :]
		shuffled_dataset = shuffled_dataset.reshape(signal.shape[0],signal.shape[1])
		random_result.append(shuffled_dataset)

	random_result  = np.array(random_result).reshape(-1, signal.shape[-1])

	return  random_result


def _get_rms(records):
	return math.sqrt(sum([x ** 2 for x in records]) / len(records))


def _signal_preproc_denoisenet(EEG_data, noise_data, combin_num, ratio=[8, 1, 1]):
	train_index, val_index, test_index = _random_split(EEG_data.shape[0], ratio, rand=False)	

	train_eeg, train_noise = EEG_data[train_index, :], noise_data[train_index, :]
	val_eeg, val_noise = EEG_data[val_index, :], noise_data[val_index, :]
	test_eeg, test_noise = EEG_data[test_index, :], noise_data[test_index, :]

	train_eeg, train_noise = _random_signal(train_eeg, combin_num), _random_signal(train_noise, combin_num)
	val_eeg, val_noise = _random_signal(val_eeg, combin_num), _random_signal(val_noise, combin_num)
	test_eeg, test_noise = _random_signal(test_eeg, combin_num), _random_signal(test_noise, combin_num)

	return train_eeg, train_noise, val_eeg, val_noise, test_eeg, test_noise


def _get_semi_data_from_eegdir(data_dir):
	train_contaminate_path = os.path.join(data_dir, 'Semi-simulated_train_feature.npy')
	train_clean_path = os.path.join(data_dir, 'Semi-simulated_train_target.npy')
	val_contaminate_path = os.path.join(data_dir, 'Semi-simulated_val_feature.npy')
	val_clean_path = os.path.join(data_dir, 'Semi-simulated_val_target.npy')
	test_contaminate_path = os.path.join(data_dir, 'Semi-simulated_test_feature.npy')
	test_clean_path = os.path.join(data_dir, 'Semi-simulated_test_target.npy')

	if all([os.path.exists(path) for path in [train_contaminate_path, train_clean_path, val_contaminate_path, val_clean_path, test_contaminate_path, test_clean_path]]):
		train_contaminate = np.load(train_contaminate_path)
		train_clean = np.load(train_clean_path)
		val_contaminate = np.load(val_contaminate_path)
		val_clean = np.load(val_clean_path)
		test_contaminate = np.load(test_contaminate_path)
		test_clean = np.load(test_clean_path)
	else:

		train_dir = os.path.join(data_dir, 'SS2016_EOG', 'train')
		test_dir = os.path.join(data_dir, 'SS2016_EOG', 'test')

		train_set = Dataset.load_from_disk(train_dir)
		train_set.set_format('numpy')  
		test_set = Dataset.load_from_disk(test_dir)
		test_set.set_format('numpy')

		train_contaminate, train_clean = train_set['y'], train_set['x']
		test_contaminate, test_clean = test_set['y'], test_set['x']

		val_contaminate, val_clean = test_contaminate[1::2], test_clean[1::2]
		test_contaminate, test_clean = test_contaminate[::2], test_clean[::2]
		# val_num = round(train_contaminate.shape[0] * 0.1)
		# val_contaminate, val_clean = train_contaminate[-val_num:], train_clean[-val_num:]
		# train_contaminate, train_clean = train_contaminate[:-val_num], train_clean[:-val_num]

		np.save(os.path.join(data_dir, 'Semi-simulated_train_feature.npy'), train_contaminate)
		np.save(os.path.join(data_dir, 'Semi-simulated_train_target.npy'), train_clean)
		np.save(os.path.join(data_dir, 'Semi-simulated_val_feature.npy'), val_contaminate)
		np.save(os.path.join(data_dir, 'Semi-simulated_val_target.npy'), val_clean)
		np.save(os.path.join(data_dir, 'Semi-simulated_test_feature.npy'), test_contaminate)
		np.save(os.path.join(data_dir, 'Semi-simulated_test_target.npy'), test_clean)

	return train_contaminate, train_clean, val_contaminate, val_clean, test_contaminate, test_clean



if __name__ == '__main__':
	data_dir = r'/files/gordon/projects/paper_related/data/04-EEG'
	_get_semi_data_from_eegdir(data_dir)
