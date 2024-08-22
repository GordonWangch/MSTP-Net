import numpy as np
import os
import math


def calculate_snr(contaminate, clean):
  noise = contaminate - clean
  signal = clean
  snr = 10 * np.log10(np.sum(signal ** 2) / np.sum(noise ** 2))
  return snr

def _calculate_coe(signal, noise, snr):
  '''Calculate coe'''
  coe = _get_rms(signal) / (_get_rms(noise) * snr)

  return coe


def _get_rms(records):
  return math.sqrt(sum([x ** 2 for x in records]) / len(records))



if __name__ == '__main__':
  source_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  data_dir = os.path.join(source_dir, 'data', '04-EEG')
  # noise_type = ['EMG', 'EOG', 'EMG_EOG', 'Semi']
  noise_type = ['Semi']
  snr_db = -2

  for nt in noise_type:
    if nt != 'Semi':
      if nt == 'EMG': index = 21
      elif nt == 'EOG': index = 213
      else : index = 138

      eeg_path = os.path.join(data_dir, f'{nt}(EEG).npy')
      noise_path = os.path.join(data_dir, f'{nt}(noise).npy')

      eeg = np.load(eeg_path, allow_pickle=True)
      noise = np.load(noise_path, allow_pickle=True)

      train_num = round(0.8 * eeg.shape[0])
      validation_num = round((eeg.shape[0] - train_num) / 2)

      eeg = eeg[train_num + validation_num:, :]
      noise = noise[train_num + validation_num:, :]

      coe = _calculate_coe(eeg[index], noise[index], 10 ** (0.1 * snr_db))

      contaminate = eeg[index] + coe * noise[index]
      clean = eeg[index]

      std_value = np.std(contaminate)

      contaminate = contaminate / std_value
      clean = clean / std_value

      save_result = {
        'contaminated': contaminate,
        'clean': clean,
        'eeg': eeg[index],
        'noise': noise[index],
      }

    else:
      index = 3 # 3, 13, 28, 33
      contaminate_path = os.path.join(data_dir, 'Semi-simulated_test_feature.npy')
      clean_path = os.path.join(data_dir, 'Semi-simulated_test_target.npy')

      contaminate = np.load(contaminate_path, allow_pickle=True)
      clean = np.load(clean_path, allow_pickle=True)
      snr = [calculate_snr(co, cl) for co, cl in zip(contaminate, clean)]

      contaminated = contaminate[index]
      clean = clean[index]
      save_result = {
        'contaminated': contaminated,
        'clean': clean,
      }


    np.save(os.path.join(data_dir, f'test_data_{nt}.npy'), save_result)

