import numpy as np
import matplotlib.pyplot as plt
import os
import re

from copy import copy, deepcopy
from scipy import signal




def draw_snr_db(metric_dict, snr_db_list, figure_size, networks, noise_type):
  fig, ax = plt.subplots(1, 4, figsize=figure_size)
  # colors = ['#934B43', '#f1d77e', '#6f6f6f', '#b883d4', '#84ba42', '#4a5f7e', '#496c88', '#f6cae5', '#96cccb', '#a1a9d0']
  colors = [
    (38, 70, 83),
    (42, 157, 142),
    (233, 196, 107),
    (243, 162, 97),
    (230, 111, 81),
    (183, 181, 160),
    (131, 64, 38),
    (75, 116, 178),
  ]
  colors = [tuple(val / 255.0 for val in color) for color in colors]
  marker = 'h'
  alpha = 0.8

  for idx, net in enumerate(networks):
    rrmse_temporal_list = metric_dict['RRMSE_temporal'][net]
    rrmse_spectrum_list = metric_dict['RRMSE_spectrum'][net]
    cc_list = np.abs(metric_dict['CC'][net]).tolist()
    snr_list = metric_dict['SNR'][net]

    linewidth = 2 if net == networks[0] else 1

    if net == 'proposed': net = 'DPCD-Net (proposed)'

    ax[0].plot(snr_db_list, rrmse_temporal_list, label=net, color=colors[idx], linewidth=linewidth, marker=marker, alpha=alpha)
    ax[1].plot(snr_db_list, rrmse_spectrum_list, label=net, color=colors[idx], linewidth=linewidth, marker=marker, alpha=alpha)
    ax[2].plot(snr_db_list, cc_list, label=net, color=colors[idx], linewidth=linewidth, marker=marker, alpha=alpha)
    ax[3].plot(snr_db_list, snr_list, label=net, color=colors[idx], linewidth=linewidth, marker=marker, alpha=alpha)


  ax[0].set_title('RRMSE Temporal')
  ax[1].set_title('RRMSE Spectrum')
  ax[2].set_title('CC')
  ax[3].set_title('SNR')
  
  if noise_type == 'EMG_EOG': 
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(networks))

  # if noise_type == 'EMG_EOG': noise_type = 'EMG+EOG'
  # fig.suptitle(f'Noise Type: {noise_type}')
  plt.tight_layout(rect=[0, 0.1, 1, 0.9])

  # plt.show()

  source_dir = os.path.dirname(os.path.abspath(__file__))
  save_dir = os.path.join(source_dir, 'figures')
  plt.savefig(os.path.join(save_dir, f'comparation_{noise_type}.svg'), bbox_inches='tight')


def draw_snr_db_ablation(metric_dict, snr_db_list, figure_size, networks, noise_type):
  fig, ax = plt.subplots(1, 4, figsize=figure_size)
  # colors = ['#934B43', '#f1d77e', '#6f6f6f', '#b883d4', '#84ba42', '#4a5f7e', '#496c88', '#f6cae5', '#96cccb', '#a1a9d0']
  colors = [
    (38, 70, 83),
    (42, 157, 142),
    (233, 196, 107),
    (243, 162, 97),
    (230, 111, 81),
    (183, 181, 160),
    (131, 64, 38),
    (75, 116, 178),
  ]
  colors = [tuple(val / 255.0 for val in color) for color in colors]
  marker = 'h'
  alpha = 0.8

  for idx, net in enumerate(networks):
    rrmse_temporal_list = metric_dict['RRMSE_temporal'][net]
    rrmse_spectrum_list = metric_dict['RRMSE_spectrum'][net]
    cc_list = np.abs(metric_dict['CC'][net]).tolist()
    snr_list = metric_dict['SNR'][net]

    linewidth = 2 if net == networks[0] else 1

    if net == 'proposed': net = 'DPCD-Net (proposed)'
    elif net == 'proposed_all_local': net = 'Variant Model A'
    elif net == 'proposed_all_global': net = 'Variant Model B'
    elif net == 'proposed_dilated_1': net = 'Variant Model C'
    elif net == 'proposed_nonalternation': net = 'Variant Model D'
    elif net == 'proposed_nonalternation_global_first': net = 'Variant Model E'

    ax[0].plot(snr_db_list, rrmse_temporal_list, label=net, color=colors[idx], linewidth=linewidth, marker=marker, alpha=alpha)
    ax[1].plot(snr_db_list, rrmse_spectrum_list, label=net, color=colors[idx], linewidth=linewidth, marker=marker, alpha=alpha)
    ax[2].plot(snr_db_list, cc_list, label=net, color=colors[idx], linewidth=linewidth, marker=marker, alpha=alpha)
    ax[3].plot(snr_db_list, snr_list, label=net, color=colors[idx], linewidth=linewidth, marker=marker, alpha=alpha)


  ax[0].set_title('RRMSE Temporal')
  ax[1].set_title('RRMSE Spectrum')
  ax[2].set_title('CC')
  ax[3].set_title('SNR')
  
  handles, labels = ax[0].get_legend_handles_labels()
  fig.legend(handles, labels, loc='lower center', ncol=len(networks))

  # if noise_type == 'EMG_EOG': noise_type = 'EMG+EOG'
  # fig.suptitle(f'Noise Type: {noise_type}')
  plt.tight_layout(rect=[0, 0.1, 1, 0.9])

  # plt.show()

  source_dir = os.path.dirname(os.path.abspath(__file__))
  save_dir = os.path.join(source_dir, 'figures')
  plt.savefig(os.path.join(save_dir, f'ablation_{noise_type}.svg'), bbox_inches='tight')


def draw_amplitude_and_psd(contaminate_eeg, clean_eeg, pred_eeg, noise_type, network, figsize=(12, 12)):

  def signal_psd(data_in, fs, noise_type):
    fft_length = 1024 if noise_type == 'EMG' else 512
    f, pxx = signal.welch(data_in, fs, nfft=fft_length, nperseg=fft_length)
    return f, 10*np.log10(pxx) 

  fig, ax = plt.subplots(2, 1, figsize=figsize)
  sample_rate = 512 if noise_type == 'EMG' else 256
  linewidth = 2

  # Amplitude  
  x = np.arange(0, len(contaminate_eeg) / sample_rate, 1 / sample_rate)

  ax[0].plot(x, contaminate_eeg, label='Contaminate EEG', color='gray', alpha=0.2, linewidth=linewidth)
  ax[0].plot(x, clean_eeg, label='Clean EEG', color='r', alpha=0.5, linewidth=linewidth)
  ax[0].plot(x, pred_eeg, label='Pred EEG', color='b', alpha=0.5, linewidth=linewidth)
  # ax[0].set_title('Time Domain (s)')

  # ax[0].set_ylabel('Time Domain')
  ax[0].set_xticks([])
  ax[0].set_yticks([])

  # PSD
  psd_x, psd_contaminate = signal_psd(contaminate_eeg, sample_rate, noise_type)
  psd_x, psd_clean = signal_psd(clean_eeg, sample_rate, noise_type)
  psd_x, psd_pred = signal_psd(pred_eeg, sample_rate, noise_type)
  hz = 80 if noise_type != 'Semi' else 40
  point = hz * 2

  ax[1].plot(psd_x[:point], psd_contaminate[:point], label='Contaminate EEG', color='gray', alpha=0.2, linewidth=linewidth)
  ax[1].plot(psd_x[:point], psd_clean[:point], label='Clean EEG', color='r', alpha=0.5, linewidth=linewidth)
  ax[1].plot(psd_x[:point], psd_pred[:point], label='Pred EEG', color='b', alpha=0.5, linewidth=linewidth)
  # ax[1].set_title('Frequency Domain (Hz)')

  # ax[1].set_ylabel('Frequency Domain')
  ax[1].set_xticks([])
  ax[1].set_yticks([])

  # fig.suptitle(f'Noise Type: {noise_type} Network: {network}')
  # plt.show()

  source_dir = os.path.dirname(os.path.abspath(__file__))
  save_dir = os.path.join(source_dir, 'figures')
  plt.savefig(os.path.join(save_dir, f'qualitative_{noise_type}_{network}.svg'), bbox_inches='tight')


def draw_contaminate_and_clean(contaminate, clean, pred):
  sample_rate = 512 
  linewidth = 2

  x = np.arange(0, len(contaminate) / sample_rate, 1 / sample_rate)

  fig, ax = plt.subplots(1, 3)

  ax[0].plot(x, contaminate, label='Contaminate EEG', color='black', alpha=0.8, linewidth=linewidth)
  ax[1].plot(x, clean, label='Clean EEG', color='black', alpha=0.8, linewidth=linewidth)
  ax[2].plot(x, pred, label='Pred EEG', color='black', alpha=0.8, linewidth=linewidth)
  # ax.set_xticks([])
  # ax.set_yticks([])
  plt.show()


def parse_npy_files(metric_result_dir, noise_type, networks, average_dim='snr'):
  all_files = os.listdir(metric_result_dir)
  average_dim = 1 if average_dim == 'net' else 0
  whole_metric_dict = {}

  for nt in noise_type:
    whole_metric_dict[nt] = {
      'RRMSE_temporal': {},
      'RRMSE_spectrum': {},
      'CC': {},
      'SNR': {},
      }

    for net in networks:
      pattern = re.compile(f'{nt} {net} \d+\.npy')
      npy_file_names = sorted([f for f in all_files if pattern.match(f)], key=lambda x: int(x.split(' ')[-1].split('.')[0]))
      rrmse_t, rrmse_s, cc, snr = [], [], [], []
      print(f'[INFO] {nt} {net} = {npy_file_names}')

      for name in npy_file_names:
        npy_file = os.path.join(metric_result_dir, name)
        metric_dict = np.load(npy_file, allow_pickle=True).item()

        if nt != 'Semi':
          rrmse_t.append([metric_dict[k]['RRMSE_temporal'] for k in metric_dict.keys()])
          rrmse_s.append([metric_dict[k]['RRMSE_spectrum'] for k in metric_dict.keys()])
          cc.append([metric_dict[k]['CC'] for k in metric_dict.keys()])
          snr.append([metric_dict[k]['SNR'] for k in metric_dict.keys()])
        else:
          rrmse_t.append([metric_dict['RRMSE_temporal']])
          rrmse_s.append([metric_dict['RRMSE_spectrum']])
          cc.append([metric_dict['CC']])
          snr.append([metric_dict['SNR']])

      whole_metric_dict[nt]['RRMSE_temporal'][net] = np.mean(rrmse_t, axis=average_dim)
      whole_metric_dict[nt]['RRMSE_spectrum'][net] = np.mean(rrmse_s, axis=average_dim)
      whole_metric_dict[nt]['CC'][net] = np.mean(cc, axis=average_dim)
      whole_metric_dict[nt]['SNR'][net] = np.mean(snr, axis=average_dim)
  
  return whole_metric_dict  


def parse_npy_files_in_snr(metric_result_dir, noise_type, networks):
  whole_metric_dict = {}

  for nt in noise_type:
    whole_metric_dict[nt] = {
      'RRMSE_temporal': {},
      'RRMSE_spectrum': {},
      'CC': {},
      'SNR': {},
      }
    
    for net in networks:
      npy_file = os.path.join(metric_result_dir, f'{nt} {net}.npy')
      rrmse_t, rrmse_s, cc, snr = [], [], [], []

      metric_dict = np.load(npy_file, allow_pickle=True).item()

      rrmse_t = [metric_dict[k]['RRMSE_temporal'] for k in metric_dict.keys()]
      rrmse_s = [metric_dict[k]['RRMSE_spectrum'] for k in metric_dict.keys()]
      cc = [metric_dict[k]['CC'] for k in metric_dict.keys()]
      snr = [metric_dict[k]['SNR'] for k in metric_dict.keys()]

      whole_metric_dict[nt]['RRMSE_temporal'][net] = rrmse_t
      whole_metric_dict[nt]['RRMSE_spectrum'][net] = rrmse_s
      whole_metric_dict[nt]['CC'][net] = cc
      whole_metric_dict[nt]['SNR'][net] = snr
  
  return whole_metric_dict  


def get_average_metric(whole_metric_dict):
  whole_metric_dict = deepcopy(whole_metric_dict)
  for noise_type in whole_metric_dict.keys():
    print('='*100)
    for metric in whole_metric_dict[noise_type].keys():
      for net in whole_metric_dict[noise_type][metric].keys():
        values = whole_metric_dict[noise_type][metric][net]
        whole_metric_dict[noise_type][metric][net] = [np.mean(values), np.std(values)]
        print(f'[{noise_type.center(10)}] [{metric.center(20)}] [{net.center(20)}] = {np.mean(values): .4f} ±{np.std(values): .4f}')

  return whole_metric_dict


def demo_draw_snr_db():
  current_dir = os.path.dirname(os.path.abspath(__file__))
  npy_dir = os.path.join(current_dir, 'metric_result')

  noise_type = ['EMG', 'EOG', 'EMG_EOG']
  networks = ['proposed', 'fcNN', 'Simple_CNN', 'Complex_CNN', 'Novel_CNN', 'RNN_lstm', 'EEGDnet', 'EEGDiR']

  whole_metric_dict = parse_npy_files_in_snr(npy_dir, noise_type, networks)

  for nt in noise_type:
    snr_db_list = list(range(-7, 4 + 1)) if nt == 'EMG' else list(range(-7, 2 + 1))
    draw_snr_db(whole_metric_dict[nt], snr_db_list, (16, 4), networks, nt)


def demo_draw_ablation():
  current_dir = os.path.dirname(os.path.abspath(__file__))
  npy_dir = os.path.join(current_dir, 'ablation_result')

  noise_type = ['EOG']
  networks = ['proposed', 'proposed_all_local', 'proposed_all_global', 'proposed_dilated_1', 'proposed_nonalternation', 'proposed_nonalternation_global_first']
  whole_metric_dict = parse_npy_files(npy_dir, noise_type, networks)

  for nt in noise_type:
    snr_db_list = list(range(-7, 4 + 1)) if nt == 'EMG' else list(range(-7, 2 + 1))
    draw_snr_db_ablation(whole_metric_dict[nt], snr_db_list, (16, 4), networks, nt)


def demo_draw_amplitude_and_psd():
  current_dir = os.path.dirname(os.path.abspath(__file__))
  npy_dir = os.path.join(current_dir, 'qualitative_result')

  noise_type = ['EMG', 'EOG', 'EMG_EOG', 'Semi']
  # noise_type = ['Semi']
  networks = ['proposed', 'fcNN', 'Simple_CNN', 'Complex_CNN', 'Novel_CNN', 'RNN_lstm', 'EEGDnet', 'EEGDiR']

  for nt in noise_type:
    for net in networks:
      file_path = os.path.join(npy_dir, f'{nt} {net}.npy')
      data = np.load(file_path, allow_pickle=True).item()
      contaminate_eeg = data['contaminate'][0, 0]
      clean_eeg = data['clean'][0, 0]
      pred_eeg = data['pred'][0, 0]

      draw_amplitude_and_psd(contaminate_eeg, clean_eeg, pred_eeg, nt, net)


def demo_draw_contaminate_and_clean():
  current_dir = os.path.dirname(os.path.abspath(__file__))
  npy_dir = os.path.join(current_dir, 'qualitative_result')

  # noise_type = ['EMG', 'EOG', 'EMG_EOG', 'Semi']
  noise_type = 'EMG'
  network = 'proposed'

  file_path = os.path.join(npy_dir, f'{noise_type} {network}.npy')
  data = np.load(file_path, allow_pickle=True).item()
  contaminate_eeg = data['contaminate'][0, 0]
  clean_eeg = data['clean'][0, 0]
  pred_eeg = data['pred'][0, 0]

  draw_contaminate_and_clean(contaminate_eeg, clean_eeg, pred_eeg)


def demo_rename():
  current_dir = os.path.dirname(os.path.abspath(__file__))
  npy_dir = os.path.join(current_dir, 'metric_result')

  noise_type = ['EMG', 'EOG', 'EMG_EOG']
  # networks = ['fcNN', 'Simple_CNN', 'Complex_CNN', 'Novel_CNN', 'RNN_lstm', 'EEGDnet', 'EEGDiR']
  networks = ['EEGDnet']

  for nt in noise_type:
    for net in networks:
      npy_file = os.path.join(npy_dir, f'{nt} {net}.npy')
      metric_dict = np.load(npy_file, allow_pickle=True).item()

      for key in metric_dict:
        metric_dict[key]['RRMSE_temporal'] = metric_dict[key].pop('rrmse_t')
        metric_dict[key]['RRMSE_spectrum'] = metric_dict[key].pop('rrmse_s')
        metric_dict[key]['CC'] = metric_dict[key].pop('cc')
        metric_dict[key]['SNR'] = metric_dict[key].pop('snr')

      np.save(npy_file, metric_dict)


def demo_calculate_average_metric():
  current_dir = os.path.dirname(os.path.abspath(__file__))
  npy_dir = os.path.join(current_dir, 'eval_result')

  noise_type = ['EMG', 'EOG', 'EMG_EOG', 'Semi']
  networks = ['proposed']
  whole_metric_dict = parse_npy_files(npy_dir, noise_type, networks, 'net')
  average_metric_dict = get_average_metric(whole_metric_dict)


def demo_get_average_metric_in_snr():
  '''
  {noise_type} proposed x.npy (x from [0, 10]) in eval_result directory
  -> {noise_type} proposed.npy in metric_result directory
  '''
  current_dir = os.path.dirname(os.path.abspath(__file__))
  npy_dir = os.path.join(current_dir, 'eval_result')
  save_dir = os.path.join(current_dir, 'metric_result')

  # noise_type = ['EMG', 'EOG', 'EMG_EOG']
  noise_type = ['EMG_EOG']
  networks = ['proposed']
  whole_metric_dict = parse_npy_files(npy_dir, noise_type, networks, 'snr')

  for nt in noise_type:
    snr_db = list(range(-7, 4 + 1)) if nt == 'EMG' else list(range(-7, 2 + 1))
    for net in networks:
      save_metric_dict = {}
      for idx, snr in enumerate(snr_db):
        save_metric_dict[f'{snr}'] = {
          'RRMSE_temporal': whole_metric_dict[nt]['RRMSE_temporal'][net][idx],
          'RRMSE_spectrum': whole_metric_dict[nt]['RRMSE_spectrum'][net][idx],
          'CC': whole_metric_dict[nt]['CC'][net][idx],
          'SNR': whole_metric_dict[nt]['SNR'][net][idx],
        }
      save_path = os.path.join(save_dir, f'{nt} {net}.npy') 

      np.save(save_path, save_metric_dict)



if __name__ == '__main__':
  # demo_draw_snr_db()
  # demo_draw_ablation()
  # demo_draw_amplitude_and_psd()
  demo_draw_contaminate_and_clean()

  # demo_calculate_average_metric()
  # demo_get_average_metric_in_snr()
  # demo_rename()
  
