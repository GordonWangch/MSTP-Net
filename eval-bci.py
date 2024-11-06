import os, sys
source_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(source_dir)
import random
import time
import numpy as np
import argparse
import torch
import torchvision
import yaml
import torch.nn as nn

from thop import profile
from torch.utils.data import DataLoader
from torchinfo import summary
from datetime import datetime
from eeg_set import get_datasets, EEGSET
from models.proposed_net import MSTP_Model
from models.proposed_net_dilate_1 import MSTP_Model_dilated
from models.proposed_net_all_local import MSTP_Model_all_local
from models.proposed_net_all_global import MSTP_Model_all_global
from models.proposed_net_nonalternation import MSTP_Model_nonalternation
from models.proposed_net_nonalternation_global_first import MSTP_Model_nonalternation_global_first
from eeg_utils import Trainer, Evaluator
from tqdm import tqdm
# from torch.utils.tensorboard import SummaryWriter



def parse_arguments():

  def print_dict(d, parent_key=''):
    for k, v in d.items():
      full_key = f"{parent_key}.{k}" if parent_key else k
      if isinstance(v, dict):
        print_dict(v, full_key)
      else:
        print(f'[*] {full_key + ":": <40}{v}')

  def dict2namespace(input_dict):
    namespace = argparse.Namespace()
    for key, value in input_dict.items():
      if isinstance(value, dict):
        new_value = dict2namespace(value)
      else:
        new_value = value
      setattr(namespace, key, new_value)
    return namespace

  parser = argparse.ArgumentParser(description="Your script description")
  parser.add_argument(
    '-c', '--config_path', type=str, 
    # default=r'/z3/home/xai_test/wch/paper_related/03-EEG_DENOISE/configs/cfg-2024-07-09-EEGDenoiseNet-EMG.yml',
    # default=r'/z3/home/xai_test/wch/paper_related/03-EEG_DENOISE/configs/cfg-2024-07-09-EEGDenoiseNet-EOG.yml',
    # default=r'/z3/home/xai_test/wch/paper_related/03-EEG_DENOISE/configs/cfg-2024-07-09-EEGDenoiseNet-EMG_EOG.yml',
    # default=r'/z3/home/xai_test/wch/paper_related/03-EEG_DENOISE/configs/cfg-2024-07-08-Semi.yml',
    default=r'E:\paper_related\MSTP-Net\configs\cfg-2024-07-08-Semi.yml',
    # default=r'E:\paper_related\MSTP-Net\configs\cfg-2024-07-09-EEGDenoiseNet-EOG.yml',
    help='File path of config file')

  parser.add_argument(
    '-m', '--mode_bp', type=str, 
    default=None, 
    choices=['train', 'eval'],
  )

  parser.add_argument(
    '-mn', '--model_name_bp', type=str, 
    default=None, 
    choices=['proposed', 'proposed_dilated_1', 'proposed_all_local', 'proposed_all_global', 'proposed_nonalternation', 'proposed_nonalternation_global_first'],
  )

  args = vars(parser.parse_args())

  with open(args['config_path'], 'r') as f:
    configs = yaml.safe_load(f)

  if args['mode_bp'] is not None:
    configs['mode'] = args['mode_bp']
  
  if args['model_name_bp'] is not None:
    configs['model_name'] = args['model_name_bp']

  task_dir = os.path.dirname(os.path.abspath(__file__))
  source_dir = os.path.dirname(task_dir)
  data_dir = os.path.join(source_dir, 'data')
  filename = os.path.basename(__file__).split('.py')[0]

  current_times = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
  config_name = os.path.basename(args['config_path']).split('.yml')[0]
  noise_type = configs['data_param']['noise_type']
  model_name = configs['model_name']
  ckpt_dir = os.path.join(task_dir, 'checkpoints', f'({config_name})-({noise_type})-({model_name})')
  ckpt_name = f'checkpoint-{noise_type}-({current_times}).pth'

  args['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  args['source_dir'] = source_dir
  args['task_dir'] = task_dir
  args['data_dir'] = data_dir
  args['file_name'] = filename
  args['ckpt_dir'] = ckpt_dir
  args['ckpt_name'] = ckpt_name

  configs.update(args)

  print('[Info] Parameters')
  print_dict(configs)

  configs = dict2namespace(configs)

  return configs

def normalization(raw_data):
  raw_shape = raw_data.shape
  data = raw_data.reshape(-1, raw_shape[-1])
  eeg_mean = np.mean(data, axis=1, keepdims=True)
  eeg_std = np.std(data, axis=1, keepdims=True)
  eeg_standard = (data - eeg_mean) /eeg_std
  eeg_standard = eeg_standard.reshape(raw_shape)

  return eeg_standard

def main():
  # ================================================================== #
  #                      01.Parameter Setting
  # ================================================================== #
  configs = parse_arguments()

  bs = configs.training.batch_size
  data_shape = tuple(configs.data_param.data_shape)
  device = configs.device

  # ================================================================== #
  #                      02.Model
  # ================================================================== #
  if configs.model_name == 'proposed':
    model = MSTP_Model(configs.model_param.proposed).to(device)
  elif configs.model_name == 'proposed_dilated_1':
    model = MSTP_Model_dilated(configs.model_param.proposed_dilated_1).to(device)
  elif configs.model_name == 'proposed_all_local':
    model = MSTP_Model_all_local(configs.model_param.proposed_all_local).to(device)
  elif configs.model_name == 'proposed_all_global':
    model = MSTP_Model_all_global(configs.model_param.proposed_all_global).to(device)
  elif configs.model_name == 'proposed_nonalternation':
    model = MSTP_Model_nonalternation(configs.model_param.proposed_nonalternation).to(device)
  elif configs.model_name == 'proposed_nonalternation_global_first':
    model = MSTP_Model_nonalternation_global_first(configs.model_param.proposed_nonalternation_global_first).to(device)
  else:
    raise ValueError(f'[*] Invalid model name: {configs.model_name}')

  # if configs.mode == 'train':
  flops, params = profile(model, inputs=(torch.randn(1, 1, data_shape[1]).to(device), ))
  print(f'[INFO] FLOPs: {flops / 1000**2: .1f} M')
  print(f'[INFO] Params: {params / 1000**2: .1f} M')

  summary(
    model, 
    input_size=(2, ) + data_shape, 
    depth=7, 
    col_width=20, 
    col_names=("input_size", "output_size", "num_params")
    )

  model = torch.nn.DataParallel(
    model, device_ids=[int(i) for i in configs.gpu_ids.split(',')])

  configs.model = model

  # if configs.model.use_tensorboard:
  #   current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
  #   log_dir = os.path.join(configs.task_dir, 'tensorboard', current_time)
  #   input_tensor_list = [
  #     torch.randn((bs, ) + data_shape).to(device),
  #     torch.randn((bs, )).to(device)
  #   ]
  #
  #   writer = SummaryWriter(log_dir=log_dir)
  #   writer.add_graph(model, input_tensor_list, verbose=True)
  #   writer.close()
  #   print(f'[*] Tensorboard log dir: {os.path.abspath(log_dir)}')


  assert configs.mode == 'eval'
  model.load_state_dict(torch.load(configs.evaluation.model_path))

  test_data_dir = os.path.join(configs.task_dir, 'data')

  files = [f for f in os.listdir(test_data_dir) if f.startswith('bci_raw_data') and 'erp' in f]

  with torch.no_grad():
    for f in files:
      print(f'[INFO] [PID]: {f.split("_")[-1].split(".")[0]}')
      test_data_path = os.path.join(test_data_dir, f)
      save_path = os.path.join(test_data_dir, f"bci_c_results_{f.split('_')[-1]}")
      test_data = np.load(test_data_path, allow_pickle=True)

      test_data = normalization(test_data)


      results = []
      for td in tqdm(test_data):
        td = torch.tensor(td, dtype=torch.float32).unsqueeze(1).to(device)
        results.append(model(td).cpu().detach().numpy())

      results = np.squeeze(results)
      np.save(save_path, results)



if __name__ == '__main__':
  main()



