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
from models.proposed_net import Dual_Tas_Model
from models.proposed_net_dilate_1 import Dual_Tas_Model_dilated
from models.proposed_net_all_local import Dual_Tas_Model_all_local
from models.proposed_net_all_global import Dual_Tas_Model_all_global
from models.proposed_net_nonalternation import Dual_Tas_Model_nonalternation
from models.proposed_net_nonalternation_global_first import Dual_Tas_Model_nonalternation_global_first
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
    default=r'/z3/home/xai_test/wch/paper_related/03-EEG_DENOISE/configs/cfg-2024-07-09-EEGDenoiseNet-EOG.yml',
    # default=r'/z3/home/xai_test/wch/paper_related/03-EEG_DENOISE/configs/cfg-2024-07-09-EEGDenoiseNet-EMG_EOG.yml',
    # default=r'/z3/home/xai_test/wch/paper_related/03-EEG_DENOISE/configs/cfg-2024-07-08-Semi.yml',
    help='File path of config file')

  parser.add_argument(
    '-m', '--mode_bp', type=str, 
    default=None, 
    choices=['train', 'eval'],
  )

  parser.add_argument(
    '-mn', '--model_name_bp', type=str, 
    default=None, 
    choices=['proposed', 'tasnet', 'proposed_dilated_1', 'proposed_all_local', 'proposed_all_global', 'proposed_without_global', 'proposed_nonalternation', 'proposed_nonalternation_global_first'],
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
    model = Dual_Tas_Model(configs.model_param.proposed).to(device)
  elif configs.model_name == 'proposed_dilated_1':
    model = Dual_Tas_Model_dilated(configs.model_param.proposed_dilated_1).to(device)
  elif configs.model_name == 'proposed_all_local':
    model = Dual_Tas_Model_all_local(configs.model_param.proposed_all_local).to(device)
  elif configs.model_name == 'proposed_all_global':
    model = Dual_Tas_Model_all_global(configs.model_param.proposed_all_global).to(device)
  elif configs.model_name == 'proposed_nonalternation':
    model = Dual_Tas_Model_nonalternation(configs.model_param.proposed_nonalternation).to(device)
  elif configs.model_name == 'proposed_nonalternation_global_first':
    model = Dual_Tas_Model_nonalternation_global_first(configs.model_param.proposed_nonalternation_global_first).to(device)
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


  # ================================================================== #
  #                      03.Data Preprocessing
  # ================================================================== #
  train_set, valid_set, test_set = get_datasets(
    dataset_type=configs.data_param.dataset_type,
    noise_type=configs.data_param.noise_type,
    data_dir=os.path.join(configs.data_dir, '04-EEG'),
    ratio_of_dataset=configs.data_param.ratio_of_dataset,
  )

  train_loader = torch.utils.data.DataLoader(
    dataset=train_set, batch_size=bs, num_workers=configs.data_param.num_workers, shuffle=True, pin_memory=True)
  val_loader = torch.utils.data.DataLoader(
    dataset=valid_set, batch_size=bs, num_workers=configs.data_param.num_workers, shuffle=False, pin_memory=True)
  test_loader = torch.utils.data.DataLoader(
    dataset=test_set, batch_size=bs, num_workers=configs.data_param.num_workers, shuffle=False, pin_memory=True)

  configs.loader = argparse.Namespace()
  configs.loader.train_loader = train_loader
  configs.loader.val_loader = val_loader
  configs.loader.test_loader = test_loader

  # ================================================================== #
  #                      04.Optimizer and Loss
  # ================================================================== #
  optimizer = torch.optim.Adam(model.parameters(), lr=configs.optim_param.lr)
  configs.optim_param.optimizer = optimizer

  # ================================================================== #
  #                      05.Training or Evaluation
  # ================================================================== #
  assert configs.mode in ['train', 'eval']

  if configs.mode == 'train':
    trainer = Trainer(configs)
    trainer.train()

  else:
    evaluators = Evaluator(configs)
    # evaluators.evaluate()
    # if isinstance(test_set, EEGSET):
    #   evaluators.inference()

    if configs.data_param.save_qualitative_result:
      evaluators.qualitative_inference()
    else:
      evaluators.evaluate()
      if isinstance(test_set, EEGSET):
        evaluators.inference()


if __name__ == '__main__':
  main()



