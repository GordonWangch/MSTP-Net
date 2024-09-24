import os, sys
source_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(source_dir)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import argparse
import yaml

from models.proposed_net import MSTP_Model
from models.proposed_net_dilate_1 import MSTP_Model_dilated
from models.proposed_net_all_local import MSTP_Model_all_local
from models.proposed_net_all_global import MSTP_Model_all_global
from models.proposed_net_nonalternation import MSTP_Model_nonalternation
from models.proposed_net_nonalternation_global_first import MSTP_Model_nonalternation_global_first
from torchinfo import summary
from datetime import datetime
from tqdm import tqdm
from matplotlib.colors import PowerNorm
from matplotlib.gridspec import GridSpec



def forward_hook(module, input, output):
  output.retain_grad()  # 使输出保留梯度
  module.saved_output = output  # 将输出保存到模块中


def _over_add(input, gap):
  '''
    Merge sequence
    input: [B, N, K, S]
    gap: padding length
    output: [B, N, L]
  '''
  B, N, K, S = input.shape
  P = K // 2
  # [B, N, S, K]
  input = input.transpose(2, 3).contiguous().view(B, N, -1, K * 2)

  input1 = input[:, :, :, :K].contiguous().view(B, N, -1)[:, :, P:]
  input2 = input[:, :, :, K:].contiguous().view(B, N, -1)[:, :, :-P]
  input = input1 + input2
  # [B, N, L]
  if gap > 0:
    input = input[:, :, :-gap]

  return input 


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
  # ckpt_dir = os.path.join(task_dir, 'checkpoints', f'{filename}-({config_name})-({current_times})_({noise_type})')
  # ckpt_dir = os.path.join(task_dir, 'checkpoints', f'({config_name})-({current_times})_({noise_type})_({model_name})')
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


def get_grad_and_use_ratio(configs, model_name, use_ratio_thres):
  data_shape = tuple(configs.data_param.data_shape)
  device = configs.device

  if model_name == 'proposed':
    model = MSTP_Model(configs.model_param.proposed).to(device)
  elif model_name == 'proposed_dilated_1':
    model = MSTP_Model_dilated(configs.model_param.proposed_dilated_1).to(device)
  elif model_name == 'proposed_all_local':
    model = MSTP_Model_all_local(configs.model_param.proposed_all_local).to(device)
  elif model_name == 'proposed_all_global':
    model = MSTP_Model_all_global(configs.model_param.proposed_all_global).to(device)
  elif model_name == 'proposed_nonalternation':
    model = MSTP_Model_nonalternation(configs.model_param.proposed_nonalternation).to(device)
  elif model_name == 'proposed_nonalternation_global_first':
    model = MSTP_Model_nonalternation_global_first(configs.model_param.proposed_nonalternation_global_first).to(device)
  else:
    raise ValueError(f'[*] Invalid model name: {configs.model_name}')

  model = torch.nn.DataParallel(
    model, device_ids=[int(i) for i in configs.gpu_ids.split(',')])

  data = torch.randn((1, ) + data_shape).to(device)
  data.requires_grad_()

  tcn_length = int(configs.model_param.proposed.separator.stack_num)

  # Create hooks for each layer
  modules = {}
  if 'nonalternation' not in model_name:
    for l in range(tcn_length):
      tcn_local = model.module.separation.dual_tas[l].TCN_local
      tcn_global = model.module.separation.dual_tas[l].TCN_global

      modules[f'Intra_{l + 1}'] = tcn_local
      modules[f'Inter_{l + 1}'] = tcn_global

      tcn_local.register_forward_hook(forward_hook)
      tcn_global.register_forward_hook(forward_hook)
  elif 'nonalternation_global_first' in model_name:
    for l in range(tcn_length):
      tcn_global = model.module.separation.dual_tas_global[l].TCN_global
      modules[f'Inter_{l + 1}'] = tcn_global
      tcn_global.register_forward_hook(forward_hook)

    for l in range(tcn_length):
      tcn_local = model.module.separation.dual_tas_local[l].TCN_local
      modules[f'Intra_{l + 1}'] = tcn_local
      tcn_local.register_forward_hook(forward_hook)
  else:
    for l in range(tcn_length):
      tcn_local = model.module.separation.dual_tas_local[l].TCN_local
      modules[f'Intra_{l + 1}'] = tcn_local
      tcn_local.register_forward_hook(forward_hook)
    
    for l in range(tcn_length):
      tcn_global = model.module.separation.dual_tas_global[l].TCN_global
      modules[f'Inter_{l + 1}'] = tcn_global
      tcn_global.register_forward_hook(forward_hook)

  # mask = model.module.separation
  # mask.register_forward_hook(forward_hook)
  # modules['mask'] = mask

  # Forward pass
  optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)

  grad_dict = {}
  for mod in modules:
    output = model(data)

    mod_output = modules[mod].saved_output
    if 'Intra' in mod:
      mod_output = mod_output.unsqueeze(0)
      mod_output = mod_output.permute(0, 2, 3, 1)
      mod_output = _over_add(mod_output, 6)
    elif 'Inter' in mod:
      mod_output = mod_output.unsqueeze(0)
      mod_output = mod_output.permute(0, 2, 1, 3)
      mod_output = _over_add(mod_output, 6)

    mod_output = torch.sum(mod_output, dim=1).squeeze()
    index = len(mod_output) // 2
    select_mod_output = mod_output[index]

    if data.grad is not None: 
      data.grad.zero_()

    select_mod_output.backward()
    grad_data = np.expand_dims(np.abs(data.grad.squeeze().cpu().numpy()), axis=0)
    # normalize
    grad_data = (grad_data - grad_data.min()) / (grad_data.max() - grad_data.min())
    grad_dict[mod] = grad_data

  use_ratio = [np.sum(grad_dict[mod] > use_ratio_thres) for mod in grad_dict]
  
  return grad_dict, use_ratio


def draw_saliency_map(grad_dict, figsize, title_dict, block_sequence_dict):
  cmap = ['magma', 'viridis'][0]
  # norm = PowerNorm(gamma=0.45) # 0.45
  fig = plt.figure(figsize=figsize)

  gs = GridSpec(4, 2, height_ratios=[1, 1, 1, 0.05])

  axes = [plt.subplot(gs[i, j]) for i in range(3) for j in range(2)]

  for i, (ax, model_name) in enumerate(zip(axes, grad_dict)):
    data = np.vstack(list(grad_dict[model_name].values()))
    cax = sns.heatmap(data, ax=ax, cmap=cmap, cbar=False)

    ax.set_xticks([])
    yticks_positions = [i + 0.5 for i in range(data.shape[0])]
    ax.set_yticks(yticks_positions)  
    ax.set_yticklabels(block_sequence_dict[model_name], rotation=0)
    ax.set_title(f'{title_dict[model_name]}')  


  cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.02])
  cbar = fig.colorbar(cax.get_children()[0], cax=cbar_ax, orientation='horizontal')

  plt.tight_layout()
  # plt.show()

  source_dir = os.path.dirname(os.path.abspath(__file__))
  save_dir = os.path.join(source_dir, 'figures')
  plt.savefig(os.path.join(save_dir, f'saliency_map_all.svg'), bbox_inches='tight')


def draw_receptive_field_estimation(use_ratio_dict, figsize, title_dict):
    marker_list = ['o', 's', 'D', '^', 'h', 'p']
    line_styles = ['-', '--', '-.', ':']         

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
    line_alpha = 0.8
    background_alpha = 0.2
    block_num = 6
    point_num = 512
    x = range(block_num)

    fig = plt.figure(figsize=figsize)
    
    for i, (model_name, color) in enumerate(zip(use_ratio_dict, colors)):
      y = [i / point_num for i in use_ratio_dict[model_name][:block_num]]
      area = np.trapz(y, x) / block_num
      line, = plt.plot(
        x, y, label=f'{title_dict[model_name]} (AUC = {area:.4f})', 
        marker=marker_list[i % len(marker_list)], 
        linestyle=line_styles[i % len(line_styles)], 
        alpha=line_alpha, 
        color=color,
        markersize=8,
        )

    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.40), ncol=2, fontsize=20, frameon=False)
    plt.xticks(ticks=x, labels=[f'Block {i + 1}' for i in x], fontsize=18)
    plt.yticks(fontsize=18)
    plt.ylabel('Receptive Field Estimation $\gamma$', fontsize=20)
    
    plt.grid(True, linestyle='--', alpha=0.2)  
    
    plt.tight_layout()
    plt.show()

    # source_dir = os.path.dirname(os.path.abspath(__file__))
    # save_dir = os.path.join(source_dir, 'figures')
    # plt.savefig(os.path.join(save_dir, 'decision_dependence.svg'), bbox_inches='tight')


def main():
  configs = parse_arguments()
  
  model_ckpts = [
    r'/z3/home/xai_test/wch/paper_related/03-EEG_DENOISE/checkpoints/(cfg-2024-07-09-EEGDenoiseNet-EOG)-(EOG)-(proposed)/checkpoint-EOG-(2024-07-17-15-31-50).pth',
    r'/z3/home/xai_test/wch/paper_related/03-EEG_DENOISE/checkpoints/(cfg-2024-07-09-EEGDenoiseNet-EOG)-(EOG)-(proposed_all_local)/checkpoint-EOG-(2024-07-16-15-11-26).pth',
    r'/z3/home/xai_test/wch/paper_related/03-EEG_DENOISE/checkpoints/(cfg-2024-07-09-EEGDenoiseNet-EOG)-(EOG)-(proposed_all_global)/checkpoint-EOG-(2024-07-16-16-00-46).pth',
    r'/z3/home/xai_test/wch/paper_related/03-EEG_DENOISE/checkpoints/(cfg-2024-07-09-EEGDenoiseNet-EOG)-(EOG)-(proposed_dilated_1)/checkpoint-EOG-(2024-07-17-10-16-11).pth',
    r'/z3/home/xai_test/wch/paper_related/03-EEG_DENOISE/checkpoints/(cfg-2024-07-09-EEGDenoiseNet-EOG)-(EOG)-(proposed_nonalternation)/checkpoint-EOG-(2024-07-17-11-12-37).pth',
    r'/z3/home/xai_test/wch/paper_related/03-EEG_DENOISE/checkpoints/(cfg-2024-07-09-EEGDenoiseNet-EOG)-(EOG)-(proposed_nonalternation_global_first)/checkpoint-EOG-(2024-07-17-12-52-02).pth',
  ]
  
  title_dict_saliency_map = {
    'proposed': 'MSTP-Net (Proposed, Dilation factor: exponential growth)',
    'proposed_all_local': 'Variant Model A (Dilation factor: exponential growth)',
    'proposed_all_global': 'Variant Model B (Dilation factor: exponential growth)',
    'proposed_dilated_1': 'Variant Model C (Dilation factor: 1)',
    'proposed_nonalternation': 'Variant Model D (Dilation factor: exponential growth)',
    'proposed_nonalternation_global_first': 'Variant Model E (Dilation factor: exponential growth)',
  }
  
  title_dict_decision_dependence = {
    'proposed': 'MSTP-Net (Proposed)',
    'proposed_all_local': 'Variant Model A',
    'proposed_all_global': 'Variant Model B',
    'proposed_dilated_1': 'Variant Model C',
    'proposed_nonalternation': 'Variant Model D',
    'proposed_nonalternation_global_first': 'Variant Model E',
  }
  
  block_sequence_dict = {
    'proposed': ['Intra', 'Inter', 'Intra', 'Inter', 'Intra', 'Inter'],
    'proposed_all_local': ['Intra', 'Intra', 'Intra', 'Intra', 'Intra', 'Intra'],
    'proposed_all_global': ['Inter', 'Inter', 'Inter', 'Inter', 'Inter', 'Inter'],
    'proposed_dilated_1': ['Intra', 'Inter', 'Intra', 'Inter', 'Intra', 'Inter'],
    'proposed_nonalternation': ['Intra', 'Intra', 'Intra', 'Inter', 'Inter', 'Inter'], 
    'proposed_nonalternation_global_first': ['Inter', 'Inter', 'Inter', 'Intra', 'Intra', 'Intra']
  }
  
  configs.model = 'eval'

  figsize = (14, 8)
  use_ratio_thres = 0.01
  grad_dict = {}
  use_ratio_dict = {}

  for ckpt in tqdm(model_ckpts):
    model_name = ckpt.split('/')[-2].split('(')[-1].split(')')[0]
    configs.evaluation.model_path = ckpt
    configs.model_name = model_name

    grad_dict[model_name], use_ratio_dict[model_name] = get_grad_and_use_ratio(
      configs, model_name, use_ratio_thres)
    
  # draw_saliency_map(grad_dict, figsize, title_dict_saliency_map, block_sequence_dict)
  draw_receptive_field_estimation(use_ratio_dict, figsize, title_dict_decision_dependence)



if __name__ == '__main__':
  main()