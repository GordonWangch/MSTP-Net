import numpy as np
import torch
import eeg_metric
import os
import torch.nn as nn
import matplotlib.pyplot as plt
import random

from tqdm import tqdm
from datetime import datetime



class Trainer:
  def __init__(self, configs):
    self.configs = configs

    self.n_epochs = self.configs.training.n_epochs
    self.loader = self.configs.loader
    self.device = self.configs.device

    self.model = self.configs.model

    self.optimizer = self.configs.optim_param.optimizer
    self.loss_name = self.configs.optim_param.loss.name

    self.early_stopping = EarlyStopping(
      patience=self.configs.training.patience, 
      ckpt_dir=self.configs.ckpt_dir,
      lower_is_better=True,
      ckpt_name = self.configs.ckpt_name,
      )


  def update_lr(self, new_lr):
    for param_group in self.optimizer.param_groups:
      param_group['lr'] = new_lr


  def loss_func(self, gt, pred):
    loss_dict = {}

    if self.loss_name == 'mse':
      loss_dict['MSE'] = nn.MSELoss()(gt, pred)
    elif self.loss_name == 'lncosh':
      lncosh_loss = eeg_metric.lncosh_loss(gt, pred, self.configs.optim_param.loss.lncosh.lamda)
      loss_dict['Lncosh'] = lncosh_loss
    elif self.loss_name == 'mse, lncosh':
      mse_loss = nn.MSELoss()(gt, pred)
      lncosh_loss = eeg_metric.lncosh_loss(gt, pred, self.configs.optim_param.loss.lncosh.lamda)
      
      mse_loss = mse_loss * self.configs.optim_param.loss.mse.weight
      lncosh_loss = lncosh_loss * self.configs.optim_param.loss.lncosh.weight

      loss_dict['MSE'] = mse_loss
      loss_dict['Lncosh'] = lncosh_loss
    else:
      raise NotImplementedError(f'[ERROR] Loss function {self.loss_name} is not implemented.')
    
    return loss_dict


  @torch.no_grad()
  def valid(self):
    self.model.eval()
    mode = 'valid'

    total_num = 0
    loss_all = 0
    mse_all = 0
    lncosh_all = 0
    rrmse_temporal_all = 0
    rrmse_spectrum_all = 0
    cc_all = 0
    snr_all = 0

    loader = self.loader.val_loader if mode == 'valid' else self.loader.test_loader
    step_bar = Stepbar(mode, loader)

    for step, (feature, target) in step_bar:
      feature = feature.to(self.device)
      target = target.to(self.device)

      output = self.model(feature)
      loss_dict = self.loss_func(output, target)
      loss = sum(loss_dict.values())

      gt = target.cpu().data.numpy()
      pred = output.cpu().data.numpy()
      batch_num = pred.shape[0]

      # Calculate Metrics
      rrmse_temporal = eeg_metric.get_metrics('RRMSE_temporal', gt, pred)
      rrmse_spectrum = eeg_metric.get_metrics('RRMSE_spectrum', gt, pred)
      cc = eeg_metric.get_metrics('CC', gt, pred)
      snr = eeg_metric.get_metrics('SNR', gt, pred)

      rrmse_temporal = np.mean(rrmse_temporal)
      rrmse_spectrum = np.mean(rrmse_spectrum)
      cc = np.mean(cc)
      snr = np.mean(snr)

      total_num = total_num + batch_num
      loss_all = loss_all + loss.item() * batch_num
      mse_all = mse_all + loss_dict['MSE'].item() * batch_num if 'MSE' in loss_dict.keys() else 0
      lncosh_all = lncosh_all + loss_dict['Lncosh'].item() * batch_num if 'Lncosh' in loss_dict.keys() else 0
      rrmse_temporal_all = rrmse_temporal_all + rrmse_temporal * batch_num 
      rrmse_spectrum_all = rrmse_spectrum_all + rrmse_spectrum * batch_num
      cc_all = cc_all + cc * batch_num
      snr_all = snr_all + snr * batch_num

      step_bar.set_postfix(
        {'Loss': f'{loss_all / total_num: .4f}', 
        'MSE': f'{mse_all / total_num: .4f}',
        'Lncosh': f'{lncosh_all / total_num: .4f}',
        'RRMSE_temporal': f'{rrmse_temporal_all / total_num: .4f}', 
        'RRMSE_spectrum': f'{rrmse_spectrum_all / total_num: .4f}', 
        'CC': f'{cc_all / total_num: .4f}',
        'SNR': f'{snr_all / total_num: .4f}'
        })
      
    metric_dict = {
      'MSE': loss_all / total_num,
      'RRMSE_temporal': rrmse_temporal_all / total_num,
      'RRMSE_spectrum': rrmse_spectrum_all / total_num,
      'CC': cc_all / total_num,
      'SNR': snr_all / total_num
    }

    # metric_visual = [f'{k}:{v: .4f}' for k, v in metric_dict.items()]
    # print(f'[INFO] [Valid] {" | ".join(metric_visual)}')
    if mode == 'valid':
      self.early_stopping(metric_dict, self.model)
    else:
      print("\n".join([f'[INFO] [{mode.capitalize()}] {k} = {v: .4f}' for k, v in metric_dict.items()]))

    self.model.train()


  def train(self):
    current_lr = self.configs.optim_param.lr

    for epoch in range(self.n_epochs):
      total_num = 0
      loss_all = 0
      mse_all = 0
      lncosh_all = 0
      rrmse_temporal_all = 0
      rrmse_spectrum_all = 0
      cc_all = 0
      snr_all = 0

      times = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
      step_bar = tqdm(
        enumerate(self.loader.train_loader),
        desc=f'[{times}] [Train] Epoch {epoch + 1}',
        total=len(self.loader.train_loader),
        dynamic_ncols=True,
        ascii='->=',
        )

      for step, (feature, target) in step_bar:
        # if self.configs.data_param.noise_type == 'EMG':
        #   # random_size = random.choice([256, 512, 1024])
        #   random_size = np.random.randint(128, 1024 + 1)
        # else:
        #   random_size = random.choice([64, 256, 512])
        #   # random_size = np.random.randint(64, 512 + 1)

        # start_index = np.random.randint(0, feature.size()[-1] - random_size + 1)

        # feature = feature[:, :, start_index:start_index + random_size]
        # target = target[:, :, start_index:start_index + random_size]

        feature = feature.to(self.device)
        target = target.to(self.device)

        # Forward
        output = self.model(feature)
        loss_dict = self.loss_func(output, target)
        loss = sum(loss_dict.values())
        
        # Backward and optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Information Summary
        gt = target.cpu().data.numpy()
        pred = output.cpu().data.numpy()
        batch_num = pred.shape[0]

        # Calculate Metrics
        rrmse_temporal = np.mean(eeg_metric.get_metrics('RRMSE_temporal', gt, pred))
        rrmse_spectrum = np.mean(eeg_metric.get_metrics('RRMSE_spectrum', gt, pred))
        cc = np.mean(eeg_metric.get_metrics('CC', gt, pred))
        snr = np.mean(eeg_metric.get_metrics('SNR', gt, pred))

        total_num = total_num + batch_num
        loss_all = loss_all + loss.item() * batch_num
        mse_all = mse_all + loss_dict['MSE'].item() * batch_num if 'MSE' in loss_dict.keys() else 0
        lncosh_all = lncosh_all + loss_dict['Lncosh'].item() * batch_num if 'Lncosh' in loss_dict.keys() else 0
        rrmse_temporal_all = rrmse_temporal_all + rrmse_temporal * batch_num 
        rrmse_spectrum_all = rrmse_spectrum_all + rrmse_spectrum * batch_num
        cc_all = cc_all + cc * batch_num
        snr_all = snr_all + snr * batch_num
        
        step_bar.set_postfix(
          {'Loss': f'{loss_all / total_num: .4f}', 
          'MSE': f'{mse_all / total_num: .4f}',
          'Lncosh': f'{lncosh_all / total_num: .4f}',
          'RRMSE_temporal': f'{rrmse_temporal_all / total_num: .4f}', 
          'RRMSE_spectrum': f'{rrmse_spectrum_all / total_num: .4f}', 
          'CC': f'{cc_all / total_num: .4f}',
          'SNR': f'{snr_all / total_num: .4f}'
          })
        
        # Decay learning rate
        # if (epoch + 1) % 5 == 0:
        #   current_lr /= 3
        #   self.update_lr(current_lr)

      # Validation
      if (epoch + 1) % self.configs.training.validation_freq_in_epoch == 0:
        self.valid()

      
      if self.early_stopping.early_stop:
        print("[INFO] Early stopping")
        ckpt_path = os.path.join(self.configs.ckpt_dir, self.configs.ckpt_name)
        print('[INFO] Model finally saved to', ckpt_path)
        break



class Evaluator:
  def __init__(self, configs):
    self.configs = configs

    self.device = self.configs.device
    self.model_path = self.configs.evaluation.model_path
    self.loader = self.configs.loader.test_loader

    self.loss_name = self.configs.optim_param.loss.name

    self.model = self.configs.model
    self.model.load_state_dict(torch.load(self.model_path))


  def loss_func(self, gt, pred):
    loss_dict = {}

    if self.loss_name == 'mse':
      loss_dict['MSE'] = nn.MSELoss()(gt, pred)
    elif self.loss_name == 'lncosh':
      lncosh_loss = eeg_metric.lncosh_loss(gt, pred, self.configs.optim_param.loss.lncosh.lamda)
      loss_dict['Lncosh'] = lncosh_loss
    elif self.loss_name == 'mse, lncosh':
      mse_loss = nn.MSELoss()(gt, pred)
      lncosh_loss = eeg_metric.lncosh_loss(gt, pred, self.configs.optim_param.loss.lncosh.lamda)
      
      mse_loss = mse_loss * self.configs.optim_param.loss.mse.weight
      lncosh_loss = lncosh_loss * self.configs.optim_param.loss.lncosh.weight

      loss_dict['MSE'] = mse_loss
      loss_dict['Lncosh'] = lncosh_loss
    else:
      raise NotImplementedError(f'[ERROR] Loss function {self.loss_name} is not implemented.')
    
    return loss_dict


  @torch.no_grad()
  def evaluate(self):
    self.model.eval()

    rrmse_temporal_list = []
    rrmse_spectrum_list = []
    cc_list = []
    loss_all = 0
    mse_all = 0
    lncosh_all = 0
    clean_list = []
    pred_list = []
    snr_db_list = []
    snr_list = []
    contaminate_list = []

    times = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    step_bar = tqdm(
      enumerate(self.loader),
      desc=f'[{times}] [Eval]',
      total=len(self.loader),
      dynamic_ncols=True,
      leave=True,
      ascii='->='
      )

    for step, (feature, target, snr_db) in step_bar:
      feature = feature.to(self.device)
      target = target.to(self.device)

      output = self.model(feature)
      loss_dict = self.loss_func(output, target)
      loss = sum(loss_dict.values())

      gt = target.cpu().data.numpy()
      pred = output.cpu().data.numpy()
      batch_num = pred.shape[0]

      # 
      clean_list.append(gt)
      pred_list.append(pred)
      contaminate_list.append(feature.cpu().data.numpy())
      snr_db_list.append(snr_db.data.numpy())     

      rrmse_temporal = eeg_metric.get_metrics('RRMSE_temporal', gt, pred)
      rrmse_spectrum = eeg_metric.get_metrics('RRMSE_spectrum', gt, pred)
      cc = eeg_metric.get_metrics('CC', gt, pred)
      snr = eeg_metric.get_metrics('SNR', gt, pred)

      rrmse_temporal_list = rrmse_temporal_list + rrmse_temporal.tolist()
      rrmse_spectrum_list = rrmse_spectrum_list + rrmse_spectrum.tolist()
      cc_list = cc_list + cc.tolist()
      snr_list = snr_list + snr.tolist()
      loss_all = loss_all + loss.item() * batch_num 
      mse_all = mse_all + loss_dict['MSE'].item() * batch_num if 'MSE' in loss_dict.keys() else 0
      lncosh_all = lncosh_all + loss_dict['Lncosh'].item() * batch_num if 'Lncosh' in loss_dict.keys() else 0

      step_bar.set_postfix(
        {'Loss': f'{loss_all / len(cc_list):.4f}', 
        'MSE': f'{mse_all / len(cc_list): .4f}',
        'Lncosh': f'{lncosh_all / len(cc_list): .4f}',
        'RRMSE_temporal': f'{np.mean(rrmse_temporal_list): .4f}', 
        'RRMSE_spectrum': f'{np.mean(rrmse_spectrum_list): .4f}', 
        'CC': f'{np.mean(cc_list): .4f}',
        'SNR': f'{np.mean(snr_list): .4f}'
         })
      
    metric_dict = {
      'Loss': loss_all / len(self.loader.dataset),
      'MSE': mse_all / len(self.loader.dataset),
      'Lncosh': lncosh_all / len(self.loader.dataset),
      'RRMSE_temporal': np.mean(rrmse_temporal_list),
      'RRMSE_spectrum': np.mean(rrmse_spectrum_list),
      'CC': np.mean(cc_list),
      'SNR': np.mean(snr_list)
    }

    print("\n".join([f'[INFO] [Eval] {k} = {v: .4f}' for k, v in metric_dict.items()]))

    
    clean_list = np.concatenate(clean_list, axis=0)
    pred_list = np.concatenate(pred_list, axis=0)
    contaminate_list = np.concatenate(contaminate_list, axis=0)
    snr_db_list = np.concatenate(snr_db_list, axis=0)

    if 'Semi' in self.configs.data_param.noise_type:
      eval_result_save_path = os.path.join(self.configs.task_dir, 'eval_result', f'{self.configs.data_param.noise_type} {self.configs.model_name}.npy')
      if self.configs.data_param.save_eval_result: np.save(eval_result_save_path, metric_dict)

  
  @torch.no_grad()
  def inference(self):
    self.model.eval()

    testset = self.loader.dataset
    batch_size = self.configs.training.batch_size 
    metric_dict = {}
    snr_db_list = list(range(testset.snr_db[0], testset.snr_db[1] + 1))

    for snr_db in snr_db_list:
      rrmse_temporal_list = []
      rrmse_spectrum_list = []
      cc_list = []
      loss_all = 0
      mse_all = 0
      lncosh_all = 0
      clean_list = []
      pred_list = []
      snr_list = []
      contaminate_list = [] 

      loader = testset._get_data_with_fixed_snr(snr_db, batch_size)

      times = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
      step_bar = tqdm(
        enumerate(loader),
        desc=f'[{times}] [Eval] [SNR:{snr_db: >4}dB]',
        total=len(loader),
        dynamic_ncols=True,
        leave=True,
        ascii='->='
        )

      for step, (feature, target) in step_bar:
        feature = torch.from_numpy(feature).to(self.device)
        target = torch.from_numpy(target).to(self.device)

        output = self.model(feature)
        loss_dict = self.loss_func(output, target)
        loss = sum(loss_dict.values())

        gt = target.cpu().data.numpy()
        pred = output.cpu().data.numpy()
        batch_num = pred.shape[0]

        clean_list.append(gt)
        pred_list.append(pred)
        contaminate_list.append(feature.cpu().data.numpy())

        rrmse_temporal = eeg_metric.get_metrics('RRMSE_temporal', gt, pred)
        rrmse_spectrum = eeg_metric.get_metrics('RRMSE_spectrum', gt, pred)
        cc = eeg_metric.get_metrics('CC', gt, pred)
        snr = eeg_metric.get_metrics('SNR', gt, pred)

        rrmse_temporal_list = rrmse_temporal_list + rrmse_temporal.tolist()
        rrmse_spectrum_list = rrmse_spectrum_list + rrmse_spectrum.tolist()
        cc_list = cc_list + cc.tolist()
        snr_list = snr_list + snr.tolist()
        loss_all = loss_all + loss.item() * batch_num 
        mse_all = mse_all + loss_dict['MSE'].item() * batch_num if 'MSE' in loss_dict.keys() else 0
        lncosh_all = lncosh_all + loss_dict['Lncosh'].item() * batch_num if 'Lncosh' in loss_dict.keys() else 0

        step_bar.set_postfix(
          {'Loss': f'{loss_all / len(cc_list):.4f}', 
          'MSE': f'{mse_all / len(cc_list): .4f}',
          'Lncosh': f'{lncosh_all / len(cc_list): .4f}',
          'RRMSE_temporal': f'{np.mean(rrmse_temporal_list): .4f}', 
          'RRMSE_spectrum': f'{np.mean(rrmse_spectrum_list): .4f}', 
          'CC': f'{np.mean(cc_list): .4f}',
          'SNR': f'{np.mean(snr_list): .4f}'
          })
        
      metric_dict[snr_db] = {
        'Loss': loss_all / len(self.loader.dataset),
        'MSE': mse_all / len(self.loader.dataset),
        'Lncosh': lncosh_all / len(self.loader.dataset),
        'RRMSE_temporal': np.mean(rrmse_temporal_list),
        'RRMSE_spectrum': np.mean(rrmse_spectrum_list),
        'CC': np.mean(cc_list),
        'SNR': np.mean(snr_list)
      }

      print("\n".join([f'[INFO] [Eval] [SNR:{snr_db: >3}dB] {k} = {v: .4f}' for k, v in metric_dict[snr_db].items()]))

    metric_save_path = os.path.join(self.configs.task_dir, 'metric_result', f'{self.configs.data_param.noise_type} ours.npy')
    ablation_result_save_path = os.path.join(self.configs.task_dir, 'ablation_result', f'{self.configs.data_param.noise_type} {self.configs.model_name}.npy')
    eval_result_save_path = os.path.join(self.configs.task_dir, 'eval_result', f'{self.configs.data_param.noise_type} {self.configs.model_name}.npy')

    if self.configs.data_param.save_metric: np.save(metric_save_path, metric_dict)
    if self.configs.data_param.save_ablation_result: np.save(ablation_result_save_path, metric_dict)
    if self.configs.data_param.save_eval_result: np.save(eval_result_save_path, metric_dict)

    draw_snr_db(metric_dict, snr_db_list)
    

  @torch.no_grad()
  def qualitative_inference(self):
    self.model.eval()
    test_data_path = os.path.join(self.configs.data_dir, '04-EEG', f'test_data_{self.configs.data_param.noise_type}.npy')

    test_data = np.load(test_data_path, allow_pickle=True).item()

    contaminate = test_data['contaminated']
    clean = test_data['clean']

    contaminate = np.expand_dims(np.expand_dims(contaminate, axis=0), axis=0).astype(np.float32)
    clean = np.expand_dims(np.expand_dims(clean, axis=0), axis=0)

    feature = torch.from_numpy(contaminate).to(self.device)
    pred = self.model(feature).cpu().data.numpy()

    save_path = os.path.join(
      self.configs.task_dir, 'qualitative_result', 
      f'{self.configs.data_param.noise_type} {self.configs.model_name}.npy')
    
    results = {
      'contaminate': contaminate,
      'clean': clean,
      'pred': pred,
    }

    np.save(save_path, results)



class EarlyStopping:
  def __init__(
    self, 
    main_metric='MSE',
    patience=20, 
    ckpt_dir=None, 
    lower_is_better=True,
    ckpt_name='checkpoint',
    ):

    self.patience = patience
    self.counter = 0

    self.main_metric = main_metric
    self.best_score = None
    self.best_metrics = None
    self.best_model = None

    self.lower_is_better = lower_is_better
    self.early_stop = False

    self.compare = np.less if lower_is_better else np.greater

    self.ckpt_dir = ckpt_dir
    self.ckpt_name = ckpt_name if '.pth' in ckpt_name else ckpt_name + '.pth'


  def __call__(self, metric_dict, model):
    assert self.main_metric in metric_dict.keys()
    main_metric_value = metric_dict[self.main_metric]

    if self.best_score is None or self.compare(main_metric_value, list(self.best_score.values())[0]):
      print(f'[INFO] [Valid] {self.main_metric} = {main_metric_value: .4f} <New Records>')
      print("\n".join([f'[INFO] [Valid] {k} = {v: .4f}' for k, v in metric_dict.items() if k != self.main_metric]))

      self.best_score = {self.main_metric: main_metric_value}
      self.best_metrics = metric_dict

      self.counter = 0

      self.save_checkpoint(model)
    else:
      self.counter += 1
      if self.counter >= self.patience:
        self.early_stop = True
        
      print(f'[INFO] [Valid] {self.main_metric} = {main_metric_value: .4f} (Best: {list(self.best_score.values())[0]: .4f}, Patience: {self.counter}/{self.patience})')
      print("\n".join([f'[INFO] [Valid] {k} = {v: .4f}' for k, v in self.best_metrics.items() if k != self.main_metric]))
  

  def save_checkpoint(self, model):
    if not os.path.exists(self.ckpt_dir): 
      os.makedirs(self.ckpt_dir)

    ckpt_path = os.path.join(self.ckpt_dir, self.ckpt_name)
    torch.save(model.state_dict(), ckpt_path)
    print('[INFO] Model have saved to', ckpt_path)


def Stepbar(mode, loader, ascii='->='):
  times = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
  step_bar = tqdm(
    enumerate(loader),
    desc=f'[{times}] [{mode.capitalize()}]',
    total=len(loader),
    dynamic_ncols=True,
    leave=True,
    ascii=ascii,
    )

  return step_bar 


def draw_snr_db(metric_dict, snr_db_list, figure_size=(16, 4)):
  rrmse_temporal_list = [metric_dict[snr_db]['RRMSE_temporal'] for snr_db in snr_db_list]
  rrmse_spectrum_list = [metric_dict[snr_db]['RRMSE_spectrum'] for snr_db in snr_db_list]
  cc_list = [-metric_dict[snr_db]['CC'] for snr_db in snr_db_list]
  snr_list = [metric_dict[snr_db]['SNR'] for snr_db in snr_db_list]

  fig, ax = plt.subplots(1, 4, figsize=figure_size)

  ax[0].plot(snr_db_list, rrmse_temporal_list, label='RRMSE Temporal')
  ax[1].plot(snr_db_list, rrmse_spectrum_list, label='RRMSE Spectrum')
  ax[2].plot(snr_db_list, cc_list, label='CC')
  ax[3].plot(snr_db_list, snr_list, label='SNR')

  ax[0].set_title('RRMSE Temporal')
  ax[1].set_title('RRMSE Spectrum')
  ax[2].set_title('CC')
  ax[3].set_title('SNR')

  plt.show()