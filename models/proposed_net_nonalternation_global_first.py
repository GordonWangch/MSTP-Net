import sys
import os
source_dir = os.path.abspath('./')
sys.path.append(source_dir)
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np

from torchinfo import summary
from torch.autograd import Variable


class GlobalLayerNorm(nn.Module):
  def __init__(self, dim, shape, eps=1e-8, elementwise_affine=True):
    super(GlobalLayerNorm, self).__init__()
    self.dim = dim
    self.eps = eps
    self.elementwise_affine = elementwise_affine

    if self.elementwise_affine:
      if shape == 3:
        self.weight = nn.Parameter(torch.ones(self.dim, 1))
        self.bias = nn.Parameter(torch.zeros(self.dim, 1))
      if shape == 4:
        self.weight = nn.Parameter(torch.ones(self.dim, 1, 1))
        self.bias = nn.Parameter(torch.zeros(self.dim, 1, 1))
    else:
      self.register_parameter('weight', None)
      self.register_parameter('bias', None)

  def forward(self, x):
    # x = N x C x K x S or N x C x L
    # N x 1 x 1
    # cln: mean,var N x 1 x K x S
    # gln: mean,var N x 1 x 1
    if x.dim() == 4:
      mean = torch.mean(x, (1, 2, 3), keepdim=True)
      var = torch.mean((x-mean)**2, (1, 2, 3), keepdim=True)
      if self.elementwise_affine:
        x = self.weight*(x-mean)/torch.sqrt(var+self.eps)+self.bias
      else:
        x = (x-mean)/torch.sqrt(var+self.eps)

    if x.dim() == 3:
      mean = torch.mean(x, (1, 2), keepdim=True)
      var = torch.mean((x-mean)**2, (1, 2), keepdim=True)
      if self.elementwise_affine:
        x = self.weight*(x-mean)/torch.sqrt(var+self.eps)+self.bias
      else:
        x = (x-mean)/torch.sqrt(var+self.eps)

    return x


class CumulativeLayerNorm(nn.LayerNorm):
  '''
    Calculate Cumulative Layer Normalization
    dim: you want to norm dim
    elementwise_affine: learnable per-element affine parameters 
  '''

  def __init__(self, dim, elementwise_affine=True):
    super(CumulativeLayerNorm, self).__init__(
      dim, elementwise_affine=elementwise_affine, eps=1e-8)


  def forward(self, x):
    # x: N x C x K x S or N x C x L
    # N x K x S x C
    if x.dim() == 4:
      x = x.permute(0, 2, 3, 1).contiguous()
      # N x K x S x C == only channel norm
      x = super().forward(x)
      # N x C x K x S
      x = x.permute(0, 3, 1, 2).contiguous()
    if x.dim() == 3:
      x = torch.transpose(x, 1, 2)
      # N x L x C == only channel norm
      x = super().forward(x)
      # N x C x L
      x = torch.transpose(x, 1, 2)
    return x


class cLN(nn.Module):
	def __init__(self, dimension, eps=1e-8, trainable=True):
		super(cLN, self).__init__()

		self.eps = eps
		if trainable:
			self.gain = nn.Parameter(torch.ones(1, dimension, 1))
			self.bias = nn.Parameter(torch.zeros(1, dimension, 1))
		else:
			self.gain = Variable(torch.ones(1, dimension, 1), requires_grad=False)
			self.bias = Variable(torch.zeros(1, dimension, 1), requires_grad=False)

	def forward(self, input):
		# input size: (Batch, Freq, Time)
		# cumulative mean for each time step

		batch_size = input.size(0)
		channel = input.size(1)
		time_step = input.size(2)

		step_sum = input.sum(1)  # B, T
		step_pow_sum = input.pow(2).sum(1)  # B, T
		cum_sum = torch.cumsum(step_sum, dim=1)  # B, T
		cum_pow_sum = torch.cumsum(step_pow_sum, dim=1)  # B, T

		entry_cnt = np.arange(channel, channel * (time_step + 1), channel)
		entry_cnt = torch.from_numpy(entry_cnt).type(input.type())
		entry_cnt = entry_cnt.view(1, -1).expand_as(cum_sum)

		cum_mean = cum_sum / entry_cnt  # B, T
		cum_var = (cum_pow_sum - 2 * cum_mean * cum_sum) / entry_cnt + cum_mean.pow(2)  # B, T
		cum_std = (cum_var + self.eps).sqrt()  # B, T

		cum_mean = cum_mean.unsqueeze(1)
		cum_std = cum_std.unsqueeze(1)

		x = (input - cum_mean.expand_as(input)) / cum_std.expand_as(input)
		return x * self.gain.expand_as(x).type(x.type()) + self.bias.expand_as(x).type(x.type())


class DepthConv1d(nn.Module):

	def __init__(self, input_channel, hidden_channel, kernel, padding, dilation=1, skip=True, causal=False):
		super(DepthConv1d, self).__init__()

		self.causal = causal
		self.skip = skip

		self.conv1d = nn.Conv1d(input_channel, hidden_channel, 1)
		if self.causal:
			self.padding = (kernel - 1) * dilation
		else:
			self.padding = padding
		self.dconv1d = nn.Conv1d(
      hidden_channel, hidden_channel, kernel, dilation=dilation, 
      groups=hidden_channel, padding=self.padding)
		self.res_out = nn.Conv1d(hidden_channel, input_channel, 1)
		self.nonlinearity1 = nn.PReLU()
		self.nonlinearity2 = nn.PReLU()
		if self.causal:
			self.reg1 = cLN(hidden_channel, eps=1e-08)
			self.reg2 = cLN(hidden_channel, eps=1e-08)
		else:
			self.reg1 = nn.GroupNorm(1, hidden_channel, eps=1e-08)
			self.reg2 = nn.GroupNorm(1, hidden_channel, eps=1e-08)

		if self.skip:
			self.skip_out = nn.Conv1d(hidden_channel, input_channel, 1)

	def forward(self, input):
		output = self.conv1d(input)
		# # Add at 2024-06-25
		# output = nn.Dropout(0.2)(output)
		# # Add at 2024-06-25
		output = self.reg1(self.nonlinearity1(output))
		if self.causal:
			output = self.reg2(self.nonlinearity2(self.dconv1d(output)[:, :, :-self.padding]))
		else:
			output = self.dconv1d(output)
			# # Add at 2024-06-25
			# output = nn.Dropout(0.2)(output)
			# # Add at 2024-06-25
			output = self.reg2(self.nonlinearity2(output))

		residual = self.res_out(output)

		if self.skip:
			skip = self.skip_out(output)
			return residual, skip
		else:
			return residual


class TCN(nn.Module):
  def __init__(
    self, 
    input_dim,
		output_dim,
		BN_dim,
		hidden_dim,
		layer_num,
		kernel_size=3,
		skip=True,
		causal=False,
		dilated=True
    ):
    super(TCN, self).__init__()
    # normalization
    if not causal:
      self.LN = nn.GroupNorm(1, input_dim, eps=1e-8)
    else:
      self.LN = cLN(input_dim, eps=1e-8)
    
    self.BN = nn.Conv1d(input_dim, BN_dim, 1)

    # TCN for feature extraction 
    self.receptive_field = 0
    self.dilated = dilated 
    
    self.TCN = nn.ModuleList([])
    for i in range(layer_num): 
      if self.dilated: 
        self.TCN.append(
            DepthConv1d(BN_dim, hidden_dim, kernel_size, dilation=2 ** i, padding=2 ** i, skip=skip, causal=causal))
      else:
        self.TCN.append(DepthConv1d(BN_dim, hidden_dim, kernel_size, dilation=1, padding=1, skip=skip, causal=causal))
      
    self.output = nn.Sequential(nn.PReLU(), nn.Conv1d(BN_dim, output_dim, 1))
    self.skip = skip

  def forward(self, input):
    # input shape: (B, N, L)
    # normalization
    output = self.LN(input)
    output = self.BN(output)
    
    # pass to TCN
    if self.skip:
      skip_connection = 0.
      for i in range(len(self.TCN)):
        residual, skip = self.TCN[i](output)
        output = output + residual
        skip_connection = skip_connection + skip
    else:
      for i in range(len(self.TCN)):
        residual = self.TCN[i](output)
        output = output + residual

    # output layer
    if self.skip:
      output = self.output(skip_connection)
    else:
      output = self.output(output)
    
    return output


def select_norm(norm, dim, shape):
  if norm == 'gln':
    return GlobalLayerNorm(dim, shape, elementwise_affine=True)
  if norm == 'cln':
    return CumulativeLayerNorm(dim, elementwise_affine=True)
  if norm == 'ln':
    return nn.GroupNorm(1, dim, eps=1e-8)
  else:
    return nn.BatchNorm1d(dim)


class Encoder(nn.Module):
  '''
    kernel_size: the length of filters
    out_channels: the number of filters
    stride: the length of stride
  '''

  def __init__(self, kernel_size=2, out_channels=64, stride=1):
    super(Encoder, self).__init__()
    self.conv1d = nn.Conv1d(
      in_channels=1, out_channels=out_channels, kernel_size=kernel_size, 
      stride=stride, groups=1, bias=False)


  def forward(self, x):
    """
    Input:
      x: [B, 1, T], B is batch size, T is times
    Returns:
      x: [B, C, T_out]
      T_out is the number of time steps
    """
    x = self.conv1d(x)    # [B, 1, T] -> [B, C, T_out]
    x = F.relu(x)

    return x


class Decoder(nn.ConvTranspose1d):
  def __init__(self, *args, **kwargs):
    super(Decoder, self).__init__(*args, **kwargs)

  def forward(self, x): 
    """
      x: [B, N, L]
    """
    if x.dim() not in [2, 3]:
      raise RuntimeError("{} accept 3/4D tensor as input".format(self.__name__))

    x = super().forward(x if x.dim() == 3 else torch.unsqueeze(x, 1))

    if torch.squeeze(x).dim() == 1:
      x = torch.squeeze(x, dim=1)
    else:
      x = torch.squeeze(x)

    return x


class Intra_Block(nn.Module):
  def __init__(self, configs):
    super(Intra_Block, self).__init__()

    self.configs = configs

    self.layer_num_local = self.configs.separator.layer_num_local

    self.norm = self.configs.separator.norm

    self.out_channels = self.configs.separator.feature_dim
    self.hidden_channels = self.configs.separator.hidden_dim

    self.kernel_size = self.configs.separator.kernel_size

    ## TCN
    self.TCN_local = TCN(
      input_dim=self.out_channels,
      output_dim=self.out_channels,
      BN_dim=self.hidden_channels,
      hidden_dim=self.hidden_channels * 4,
      layer_num=self.layer_num_local,
      kernel_size=self.kernel_size,
      skip=False,
    ) 

    # Norm
    self.intra_norm = select_norm(self.norm, self.out_channels, 4)
      

  def forward(self, x):
    '''
      x: [B, F, K, S] -> [bs, 64, 100, 22]
      out: [B, F, K, S] -> [bs, 64, 100, 22]
    '''
    B, F, K, S = x.shape
    # intra for local 
    # [B, F, K, S] -> [B, S, F, K] -> [B * S, F, K]
    intra = x.permute(0, 3, 1, 2).contiguous().view(B*S, F, K)      ## [BS, F, K] -> [bs * 22, 64, 100]
    intra = self.TCN_local(intra)                                   ## [BS, F, K] -> [bs * 22, 64, 100]
    intra = intra.view(B, S, F, K)                                  ## [B, S, F, K] -> [bs, 22, 64, 100]
    intra = intra.permute(0, 2, 3, 1).contiguous()                  ## [B, F, K, S] -> [bs, 64, 100, 22]
    intra = self.intra_norm(intra)                                  ## [B, F, K, S] -> [bs, 64, 100, 22]

    # Add 
    intra = intra + x                                               ## [B, F, K, S] -> [bs, 64, 100, 22]
    return intra


class Inter_Block(nn.Module):
  def __init__(self, configs):
    super(Inter_Block, self).__init__()

    self.configs = configs

    self.layer_num_global = self.configs.separator.layer_num_global

    self.norm = self.configs.separator.norm

    self.out_channels = self.configs.separator.feature_dim
    self.hidden_channels = self.configs.separator.hidden_dim

    self.kernel_size = self.configs.separator.kernel_size

    ## TCN
    self.TCN_global= TCN(
      input_dim=self.out_channels,
      output_dim=self.out_channels,
      BN_dim=self.hidden_channels,
      hidden_dim=self.hidden_channels * 4,
      layer_num=self.layer_num_global,
      kernel_size=self.kernel_size,
      skip=False,
    )
    
    # Norm
    self.inter_norm = select_norm(self.norm, self.out_channels, 4)
      

  def forward(self, x):
    '''
      x: [B, F, K, S] -> [bs, 64, 100, 22]
      out: [B, F, K, S] -> [bs, 64, 100, 22]
    '''
    B, F, K, S = x.shape

    # inter for global
    # [B, F, K, S] -> [B, K, F, S] -> [B * K, F, S]
    inter = x.permute(0, 2, 1, 3).contiguous().view(B*K, F, S)  ## [BK, F, S] -> [bs * 100, 64, 22]
    inter = self.TCN_global(inter)                                  ## [BK, F, S] -> [bs * 100, 64, 22]
    inter = inter.view(B, K, F, S)                                  ## [B, K, F, S] -> [bs, 100, 64, 22]
    inter = inter.permute(0, 2, 1, 3).contiguous()                  ## [B, F, K, S] -> [bs, 64, 100, 22]
    inter = self.inter_norm(inter)                                  ## [B, F, K, S] -> [bs, 64, 100, 22]
    out = inter + x                                     ## [B, F, K, S] -> [bs, 64, 100, 22]

    return out


class Mask_Generator(nn.Module):
  def __init__(self, configs):
    super(Mask_Generator, self).__init__()
    self.configs = configs

    self.in_channels = configs.encoder.encoder_dim 
    self.feature_dim = configs.separator.feature_dim
    self.hidden_dim = configs.separator.hidden_dim
    
    self.K = self.configs.separator.K
    self.stack_num = self.configs.separator.stack_num

    # Norm
    self.norm = select_norm(self.configs.separator.norm, self.in_channels, 3)

    # Conv1D
    self.conv1d = nn.Conv1d(self.in_channels, self.feature_dim, 1, bias=False)

    # Dual-Path-Tas
    self.dual_tas_local = nn.ModuleList([])
    self.dual_tas_global = nn.ModuleList([])
    for _ in range(self.stack_num):
      self.dual_tas_local.append(Intra_Block(self.configs))
    for _ in range(self.stack_num):
      self.dual_tas_global.append(Inter_Block(self.configs))

    self.conv2d = nn.Conv2d(self.feature_dim, self.feature_dim, kernel_size=1)
    self.end_conv1x1 = nn.Conv1d(self.feature_dim, self.in_channels, 1, bias=False)
    self.prelu = nn.PReLU()
    self.activation = nn.ReLU()

    # gated output layer
    self.output = nn.Sequential(
      nn.Conv1d(self.feature_dim, self.feature_dim, 1), 
      nn.Tanh()
      )
    self.output_gate = nn.Sequential(
      nn.Conv1d(self.feature_dim, self.feature_dim, 1), 
      nn.Sigmoid()
      )


  def _padding(self, input, K):
    '''
      padding the audio times
      K: chunks of length
      P: hop size
      input: [B, N, L]
    '''
    B, N, L = input.shape
    P = K // 2
    gap = K - (P + L % K) % K
    # gap = P - (L - K) % P
    if gap > 0:
      pad = torch.Tensor(torch.zeros(B, N, gap)).type(input.type())
      input = torch.cat([input, pad], dim=2)

    _pad = torch.Tensor(torch.zeros(B, N, P)).type(input.type())
    input = torch.cat([_pad, input, _pad], dim=2)

    return input, gap


  def _segmentation(self, input, K):
      '''
        the segmentation stage splits
        K: chunks of length
        P: hop size
        input: [B, N, L]
        output: [B, N, K, S]
      '''
      B, N, L = input.shape
      P = K // 2
      input, gap = self._padding(input, K)
      # [B, N, K, S]
      input1 = input[:, :, :-P].contiguous().view(B, N, -1, K)
      input2 = input[:, :, P:].contiguous().view(B, N, -1, K)
      input = torch.cat([input1, input2], dim=3).view(
          B, N, -1, K).transpose(2, 3)

      return input.contiguous(), gap


  def _over_add(self, input, gap):
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


  def forward(self, x):
    '''
      x: [B, N, L]
    '''
    x = self.norm(x)                          ## [B, N, L] -> [bs, 256, 1022]
    x = self.conv1d(x)                        ## [B, F, L] -> [bs, 64, 1022]
    x, gap = self._segmentation(x, self.K)    ## [B, F, K, S] -> [bs, 64, 100, 22]

    for i in range(self.stack_num):
      x = self.dual_tas_global[i](x)                ## [B, F, K, S] -> [bs, 64, 100, 22]
    for i in range(self.stack_num):
      x = self.dual_tas_local[i](x)                 ## [B, F, K, S] -> [bs, 64, 100, 22]

    x = self.prelu(x)                         ## [B, F, K, S] -> [bs, 64, 100, 22]
    x = self.conv2d(x)                        ## [B, F, K, S] -> [bs, 64, 100, 22]

    B, _, K, S = x.shape
    x = x.view(B, -1, K, S)                   ## [B, F, K, S] -> [bs, 64, 100, 22]

    x = self._over_add(x, gap)                ## [B, F, L] -> [bs, 64, 1022]

    # x = self.output(x) * self.output_gate(x)  ## [B, F, L] -> [bs, 64, 1022]
    x = self.output_gate(x)                     ## [B, F, L] -> [bs, 64, 1022]

    x = self.end_conv1x1(x)                   ## [B, N, L] -> [bs, 256, 1022]
    x = self.activation(x)                    ## [B, N, L] -> [bs, 256, 1022]

    return x


class MSTP_Model_nonalternation_global_first(nn.Module):
  def __init__(self, config):
    super(MSTP_Model_nonalternation_global_first,self).__init__()
    self.config = config

    # Encoder
    self.encoder_dim = self.config.encoder.encoder_dim
    self.encoder_kernel_size = self.config.encoder.kernel_size
    self.encoder_stride = self.encoder_kernel_size // 2

    self.encoder = Encoder(
      kernel_size=self.encoder_kernel_size, 
      out_channels=self.encoder_dim, 
      stride=self.encoder_stride
      )

    # Separation
    self.separation = Mask_Generator(self.config)

    # Decoder 
    self.decoder_dim = self.encoder_dim
    self.decoder_kernel_size = self.encoder_kernel_size
    self.decoder_stride = self.encoder_stride
    
    self.decoder = Decoder(
      in_channels=self.decoder_dim, 
      out_channels=1, 
      kernel_size=self.decoder_kernel_size, 
      stride=self.decoder_stride, 
      bias=False
      )


  def pad_signal(self, input):
    '''
    Parameter:
      input: [bs, channel, length] -> [128, 1, 1024]
    '''
    # input is the waveforms: (B, T) or (B, 1, T)
    # reshape and padding
    if input.dim() not in [2, 3]:
      raise RuntimeError("Input can only be 2 or 3 dimensional.")

    if input.dim() == 2:
      input = input.unsqueeze(1)

    batch_size = input.size(0)
    nsample = input.size(2)
    rest = self.encoder_kernel_size - (self.encoder_stride + nsample % self.encoder_kernel_size) % self.encoder_kernel_size
    if rest > 0:
      pad = Variable(torch.zeros(batch_size, 1, rest)).type(input.type())
      input = torch.cat([input, pad], 2)

    pad_aux = Variable(torch.zeros(batch_size, 1, self.encoder_stride)).type(input.type())
    input = torch.cat([pad_aux, input, pad_aux], 2)

    return input, rest


  def forward(self, x):
    '''
      x: [B, C, L]
    '''
    x, rest = self.pad_signal(x)
    encoder_feature = self.encoder(x)           ## [B, N, L]
    mask = self.separation(encoder_feature)     ## [B, N, L]
    out = mask * encoder_feature                ## [B, N, L]
    output = self.decoder(out)                  ## [B, L]
    output = output[:, self.encoder_stride:-(rest + self.encoder_stride)].contiguous()  											# [Bs*C, 1, T]	->	[128 * c, 1, 1024]
    output = torch.unsqueeze(output, dim=1)     ## [B, C, L]

    return output



if __name__ == "__main__":
  pass


