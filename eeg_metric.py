import numpy as np
import torch



# region: Metric
def metric_rrmse_temporal(gt, pred):
	return _get_rms(pred - gt) / _get_rms(gt)


def metric_rrmse_spectrum(gt, pred):
	psd_gt = _get_psd(gt)
	psd_pred = _get_psd(pred)
	return _get_rms(psd_pred - psd_gt) / _get_rms(psd_gt)


def metric_cc(gt, pred):

	if len(gt.shape) == 3: gt = np.squeeze(gt, axis=1)
	if len(pred.shape) == 3: pred = np.squeeze(pred, axis=1)

	x, y = np.array(gt), np.array(pred)
	mx = np.mean(x, axis=1, keepdims=True)
	my = np.mean(y, axis=1, keepdims=True)
	xm, ym = x - mx, y - my
	numerator = np.sum(xm * ym, axis=1)
	denominator = np.sqrt(np.maximum(np.sum(np.square(xm), axis=1) * np.sum(np.square(ym), axis=1), 1e-12))

	return - numerator / denominator


def metric_snr(gt, pred):

	if len(gt.shape) == 3: gt = np.squeeze(gt, axis=1)
	if len(pred.shape) == 3: pred = np.squeeze(pred, axis=1)

	snr = [_get_snr(g, p) for g, p in zip(gt, pred)]

	return np.array(snr)


def _get_rms(tensor):
	if len(tensor.shape) == 3:
		tensor = np.squeeze(tensor, axis=1)
	return np.sqrt(np.mean(np.square(tensor), axis=1))


def _get_psd(tensor):
	if tensor.ndim == 3:
		data = np.squeeze(tensor, axis=1).astype(np.complex64)
	else:
		data = tensor.astype(np.complex64)

	fft_result = np.fft.fft(data, axis=1)
	psd = np.abs(np.square(fft_result) * 2)[:, :240]

	# 如果需要以 dB 返回 PSD，则取消下面的注释
	# psd = 20 * np.log10(psd)

	return psd


def _get_snr(ground_truth, pred):
	noise = pred - ground_truth
	return 10 * np.log10(np.sum(ground_truth ** 2) / np.sum(noise ** 2))


def get_metrics(metrics, gt, pred):
	metric_dict = { 
		'RRMSE_temporal': metric_rrmse_temporal, 
		'RRMSE_spectrum': metric_rrmse_spectrum, 
		'CC': metric_cc, 
		'SNR': metric_snr,
	}

	return metric_dict[metrics](gt, pred)

# endregion: Metric

# region: Loss
def lncosh_loss(gt, pred, lamda):
	'''
	Parameter:
		gt: Ground truth, shape: [B, L]
		pred: Prediction, shape: [B, L]
	'''
	EPS = 1e-8
	error = (pred - gt) 																											## [B, L]
	error = torch.log(torch.cosh(lamda * error)) / lamda											## [B, L]
	error = torch.mean(torch.sum(error, axis=1))     			 					
	if error.max() == float('inf'):
		error = torch.zeros_like(error)
	
	return error

# endregion: Loss

# Demo
def test_demo():
	gt_list = list(range(0, 100))
	pred_list = list(range(0, 100))[::-1]

	gt = np.array(gt_list).reshape(5, 20)
	pred = np.array(pred_list).reshape(5, 20)

	print(metric_rrmse_temporal(gt, pred))
	print(metric_rrmse_spectrum(gt, pred))
	print(metric_cc(gt, pred))



if __name__ == '__main__':
	test_demo()