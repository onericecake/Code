import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, wasserstein_distance, pearsonr
from statsmodels.tsa.stattools import acf

class SimilarityMetrics():

	@staticmethod
	def convert_to_array(historical_price_train):
		if isinstance(historical_price_train, pd.Series):
			price_array = historical_price_train.values
		elif isinstance(historical_price_train, pd.DataFrame):
			price_array = historical_price_train.values[:, 0]
		elif isinstance(historical_price_train, list):
			price_array = np.array(historical_price_train)
		elif isinstance(historical_price_train, np.ndarray):
			price_array = historical_price_train
		else:
			raise TypeError("Input data cannot be converted into a numpy array!")

		return price_array

	@staticmethod
	def convert_to_2darray(simulated_price_paths):
		if isinstance(simulated_price_paths, pd.Series):
			price_array = simulated_price_paths.values
		elif isinstance(simulated_price_paths, pd.DataFrame):
			price_array = simulated_price_paths.values
		elif isinstance(simulated_price_paths, list):
			price_array = np.array(simulated_price_paths)
		elif isinstance(simulated_price_paths, np.ndarray):
			price_array = simulated_price_paths
		else:
			raise TypeError("Input data cannot be converted into a numpy array!")

		if price_array.ndim == 1:
			raise TypeError("Simulated Price Paths should be 2-dimensional: Number of Timesteps * Number of Paths")

		return price_array

	@staticmethod
	def rmse(x, y=0, mean_axis=0):
		return np.sqrt(np.mean((x - y)**2, axis=mean_axis))

	@staticmethod
	def l2_norm(x, y=0, sum_axis=0):
		return np.sqrt(np.sum((x - y)**2, axis=sum_axis))

	@staticmethod
	def qvar(x, axis=0):
		return np.sum(np.diff(x, axis=axis)**2, axis=axis)

	@staticmethod
	def acf_all_lags(x):
		'''
		x is T * N arrays
		this function would return N * T array ()
		'''
		return np.array([acf(x[:, i], nlags=x.shape[0]-1, fft=False) for i in range(x.shape[1])])

	@staticmethod
	def move_direction(x, axis=0):
		return np.sign(np.diff(x, axis=axis))

	@staticmethod
	def transform(x):
		'''
		transform to between zero and one
		'''
		return x / (1+x)


	def evaluate_simulations(self, historical_price_test, similated_price_paths, eval_basis='price', metrics=[]):
		realized = self.convert_to_array(historical_price_test).reshape((-1, 1)) # to convert it to 2d array as well for dimensionality matching
		simulated = self.convert_to_2darray(similated_price_paths)

		if eval_basis == 'price':
			pass # do nothing
		elif eval_basis == 'return':
			realized = np.diff(realized, axis=0) / realized[:-1]
			simulated = np.diff(simulated, axis=0) / simulated[:-1, :]
		elif eval_basis == 'log_return':
			realized = np.diff(np.log(realized), axis=0)
			simulated = np.diff(np.log(simulated), axis=0)

		metric_results = {}
		
		if not metrics:
			metrics = ['mean_score', 'p95_score', 'p05_score', 'skew_score', 'kurt_score', 'qvar_score', 'acf_score', 'acf_sq_score', 
						'rmse_score', 'emd_score', 'corr_score', 'gf_score', 'ds_score']
		
		for metric in metrics:
			score = None # to clear existing memory

			# these metrics calculate some statistical properties of each path and quantify how different they are with those of the actual realized path
			if metric == 'mean_score': mean_score = self.rmse(np.mean(simulated, axis=0), np.mean(realized)); score = self.transform(mean_score)
			if metric == 'p95_score': p95_score = self.rmse(np.percentile(simulated, 95, axis=0), np.percentile(realized, 95)); score = self.transform(p95_score)
			if metric == 'p05_score': p05_score = self.rmse(np.percentile(simulated, 5, axis=0), np.percentile(realized, 5)); score = self.transform(p05_score)
			if metric == 'skew_score': skew_score = self.rmse(skew(simulated, axis=0), skew(realized)); score = self.transform(skew_score)
			if metric == 'kurt_score': kurt_score = self.rmse(kurtosis(simulated, axis=0), kurtosis(realized)); score = self.transform(kurt_score)
			if metric == 'qvar_score': qvar_score = self.rmse(self.qvar(simulated, axis=0), self.qvar(realized)); score = self.transform(qvar_score)
			if metric == 'acf_score': acf_score = self.l2_norm(np.mean(self.acf_all_lags(simulated), axis=0), np.mean(self.acf_all_lags(realized), axis=0)); score = self.transform(acf_score)
			if metric == 'acf_sq_score': acf_sq_score = self.l2_norm(np.mean(self.acf_all_lags(simulated)**2, axis=0), np.mean(self.acf_all_lags(realized)**2, axis=0)); score = self.transform(acf_sq_score)
			
			# these metrics calculate the (dis)similarity of each path from(to) he actual realized path and calculate the mean of such measure
			if metric == 'rmse_score': rmse_score = np.mean(self.rmse(simulated, realized, mean_axis=0)); score = self.transform(rmse_score) 
			if metric == 'l2_score': l2_score = np.mean([self.l2_norm(simulated[:, i], realized[:, 0]) for i in range(simulated.shape[1])]); score = self.transform(l2_score) 
			if metric == 'emd_score': emd_score = np.mean([wasserstein_distance(simulated[:, i], realized[:, 0]) for i in range(simulated.shape[1])]); score = self.transform(emd_score) 
			if metric == 'corr_score': corr_score = np.mean([pearsonr(simulated[:, i], realized[:, 0])[0] for i in range(simulated.shape[1])]); score = (1 - corr_score) / 2  # similarity measure here

			gf_mask = np.where(realized != 0)[0]
			if metric == 'gf_positive_score' or metric == 'gf_score': gf_positive_score = np.percentile(
				(simulated[gf_mask] - realized[gf_mask]) / np.abs(realized[gf_mask]), 95); score = self.transform(
				gf_positive_score)
			if metric == 'gf_negative_score' or metric == 'gf_score': gf_negative_score = np.percentile(
				(simulated[gf_mask] - realized[gf_mask]) / np.abs(realized[gf_mask]), 5); score = self.transform(
				gf_negative_score)
			if metric == 'gf_score': gf_score = max(gf_positive_score, 0) - min(gf_negative_score,
																				0); score = self.transform(gf_score)

			if metric == 'ds_score': ds_score = np.mean(self.move_direction(simulated) != self.move_direction(realized)); score = self.transform(ds_score) # similarity measure here

			metric_results[f"{eval_basis}_{metric}"] = score

		return metric_results





