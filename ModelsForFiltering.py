import numpy as np

class EWMA_OU():

	def __init__(self, _lambda):
		self._lambda = _lambda

	@staticmethod
	def calculate_vol_ewma(return_array, _lambda=0.9):
		variances = np.zeros_like(return_array)
		variances[0] = np.maximum(return_array[0]**2, 1e-8)
		for i in range(1, len(return_array)):
			variances[i] = _lambda * variances[i-1] + (1 - _lambda) * return_array[i]**2
		return np.sqrt(variances)

	def fit(self, return_array):
		# log-transform volatilities
		# assume log-vol follows OU process
		# use OLS method
		# since we do not simulate, but just forecast, we do not need to solve OU parameters
		# treating it as AR(1)
		self.hist_vol_array = self.calculate_vol_ewma(return_array, self._lambda)
		log_vols = np.log(self.hist_vol_array)
		size = len(log_vols)
		X = np.hstack([np.ones((size-1,1)), log_vols[:-1, None]])
		y = log_vols[1:, None]

		ols_sol = np.linalg.inv(X.T @ X) @ X.T @ y

		self.intercept, self.slope = ols_sol[0][0], ols_sol[1][0]

		if not (0 < ols_sol[1][0] < 1):
			print(f"Failed to Fit AR(1) / OU: slope {ols_sol[1][0]}, Fallback to random walk.")
			self.intercept, self.slope = 0, 1

		self.anchor = log_vols[-1]



	def forecast(self, num_timesteps):
		t = np.arange(1, num_timesteps+1)
		if self.slope != 1:
			log_vols_forecast = self.intercept / (1 - self.slope) * (1 - self.slope**t) + self.anchor * self.slope**t
		else:
			log_vols_forecast = self.intercept + self.anchor

		return np.exp(log_vols_forecast)


# should work for both uni and multi
class EWMA_Rescale():
	def __init__(self, _lambda):
		self._lambda = _lambda

	@staticmethod
	def calculate_vol_ewma(return_array, _lambda=0.9):
		variances = np.zeros_like(return_array)
		variances[0] = np.maximum(return_array[0]**2, 1e-8)
		for i in range(1, len(return_array)):
			variances[i] = _lambda * variances[i-1] + (1 - _lambda) * return_array[i]**2
		return np.sqrt(variances)

	def fit(self, return_array):
		self.hist_vol_array = self.calculate_vol_ewma(return_array, self._lambda)
		self.last_vol = self.hist_vol_array[-1]

	def forecast(self, num_timesteps):
		return self.last_vol


class ARMA_GARCH():
	# worth checking ParametricUnivariateSG
	pass


class StoVol():

	# rt = exp(ht/2) * N(0,1)
	# ht ~ some (AR) process with N(0,1) noise
	# two noises are uncorrelated (for simplicity sake)
	# parameters are estimated by MLE, where ht is log variance (hidden state)

	pass

