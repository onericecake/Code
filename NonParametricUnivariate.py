from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

class NonParametricUnivariateSG(ABC):
	def __init__(self, **hyperparams):
		self.hyperparams = hyperparams
		self.random_state = np.random.RandomState(hyperparams.get('seed', None))


	@staticmethod
	def convert_to_array(historical_price_train):
		date_array = None  # initialization

		# if date and price are provided as a tuple, separate them
		if isinstance(historical_price_train, tuple):
			date_array, historical_price_train = historical_price_train
		elif isinstance(historical_price_train, list):
			# there are two possibilites
			# 1. it is same as tuple
			if len(historical_price_train) == 2 and \
					isinstance(historical_price_train[1], (pd.Series, pd.DataFrame,np.ndarray, list)):
				date_array, historical_price_train = historical_price_train
			# 2. it is a list of prices - handled below

		# then get the price array and date_array from whatever object it is
		# if from tuple, check and convert price to np.ndarray
		# if tuple with date but also date in the price, then the date in the price will supersede
		if isinstance(historical_price_train, pd.Series):
			price_array = historical_price_train.values
			date_array = historical_price_train.index
		elif isinstance(historical_price_train, pd.DataFrame):
			price_array = historical_price_train.values[:, 0]
			date_array = historical_price_train.index
		elif isinstance(historical_price_train, list):
			price_array = np.array(historical_price_train)
		elif isinstance(historical_price_train, np.ndarray):
			price_array = historical_price_train
		else:
			raise TypeError("Input data cannot be converted into a numpy array!")

		# make sure date_array is np.datetime64[D]
		if isinstance(date_array, pd.DatetimeIndex):
			date_array = date_array.values.astype('datetime64[D]')

		return date_array, price_array

	@abstractmethod
	def fit(self, historical_price_train, exclude_last_from_training=False):
		self.price_array = self.convert_to_array(historical_price_train)

		# TODO for model fitting / parameter estimation


	@abstractmethod
	def simulate(self, num_timesteps, num_paths=1, **kwargs):
		pass


	@staticmethod
	def attach_datetime_to_simulations(simulated_price_paths, date_array):
		# TODO
		# if date array is provided, attach to simulated price paths, check for lengths
		# if not, we need to get the last date of the training data, infer the freq or explicitly be provided the frequency

		# ACTUALLY, will such date information be available from the source of data?

		pass

	def _reset_to_seed(self, seed):
		if seed is None:
			pass  # do nothing
		elif isinstance(seed, int):
			self.random_state.set_state(np.random.RandomState(seed).get_state())
		elif isinstance(seed, (np.random.RandomState, np.random.Generator)):
			self.random_state = seed  # using external generator basically
		# however currently it does not work on sarima...


	def _save_simulations(self, save_or_not, simulated_price_paths):
		if save_or_not:
			self.simulated_price_paths = simulated_price_paths
		else:
			self.simulated_price_paths = None