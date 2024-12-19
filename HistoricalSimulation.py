from server_modules.libs.NonParametricUnivariate import NonParametricUnivariateSG
from server_modules.libs.ModelsForFiltering import EWMA_OU, EWMA_Rescale

import numpy as np
import warnings


def is_weighing_valid(weighting: str):
    if weighting in ["uniform", "seasonal"]:
        return True
    if weighting[0] == "s" and weighting[1:].isdigit():
        return True
    return False


class HistoricalSimulation(NonParametricUnivariateSG):
    def __init__(self, **hyperparams):
        super().__init__(**hyperparams)

        self.bootstrap = self.hyperparams.get('bootstrap', False)
        self.filtering = self.hyperparams.get('filtering', False)
        self.backtest_only = self.hyperparams.get('backtest_only', False)
        self.weighting = self.hyperparams.get('weighting', "uniform").lower()
        if not is_weighing_valid(self.weighting):
            raise ValueError
        # can be uniform, or seasonal, or overriden by a parameter in simulate

        if self.filtering:
            vol_model = self.hyperparams.get('vol_model', 'ewma-ou').lower()

            if vol_model == 'ewma-ou':
                _lambda = self.hyperparams.get('lambda', 0.9)
                self.vol_model = EWMA_OU(_lambda)

            if vol_model == 'ewma_rescale':
                _lambda = self.hyperparams.get('lambda', 0.9)
                self.vol_model = EWMA_Rescale(_lambda)

    def fit(self, historical_price_train, exclude_last_from_training=False,
            **kwargs):

        self.date_array, self.price_array = self.convert_to_array(
            historical_price_train)
        return_array = np.diff(np.log(self.price_array))

        if exclude_last_from_training:
            return_array = return_array[:-1]

        if self.filtering:
            self.vol_model.fit(return_array)
            self.residuals = return_array / self.vol_model.hist_vol_array

        else:
            self.residuals = return_array

    def simulate(self, num_timesteps, num_paths=1, **kwargs):
        self._reset_to_seed(kwargs.get('reset_to_seed', None))
        triangular = kwargs.get('triangular', False)
        block_size = kwargs.get('block_size', 1)
        self.weight_multiplier = kwargs.get('weight_multiplier', None)
        # if supplied, must be of shape (num_timesteps-1)

        if self.backtest_only:
            simulated_price_paths = self.backtest(num_timesteps=num_timesteps,
                                                  triangular=triangular,
                                                  **kwargs)
            self._save_simulations(kwargs.get('save_simulations', False),
                                   simulated_price_paths)
            return simulated_price_paths
        prepend_latest_price = kwargs.get('prepend_latest_price', False)

        if self.filtering:
            self.forecast_vol = self.vol_model.forecast(num_timesteps).reshape(
                (-1, 1))
        else:
            self.forecast_vol = 1

        residuals = self.residuals
        if self.bootstrap:
            # construct probabilities from weighting
            if self.weighting == "uniform":
                weights = np.ones((residuals.shape[0], num_timesteps))

            elif self.weighting == "seasonal":
                weights = np.zeros((residuals.shape[0], num_timesteps))
                for i in range(num_timesteps):
                    this_month_idx = (len(residuals) + i) % 12
                    idx_pool = np.arange(len(residuals))
                    weights[idx_pool % 12 == this_month_idx, i] = 1

            elif self.weighting[0] == "s" and self.weighting[1:].isdigit():
                period = int(self.weighting[1:])
                weights = np.zeros((residuals.shape[0], num_timesteps))
                for i in range(num_timesteps):
                    this_period_idx = (len(residuals) + i) % period
                    idx_pool = np.arange(len(residuals))
                    weights[idx_pool % period == this_period_idx, i] = 1

            else:
                raise ValueError

            # apply the weights multiplier
            if self.weight_multiplier is None:
                pass
            else:
                weights *= self.weight_multiplier[:, None]

            # normalise
            weights /= np.sum(weights, axis=0)

            # regular bootstrap
            if block_size == 1:
                simulated_residuals = []
                for i in range(num_timesteps):
                    selected_residual = self.random_state.choice(residuals, size=num_paths, replace=True, p=weights[:, i])
                    simulated_residuals.append(selected_residual)
                simulated_residuals = np.array(simulated_residuals)

            # bootstrap by blocks
            else:
                # initialise selection to remove pycharm warning
                selected_idx = 0

                simulated_residuals = []
                for i in range(0, num_timesteps):
                    if i % block_size == 0:
                        if i < block_size*(num_timesteps//block_size):
                            this_block_size = block_size
                        else:
                            this_block_size = (num_timesteps % block_size)

                        probability = weights[:-(this_block_size-1), i]
                        if probability.sum() == 0: #in the very rare case probability becomes zero
                            warnings.warn("Zero weighting detected, defaulting to uniform weighting")
                            probability = np.ones(len(probability))

                        probability /= np.sum(probability)

                        selected_idx = self.random_state.choice(
                            np.arange(len(residuals)-(this_block_size-1)),
                            size=num_paths, replace=True, p=probability)
                    else:
                        # so that next return follows previous one
                        selected_idx += 1
                    selected_residual = residuals[selected_idx]
                    simulated_residuals.append(selected_residual)
                simulated_residuals = np.array(simulated_residuals)

        # non-bootstrapping
        else:
            if triangular:
                num_pad_at_start = kwargs.get('num_pad_at_start', 0)
                num_pad_at_end = kwargs.get('num_pad_at_end', num_timesteps)
                residuals = np.concatenate(
                    [[np.nan] * num_pad_at_start, self.residuals,
                     [np.nan] * num_pad_at_end])
            pool = np.lib.stride_tricks.sliding_window_view(residuals,
                                                            num_timesteps)  # num_paths will be ignored
            simulated_residuals = pool.T

        simulated_returns = simulated_residuals * self.forecast_vol

        if prepend_latest_price:
            simulated_returns = np.concatenate(
                [np.zeros((1, simulated_returns.shape[1])), simulated_returns],
                axis=0)  # much faster way than append, insert, etc

        simulated_price_paths = np.exp(np.cumsum(simulated_returns, axis=0)) * \
                                self.price_array[-1]
        self._save_simulations(kwargs.get('save_simulations', False),
                               simulated_price_paths)
        return simulated_price_paths

    def backtest(self, num_timesteps, triangular=False, **kwargs):
        if triangular:
            num_pad_at_start = kwargs.get('num_pad_at_start', 0)
            num_pad_at_end = kwargs.get('num_pad_at_end', num_timesteps)
            price_array = np.concatenate(
                [[np.nan] * num_pad_at_start, self.price_array,
                 [np.nan] * num_pad_at_end])
        else:
            price_array = self.price_array
        pool = np.lib.stride_tricks.sliding_window_view(price_array,
                                                        num_timesteps + 1)  # t0 and tN inclusive => hence N+1 price data
        out = pool.T
        return out

