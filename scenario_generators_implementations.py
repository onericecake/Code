import pprint

import numpy as np
from datetime import datetime
import statsmodels.api as sm
from arch import arch_model

from server_modules.libs.HistoricalSimulation import HistoricalSimulation
from server_modules.libs.SimilarityMetrics import SimilarityMetrics

'''
Scenario Generator Input and Output Format
## Please keep all inputs and outs in python native types if possible (including datetime objects)
## Please leave at least a brief model assumption description in the docstring

Input:
    data: list of data points, list[float]
    sim_length: number of data points to forecast, int
    n_sims: number of simulations to run, int
    **kwargs: model specific parameters, dict
Please note, forecast_length share the same data frequency as the data input, 
e.g. if the data is daily, then forecast_length is in days

Output:
    msg: dictionary of forecast results, dict
    The mandatory keys are:
        msg["generator_code_name"]: name of the model, str
        msg["run_time"]: time when the model is run, datetime
        msg["raw_data"]: raw data used for the forecast, list[float]
        msg["sim_length"]: number of data points to forecast, int
        msg["n_sims"]: number of simulations to run, int
        msg["status"]: status of the model, str
        msg["error"]: (Optional) error message if the model fails, str
        msg["forecast"]: forecast_column_names x forecast_length matrix for forecasted data, list[list[float]]
        
        msg["model_parameters"]: (Optional) other model parameters to be saved for testing and debugging, dict
        msg["forecast_column_names"]: (Optional) names of the forecast columns, can be a list of 1,2,3,..., list[str]
        msg["forecast_misc"]: (Optional) misc forecast information to be saved for testing and debugging, dict
        msg["stats_tests"]: (Optional) statistical tests for model testing etc, dict
        
    other keys can be added as well for model specific information

'''


def percentiles_as_dict(forecasts) -> dict:
    percentiles = {
        "min": np.min(forecasts, axis=1).tolist(),
        "5per": np.percentile(forecasts, 5, axis=1).tolist(),
        "10per": np.percentile(forecasts, 10, axis=1).tolist(),
        "20per": np.percentile(forecasts, 20, axis=1).tolist(),
        "30per": np.percentile(forecasts, 25, axis=1).tolist(),
        "40per": np.percentile(forecasts, 35, axis=1).tolist(),
        "50per": np.percentile(forecasts, 50, axis=1).tolist(),
        "60per": np.percentile(forecasts, 65, axis=1).tolist(),
        "70per": np.percentile(forecasts, 75, axis=1).tolist(),
        "80per": np.percentile(forecasts, 80, axis=1).tolist(),
        "90per": np.percentile(forecasts, 90, axis=1).tolist(),
        "95per": np.percentile(forecasts, 95, axis=1).tolist(),
        "max": np.max(forecasts, axis=1).tolist(),
        "mean": np.mean(forecasts, axis=1).tolist(),
        "std": np.std(forecasts, axis=1).tolist(),
    }
    return percentiles

## forecast given data set based on all historical log returns
def hs(data: list | np.ndarray, sim_length: int = 18) -> dict:
    model = HistoricalSimulation()
    model.fit(data)
    forecasts = model.simulate(sim_length)
    percentiles =percentiles_as_dict(forecasts)
    # sort columns by mean: forecasts: n_sims x forecast_length
    # Compute the average of each column
    column_averages = np.mean(forecasts, axis=0)
    # Get the indices that would sort the columns based on their average
    sorted_indices = np.argsort(column_averages)
    # Sort the array columns based on the sorted indices
    forecasts = forecasts[:, sorted_indices]
    forecasts = forecasts.tolist()
    n_sims = len(forecasts[0])
    # create the return messages
    msg = {
        "generator_code_name": "hs",
        "run_time": datetime.now(),
        "raw_data": data,
        "sim_length": sim_length,
        "n_sims": n_sims,
        "status": "success",
        "forecast": forecasts,
        "percentiles": percentiles,
    }
    return msg


def filtered_hs(data: list | np.ndarray, sim_length: int = 18) -> dict:
    model = HistoricalSimulation(filtering=True)
    model.fit(data)
    forecasts = model.simulate(sim_length)
    percentiles = percentiles_as_dict(forecasts)
    # sort columns by mean: forecasts: n_sims x forecast_length
    # Compute the average of each column
    column_averages = np.mean(forecasts, axis=0)
    # Get the indices that would sort the columns based on their average
    sorted_indices = np.argsort(column_averages)
    # Sort the array columns based on the sorted indices
    forecasts = forecasts[:, sorted_indices]
    forecasts = forecasts.tolist()
    n_sims = len(forecasts[0])
    # create the return messages
    msg = {
        "generator_code_name": "filtered_hs",
        "run_time": datetime.now(),
        "raw_data": data,
        "sim_length": sim_length,
        "n_sims": n_sims,
        "status": "success",
        "forecast": forecasts,
        "percentiles": percentiles,
    }
    return msg


def hs_bootstrap(data: list | np.ndarray,
        sim_length: int = 18, n_sims: int = 200) -> dict:
    model = HistoricalSimulation(bootstrap=True)
    model.fit(data)
    forecasts = model.simulate(sim_length, num_paths=n_sims)
    percentiles = percentiles_as_dict(forecasts)
    # sort columns by mean: forecasts: n_sims x forecast_length
    # Compute the average of each column
    column_averages = np.mean(forecasts, axis=0)
    # Get the indices that would sort the columns based on their average
    sorted_indices = np.argsort(column_averages)
    # Sort the array columns based on the sorted indices
    forecasts = forecasts[:, sorted_indices]
    forecasts = forecasts.tolist()
    n_sims = len(forecasts[0])
    # create the return messages
    msg = {
        "generator_code_name": "hs_bootstrap",
        "run_time": datetime.now(),
        "raw_data": data,
        "sim_length": sim_length,
        "n_sims": n_sims,
        "status": "success",
        "forecast": forecasts,
        "percentiles": percentiles,
    }
    return msg


def filtered_hs_bootstrap(data: list | np.ndarray,
         sim_length: int = 18, n_sims: int = 200) -> dict:
    model = HistoricalSimulation(filtering=True, bootstrap=True)
    model.fit(data)
    forecasts = model.simulate(sim_length, num_paths=n_sims)
    percentiles =percentiles_as_dict(forecasts)
    # sort columns by mean: forecasts: n_sims x forecast_length
    # Compute the average of each column
    column_averages = np.mean(forecasts, axis=0)
    # Get the indices that would sort the columns based on their average
    sorted_indices = np.argsort(column_averages)
    # Sort the array columns based on the sorted indices
    forecasts = forecasts[:, sorted_indices]
    forecasts = forecasts.tolist()
    n_sims = len(forecasts[0])
    # create the return messages
    msg = {
        "generator_code_name": "filtered_hs_bootstrap",
        "run_time": datetime.now(),
        "raw_data": data,
        "sim_length": sim_length,
        "n_sims": n_sims,
        "status": "success",
        "forecast": forecasts,
        "percentiles": percentiles,
    }
    return msg


def hs_bootstrap_seasonal(data: list | np.ndarray, freq: int,
         sim_length: int = 18, n_sims: int = 200) -> dict:
    model = HistoricalSimulation(bootstrap=True, weighting=f"S{freq}")
    model.fit(data)
    forecasts = model.simulate(sim_length, num_paths=n_sims)
    percentiles = percentiles_as_dict(forecasts)
    # sort columns by mean: forecasts: n_sims x forecast_length
    # Compute the average of each column
    column_averages = np.mean(forecasts, axis=0)
    # Get the indices that would sort the columns based on their average
    sorted_indices = np.argsort(column_averages)
    # Sort the array columns based on the sorted indices
    forecasts = forecasts[:, sorted_indices]
    forecasts = forecasts.tolist()
    n_sims = len(forecasts[0])
    # create the return messages
    msg = {
        "generator_code_name": "hs_bootstrap_seasonal",
        "run_time": datetime.now(),
        "raw_data": data,
        "sim_length": sim_length,
        "n_sims": n_sims,
        "status": "success",
        "forecast": forecasts,
        "percentiles": percentiles,
    }
    return msg


def filtered_hs_bootstrap_seasonal(data: list | np.ndarray, freq: int,
         sim_length: int = 18, n_sims: int = 200) -> dict:
    model = HistoricalSimulation(filtering=True, bootstrap=True,
                                 weighting=f"S{freq}")
    model.fit(data)
    forecasts = model.simulate(sim_length, num_paths=n_sims)
    percentiles = percentiles_as_dict(forecasts)
    # sort columns by mean: forecasts: n_sims x forecast_length
    # Compute the average of each column
    column_averages = np.mean(forecasts, axis=0)
    # Get the indices that would sort the columns based on their average
    sorted_indices = np.argsort(column_averages)
    # Sort the array columns based on the sorted indices
    forecasts = forecasts[:, sorted_indices]
    forecasts = forecasts.tolist()
    n_sims = len(forecasts[0])
    # create the return messages
    msg = {
        "generator_code_name": "filtered_hs_bootstrap_seasonal",
        "run_time": datetime.now(),
        "raw_data": data,
        "sim_length": sim_length,
        "n_sims": n_sims,
        "status": "success",
        "forecast": forecasts,
        "percentiles": percentiles,
    }
    return msg


## forecast given data set based on same frequency return ratios from historical data
def hs_freq_bootstrap(data: list | np.ndarray, freq: int,
                      sim_length: int = 18,
                      n_sims: int = 200) -> dict:
    # obtain the return ratio
    data_rr = np.array(data[1:]) / np.array(data[:-1])
    # obtain the same frequency returns
    # create a 2d list with each element as a return series for a frequency count
    data_lr_freq = [data_rr[-i::-freq] for i in range(1, freq + 1)]
    # make forecast by simulate the next value by taking frequency returns
    sims = np.empty((n_sims, sim_length))
    sims[::] = np.nan
    for i in range(sim_length):
        freq_counter = i % freq
        freq_series = data_lr_freq[freq - freq_counter - 1]
        # randomly choose the returns from the corresponding frequency series
        this_return = np.random.choice(freq_series, n_sims)
        # add the return to the last value of the data
        if i == 0:
            sims[:, i] = data[-1] * this_return
        else:
            sims[:, i] = sims[:, i - 1] * this_return

    forecasts = sims.T
    percentiles = percentiles_as_dict(forecasts)
    # sort columns by mean: forecasts: n_sims x forecast_length
    # Compute the average of each column
    column_averages = np.mean(forecasts, axis=0)
    # Get the indices that would sort the columns based on their average
    sorted_indices = np.argsort(column_averages)
    # Sort the array columns based on the sorted indices
    forecasts = forecasts[:, sorted_indices]
    forecasts = forecasts.tolist()

    # create the return messages
    msg = {
        "generator_code_name": "hs_freq_bootstrap",
        "run_time": datetime.now(),
        "raw_data": data,
        "sim_length": sim_length,
        "n_sims": n_sims,
        "status": "success",
        "model_parameters": {
            "freq": freq,
        },
        "forecast": forecasts,
        "percentiles": percentiles,
    }
    return msg


# SARIMA model, generate forecast based on selected sarima model
# model assumption: model residuals are IID and normally distributed, only works for monthly data
def sarima(data: list, order: list[int],
           sim_length: int = 18, n_sims: int = 200) -> dict:
    # run the sarima model
    # order: [p, d, q, P, D, Q, s] where s is the seasonal period

    # model fitting
    AR, D, MA, SAR, SD, SMA, SS = order
    model = sm.tsa.statespace.SARIMAX(data, trend='c', order=(AR, D, MA), seasonal_order=(SAR, SD, SMA, SS))
    sarima_res = model.fit(disp=0)
    forecasts = np.array(
        [sarima_res.simulate(sim_length, anchor='end') for x in range(n_sims)]).T
    percentiles = percentiles_as_dict(forecasts)
    # sort columns by mean: forecasts: n_sims x forecast_length
    # Compute the average of each column
    column_averages = np.mean(forecasts, axis=0)
    # Get the indices that would sort the columns based on their average
    sorted_indices = np.argsort(column_averages)
    # Sort the array columns based on the sorted indices
    forecasts = forecasts[:, sorted_indices]
    forecasts = forecasts.tolist()

    # save results
    msg = {"generator_code_name": "sarima", "run_time": datetime.now(), "raw_data": data, "sim_length": sim_length,
           "n_sims": n_sims, "status": "success", "stats_tests": {
            'AIC_BIC_HQIC': sarima_res.aic + sarima_res.bic + sarima_res.hqic,
            'AIC_BIC': sarima_res.aic + sarima_res.bic,
            'HQIC': sarima_res.hqic,
            'std_resid': np.nanstd(sarima_res.resid),
            'sarima_summary': sarima_res.summary().as_text(),
        }, "model_parameters": {
            "AR": AR,
            "D": D,
            "MA": MA,
            "SAR": SAR,
            "SD": SD,
            "SMA": SMA,
            "SS": SS,
        },
           'forecast': forecasts,
           'percentiles': percentiles,
           }
    return msg


## mutil-hsb orecast given data set based on same frequency return ratios from historical data sets
def hs_freq_bootstrap_multi(data: np.ndarray, freq: int,
                            sim_length: int = 18,
                            n_sims: int = 200) -> dict:
    # data.shape[0] should be number of timesteps

    # obtain the return ratio
    data_rr = np.array(data[1:, :]) / np.array(data[:-1, :])
    # obtain the same frequency returns
    # create a 2d list with each element as a return series for a frequency count
    data_lr_freq = [data_rr[-i::-freq, :] for i in range(1, freq + 1)]
    # make forecast by simulate the next value by taking frequency returns
    n_dim = data.shape[1]
    sims = np.empty((n_sims, sim_length, n_dim))
    sims[::] = np.nan
    for i in range(sim_length):
        freq_counter = i % freq
        freq_series = data_lr_freq[freq - freq_counter - 1]
        # randomly choose the returns from the corresponding frequency series
        choices = np.random.choice(np.array(len(freq_series)), n_sims)
        this_return = freq_series[choices]
        # add the return to the last value of the data
        if i == 0:
            sims[:, i] = data[-1] * this_return
        else:
            sims[:, i] = sims[:, i - 1] * this_return

    forecasts = sims.T
    # calculate percentiles
    percentiles = percentiles_as_dict(forecasts)
    # sort columns by mean: forecasts: num_SICs * forecast_length * n_sims
    sorted_array_3d = np.empty_like(forecasts)
    # Sort each 2D slice along the first dimension
    for i, slice_2d in enumerate(forecasts):
        column_averages = np.mean(slice_2d, axis=0)
        sorted_indices = np.argsort(column_averages)
        sorted_array_3d[i] = slice_2d[:, sorted_indices]
    forecasts = sorted_array_3d.tolist()

    # create the return messages
    msg = {
        "generator_code_name": "hs_freq_bootstrap_multi",
        "run_time": datetime.now(),
        "raw_data": data,
        "sim_length": sim_length,
        "n_sims": n_sims,
        "status": "success",
        "model_parameters": {
            "freq": freq,
        },
        "forecast": forecasts,
        "percentiles": percentiles,
    }
    return msg

# SARIMA model, generate forecast based on selected sarima model
# model assumption: model residuals are IID and normally distributed, only works for monthly data
def sarima_garch(data: list, sarima_order: list[int], garch_order: list[int],
           sim_length: int = 18, n_sims: int = 200) -> dict:
    # run the sarima model
    # sarima_order: [p, d, q, P, D, Q, s] where s is the seasonal period
    # garch_order: [gp, pq]: The order of the GARCH model (p,q).

    # model fitting
    AR, D, MA, SAR, SD, SMA, SS = sarima_order
    model = sm.tsa.statespace.SARIMAX(data, trend='c', order=(AR, D, MA), seasonal_order=(SAR, SD, SMA, SS))
    sarima_res = model.fit(disp=0)
    # Get SARIMA residuals
    sarima_resid = sarima_res.resid

    # Fit GARCH model to the SARIMA residuals
    garch = arch_model(sarima_resid, vol='Garch', p=garch_order[0], q=garch_order[1])
    garch_fit = garch.fit(disp=0)

    # Forecast using SARIMA
    sarima_forecast = np.array(
        [sarima_res.simulate(sim_length, anchor='end') for x in range(n_sims)])

    # Forecast the conditional volatility using GARCH
    garch_sim = garch_fit.forecast(start=0, horizon=sim_length, method='simulation', simulations=n_sims)
    garch_sim_values = np.sqrt(garch_sim.variance.values[-1, :]) * np.random.normal(0, 1, (n_sims, sim_length))

    # Simulate future data points
    forecasts = sarima_forecast + garch_sim_values
    forecasts = forecasts.T

    percentiles =percentiles_as_dict(forecasts)
    # sort columns by mean: forecasts: n_sims x forecast_length
    # Compute the average of each column
    column_averages = np.mean(forecasts, axis=0)
    # Get the indices that would sort the columns based on their average
    sorted_indices = np.argsort(column_averages)
    # Sort the array columns based on the sorted indices
    forecasts = forecasts[:, sorted_indices]
    forecasts = forecasts.tolist()
    # save results
    msg = {"generator_code_name": "sarima",
           "run_time": datetime.now(),
           "raw_data": data,
           "sim_length": sim_length,
           "n_sims": n_sims,
           "status": "success",
           "stats_tests": {
            'AIC_BIC_HQIC': sarima_res.aic + sarima_res.bic + sarima_res.hqic,
            'AIC_BIC': sarima_res.aic + sarima_res.bic,
            'HQIC': sarima_res.hqic,
            'std_resid': np.nanstd(sarima_res.resid),
            'std_garch_resid': np.nanstd(garch_fit.resid),
            'garch_aic': garch_fit.aic,
            'garch_bic': garch_fit.bic,
            'sarima_summary': sarima_res.summary().as_text(),
            'garch_summary': garch_fit.summary().as_text(),

        }, "model_parameters": {
            "AR": AR,
            "D": D,
            "MA": MA,
            "SAR": SAR,
            "SD": SD,
            "SMA": SMA,
            "SS": SS,
            "garch_p": garch_order[0],
            "garch_q": garch_order[1],
        },
           'forecast': forecasts,
           'percentiles': percentiles,
           }
    return msg

# SimilarityMetrics
def similarity_metrics(y, y_hat, eval_basis):
    '''
    calculate similarity metrics for two time series
    :param y: list[float], actual time series
    :param y_hat: list[list[float]], forecasted time series in n_sims x forecast_length
    :param eval_basis: str, basis to evaluate the similarity, can be "price", "return", "log_return"
    '''
    sm = SimilarityMetrics()
    return sm.evaluate_simulations(y, y_hat, eval_basis)


# test function
if __name__ == "__main__":
    # create testing data
    mean = 0
    std = 0.02
    data = np.random.normal(mean, std, 500)
    data = np.cumprod(np.exp(data))

    # run forecast
    # sarima_garch(data, [1,0,1,1,0,1,12], [1,1])
    # hs_freq_bootstrap_multi(np.array([data, data, data + 1]).T,freq=12)
    d = hs(data)
    d = filtered_hs(data)
    d = hs_bootstrap(data)
    d = filtered_hs_bootstrap(data)
    d = hs_bootstrap_seasonal(data, freq=12)
    d = filtered_hs_bootstrap_seasonal(data, freq=12)
    # hs_freq_bootstrap(data, freq=12)
    # sarima(data, order=[1, 0, 1, 1, 0, 1, 12])
