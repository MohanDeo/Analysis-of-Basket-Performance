import itertools
import os
import random
import subprocess
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from arch import arch_model
from scipy.stats import kstest, norm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.diagnostic import het_arch, het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.api import VAR
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, grangercausalitytests

warnings.filterwarnings('ignore')

# Set for reproducibility
random.seed(7809)
os.makedirs("exploratory_analysis", exist_ok=True)
os.makedirs("further_analysis", exist_ok=True)
os.makedirs("main_analysis", exist_ok=True)

# Set this to true if you want ARIMA-GARCH models to be picked by AIC value. Otherwise, the lag, p and q are all set to
# 10. When set to True, there will be warnings from scipy's `slsqp` as it tries to optimise model params.
# The code takes over an hour to run with this value set to true, so by default it is False
optimise_aic = False


def save_plot(filename, folder="exploratory_analysis"):
    """
    Saves a matplotlib plot to file, closing the plot afterwards to prevent it from showing inline.

    Parameters
    ----------
    filename : str
        The filename to save the plot to.
    folder : str, optional
        The folder to save the plot in. Defaults to 'exploratory_analysis'.
    """
    filepath = os.path.join(folder, filename)
    plt.savefig(filepath, bbox_inches="tight")
    plt.close()


def read_in_data():
    """
        Reads in the CSV file 'SPX_Qi_TimeSeries.csv', which contains the time series of the SPX and Qi indices.
        The file is expected to contain columns 'Date', 'SPX' and 'Qi'.
        The 'Date' column is converted to a datetime and set as the index of the dataframe.
        The returns of the SPX and Qi are calculated as the percentage change of the respective columns.
        The first row of the dataframe is dropped as it becomes NaN when calculating the returns.
        The dataframe is then sorted by index to ensure that the data is in the correct order.
        The sorted dataframe is then returned.
        """
    data = pd.read_csv(
        'SPX_Qi_TimeSeries.csv',
        parse_dates=['Date'],
        index_col='Date'
    )
    # Account for the fact that the data may be in the incorrect order
    data.sort_index(inplace=True)
    # The current returns data is not percentage based
    data['SPX_Return'] = data['SPX'].pct_change()
    data['Qi_Return'] = data['Qi'].pct_change()
    # The first row becomes a NaN
    data.dropna(inplace=True)
    return data


def mean_annual_return(returns):
    """
    Calculates the mean annual return from a series of daily returns.

    Parameters
    ----------
    returns : pd.Series
        A series of daily returns.

    Returns
    -------
    float
        The mean annual return.

    Notes
    -----
    The mean annual return is calculated as follows:
    1. The cumulative return is calculated by taking the product of all the daily returns.
    2. The mean annual return is then calculated by taking the cumulative return to the power of 252 divided by the 
    number of days, and subtracting one.
    """
    cumulative_return = (1 + returns).prod()
    annual_return = cumulative_return ** (252 / len(returns)) - 1
    return annual_return


def annualised_volatility(returns):
    """
    Calculates the annualised volatility of a series of daily returns.

    Parameters
    ----------
    returns : pd.Series
        A series of daily returns.

    Returns
    -------
    float
        The annualised volatility.

    Notes
    -----
    The annualised volatility is calculated by multiplying the standard deviation of the daily returns by the
    square root of 252.
    """
    return returns.std() * np.sqrt(252)


def annualised_volatility_garch(returns, max_value=10):
    """
    Calculates the annualised volatility using an ARMA-GARCH model.

    Parameters
    ----------
    returns : pd.Series
        A series of daily returns.
    max_value : int, optional
        The maximum number of autoregressive and moving average terms to consider, by default 10.

    Returns
    -------
    float
        The annualised volatility.

    Notes
    -----
    The annualised volatility is calculated by fitting an ARMA-GARCH model to the data and then calculating the mean
    of the daily conditional volatility. The mean is then multiplied by the square root of 252 to give the annualised
    volatility.
    """
    best_aic = float('inf')
    best_model = None

    if optimise_aic == True:
        # Loop over combinations of p and q
        for p in range(max_value + 1):
            for q in range(max_value + 1):
                for lag in range(max_value + 1):
                    # Otherwise an error is raised `ValueError: One of p or o must be strictly positive`
                    if p == 0:
                        continue
                    if p == 0 and q == 0:
                        continue
                    garch_model = arch_model(returns, mean='AR', lags=lag, vol='Garch', p=p, q=q, dist='normal')
                    fitted_model = garch_model.fit(disp="off")
                    # Check AIC
                    aic = fitted_model.aic
                    if aic < best_aic:
                        best_aic = aic
                        best_model = fitted_model

    else:
        # The value for lag is informed by the rule of thumb T^1/4
        # The values for p and q are informed by the ACDF plot
        garch_model = arch_model(returns, mean='AR', lags=int(len(returns) ** 0.25), vol='Garch', p=14, q=14,
                                 dist='normal')
        best_model = garch_model.fit(disp="off")

    daily_conditional_volatility = best_model.conditional_volatility
    avg_daily_volatility = np.mean(daily_conditional_volatility)
    annualised_vol = avg_daily_volatility * np.sqrt(252)

    return annualised_vol


def sharpe_ratio(returns, risk_free_rate=0.02):
    """
    Calculate the Sharpe ratio of a given set of returns.

    Parameters
    ----------
    returns : pd.Series
        Returns of the asset or strategy.
    risk_free_rate : float, optional
        The risk-free rate. Defaults to 0.02 (2%).

    Returns
    -------
    sharpe_ratio : float or NaN
        The Sharpe ratio of the given returns. If the volatility is zero, returns NaN.
    """
    mean_return = returns.mean() * 252
    volatility = returns.std() * np.sqrt(252)
    # We need to deal with the case where the volatility is effectively zero
    if volatility < 1e-8:
        return np.nan
    return (mean_return - risk_free_rate) / volatility


def sortino_ratio(returns, risk_free_rate=0.02):
    """
    Calculate the Sortino ratio of a given set of returns.

    Parameters
    ----------
    returns : pd.Series
        Returns of the asset or strategy.
    risk_free_rate : float, optional
        The risk-free rate. Defaults to 0.02 (2%).

    Returns
    -------
    sortino_ratio : float or NaN
        The Sortino ratio of the given returns. If the downside standard deviation is zero, returns NaN.
    """
    mean_return = returns.mean() * 252
    negative_returns = returns[returns < 0]
    if len(negative_returns) == 0:
        return np.nan
    downside_std = negative_returns.std() * np.sqrt(252)
    # We need to deal with the case where the downside standard deviation is effectively zero
    if downside_std < 1e-8:
        return np.nan
    return (mean_return - risk_free_rate) / downside_std


def select_best_arma_model(time_series, max_p=5, max_q=5, criterion='aic'):
    """
        Selects the best ARMA model for a given time series based on the specified information criterion.

        Parameters
        ----------
        time_series : array-like
            The time series data for which the ARMA model is to be selected.
        max_p : int, optional
            The maximum order of the autoregressive part, by default 5.
        max_q : int, optional
            The maximum order of the moving average part, by default 5.
        criterion : str, optional
            The criterion to be used for model selection ('aic' or 'bic'), by default 'aic'.

        Returns
        -------
        tuple
            A tuple containing:
            - best_order : tuple
                The order (p, q) of the best ARMA model based on the specified criterion.
            - best_model : ARIMA
                The fitted ARIMA model with the best order.
            - results_df : pd.DataFrame
                A DataFrame containing the orders and corresponding AIC and BIC scores for all models considered.

        Notes
        -----
        The function assumes the input time series is stationary.
        """
    p = range(0, max_p + 1)
    q = range(0, max_q + 1)
    pq = list(itertools.product(p, q))

    best_score = np.inf
    best_order = None
    best_model = None
    results = []

    for order in pq:
        if order == (0, 0):
            continue
        # We have confirmed the data is stationary, so use 0 for differencing order
        model = ARIMA(time_series, order=(order[0], 0, order[1])).fit()
        if criterion == 'aic':
            score = model.aic
        else:
            score = model.bic
        results.append({'order': order, 'AIC': model.aic, 'BIC': model.bic})
        if score < best_score:
            best_score = score
            best_order = order
            best_model = model

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by=criterion.upper())
    return best_order, best_model, results_df


def check_residuals(model, lags=20):
    """
        Perform a Ljung-Box test on the residuals of a given model to check for autocorrelation.

        Parameters
        ----------
        model : statsmodels model
            The fitted model whose residuals are to be tested.
        lags : int, optional
            The number of lags to include in the test, by default 20.

        Returns
        -------
        float
            The p-value of the Ljung-Box test. A small p-value suggests that there is significant
            autocorrelation in the residuals.
        """
    residuals = model.resid
    lb_test = acorr_ljungbox(residuals, lags=[lags], return_df=True)
    p_value = lb_test['lb_pvalue'].values[0]
    return p_value


def calculate_sharpe_from_arma_model(model, risk_free_rate=0.02):
    """
    Calculate the Sharpe Ratio of a given ARMA model.

    Parameters
    ----------
    model : statsmodels ARMA model
        The fitted model for which to calculate the Sharpe Ratio.
    risk_free_rate : float, optional
        The risk-free rate, by default 0.02.

    Returns
    -------
    float
        The calculated Sharpe Ratio.

    Notes
    -----
    The Sharpe Ratio is a measure of risk-adjusted return, calculated as the excess return
    over the risk-free rate divided by the volatility of the returns.
    """
    mean_return = model.params.get('const', 0)
    mean_return_annualised = mean_return * 252

    sigma = np.sqrt(model.model.endog.var())
    sigma_annualised = sigma * np.sqrt(252)

    # Excess return
    excess_return = mean_return_annualised - risk_free_rate

    # Sharpe Ratio
    sharpe_ratio = excess_return / sigma_annualised

    return sharpe_ratio


def standard_deviation_sharpe_ratio(calculated_sharpe_ratio, num_obs, skewness, kurtosis):
    """
    Calculate the standard deviation of a calculated Sharpe Ratio.

    Parameters
    ----------
    calculated_sharpe_ratio : float
        The calculated Sharpe Ratio.
    num_obs : int
        The number of observations used to calculate the Sharpe Ratio.
    skewness : float
        The sample skewness of the returns.
    kurtosis : float
        The sample kurtosis of the returns.

    Returns
    -------
    float
        The standard deviation of the Sharpe Ratio.

    Notes
    -----
    The formula used is based on the approach described in
    "The Statistics of Sharpe Ratios" by Andrew W. Lo (2002).
    """
    return np.sqrt(
        (1 - skewness * calculated_sharpe_ratio +
         (kurtosis - 1) / 4 * calculated_sharpe_ratio ** 2
         ) / (num_obs - 1)
    )


def probabilistic_sharpe_ratio(calculated_sharpe_ratio, bench_sharpe_ratio, num_obs, skewness, kurtosis):
    """
        Calculate the Probabilistic Sharpe Ratio (PSR).

        The PSR is the probability that a Sharpe ratio is greater than a benchmark Sharpe ratio,
        accounting for the sample size, skewness, and kurtosis of the returns. It provides a measure
        of the likelihood that the observed Sharpe ratio reflects the true Sharpe ratio.

        Parameters
        ----------
        calculated_sharpe_ratio : float
            The Sharpe ratio calculated for the returns.
        bench_sharpe_ratio : float
            The benchmark Sharpe ratio to compare against.
        num_obs : int
            The number of observations in the sample.
        skewness : float
            The skewness of the returns distribution.
        kurtosis : float
            The kurtosis of the returns distribution.

        Returns
        -------
        float
            The Probabilistic Sharpe Ratio, representing the probability that the calculated
            Sharpe ratio is greater than the benchmark Sharpe ratio.
        """
    sr_diff = calculated_sharpe_ratio - bench_sharpe_ratio
    sr_vol = standard_deviation_sharpe_ratio(calculated_sharpe_ratio, num_obs, skewness, kurtosis)
    psr = norm.cdf(sr_diff / sr_vol)

    return psr


def excess_return(returns_qi, returns_spx):
    """
    Calculate the excess return of Qi portfolio over SPX.

    Parameters
    ----------
    returns_qi : pd.Series
        The daily returns of the Qi portfolio.
    returns_spx : pd.Series
        The daily returns of the SPX index.

    Returns
    -------
    float
        The excess return of the Qi portfolio over the SPX index.
    """
    mean_return_qi = returns_qi.mean() * 252
    mean_return_spx = returns_spx.mean() * 252
    return mean_return_qi - mean_return_spx


def tracking_error(returns_qi, returns_spx):
    """
    Calculate the tracking error of the Qi portfolio relative to the SPX index.

    The tracking error measures the standard deviation of the difference
    between the returns of the Qi portfolio and the SPX index, annualized
    by multiplying by the square root of 252.

    Parameters
    ----------
    returns_qi : pd.Series
        The daily returns of the Qi portfolio.
    returns_spx : pd.Series
        The daily returns of the SPX index.

    Returns
    -------
    float
        The annualized tracking error of the Qi portfolio relative to the SPX index.
    """
    difference = returns_qi - returns_spx
    return difference.std() * np.sqrt(252)


def information_ratio(calculated_excess_return, calculated_tracking_err):
    """
        Calculate the Information Ratio.

        The Information Ratio measures the excess return of an asset or portfolio
        relative to a benchmark, adjusted for the tracking error. It is a useful
        metric for assessing the risk-adjusted performance of an investment strategy
        compared to the benchmark.

        Parameters
        ----------
        calculated_excess_return : float
            The calculated excess return of the asset or portfolio over the benchmark.
        calculated_tracking_err : float
            The calculated tracking error, which is the standard deviation of the
            excess returns.

        Returns
        -------
        float
            The Information Ratio, representing the risk-adjusted excess return.
        """
    return calculated_excess_return / calculated_tracking_err


def regression_analysis(returns_qi, returns_spx, risk_free_rate=0.02):
    """
    Perform regression analysis of Qi returns against SPX returns.

    This function calculates the linear regression between the excess returns
    of Qi and SPX, adjusting for a daily risk-free rate. It computes the alpha,
    beta, R-squared, percentage errors of the coefficients, the p-value of alpha,
    and tests for autocorrelation and heteroskedasticity using the Durbin-Watson
    and Breusch-Pagan tests.

    Parameters:
    returns_qi (pd.Series): Daily returns of the Qi portfolio.
    returns_spx (pd.Series): Daily returns of the SPX index.
    risk_free_rate (float, optional): Annual risk-free rate. Default is 0.02.

    Returns:
    tuple: A tuple containing:
        - alpha (float): Annualized alpha of the regression.
        - beta (float): Beta of the regression.
        - r_squared (float): R-squared value of the regression.
        - percentage_errors (list of float): Percentage errors of the coefficients.
        - alpha_p_value (float): p-value of the alpha coefficient.
        - bp_p_value (float): p-value from the Breusch-Pagan test.
        - dw_stat (float): Durbin-Watson statistic.
    """
    risk_free_rate_daily = risk_free_rate / 252
    data = pd.concat([returns_qi, returns_spx], axis=1)
    # Calculate excess returns
    data['Excess_Qi_Return'] = data['Qi_Return'] - risk_free_rate_daily
    data['Excess_SPX_Return'] = data['SPX_Return'] - risk_free_rate_daily

    X = data['Excess_SPX_Return']
    y = data['Excess_Qi_Return']
    X = sm.add_constant(X)

    max_lags = int(len(returns_qi) ** 0.25)
    model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': max_lags})

    alpha_daily = model.params['const']
    alpha = (1 + alpha_daily) ** 252 - 1

    beta = model.params['Excess_SPX_Return']
    r_squared = model.rsquared

    coefficients = model.params
    errors = model.bse
    percentage_errors = []

    for coef_value, error in zip(coefficients, errors):
        percentage_error = (error / abs(coef_value)) * 100
        percentage_errors.append(percentage_error)

    alpha_p_value = model.pvalues['const']

    # Perform Durbin-Watson test for autocorrelation
    dw_stat = durbin_watson(model.resid)

    # Perform Breusch-Pagan test for heteroskedasticity
    bp_test = het_breuschpagan(model.resid, model.model.exog)
    bp_p_value = bp_test[1]

    return alpha, beta, r_squared, percentage_errors, alpha_p_value, bp_p_value, dw_stat


def main_analysis():
    """
    Perform financial analysis on Qi and SPX returns over different time periods.

    This function reads in return data, divides it into different time periods,
    and calculates various financial metrics for each period. The calculated
    metrics include mean annual return, annualised volatility, Sharpe ratio,
    Sortino ratio, excess return, tracking error, and information ratio.
    Additionally, regression analysis and ARMA model analysis are performed
    to derive further insights.

    The results of the analysis are saved to a CSV file for further review.

    Returns:
    --------
    pd.DataFrame
        A DataFrame containing the calculated metrics and analysis results
        for each time period.
    """
    data = read_in_data()
    end_date = data.index.max()
    one_year = data.loc[end_date - pd.DateOffset(years=1):end_date]
    three_years = data.loc[end_date - pd.DateOffset(years=3):end_date]
    five_years = data.loc[end_date - pd.DateOffset(years=5):end_date]
    inception = data

    periods = {
        '1 Year': one_year,
        '3 Years': three_years,
        '5 Years': five_years,
        'Inception': inception
    }

    results = {}
    for period_name, period_data in periods.items():
        spx_returns = period_data['SPX_Return']
        qi_returns = period_data['Qi_Return']

        # Mean Annual Return
        mean_return_spx = mean_annual_return(spx_returns)
        mean_return_qi = mean_annual_return(qi_returns)

        # Annualised Volatility
        vol_spx = annualised_volatility(spx_returns)
        vol_qi = annualised_volatility(qi_returns)
        garch_vol_spx = annualised_volatility_garch(spx_returns)
        garch_vol_qi = annualised_volatility_garch(qi_returns)

        # Sharpe Ratio (Assuming risk-free rate is 2%)
        sharpe_spx = sharpe_ratio(spx_returns)
        sharpe_qi = sharpe_ratio(qi_returns)
        qi_probabilistic_sharpe = probabilistic_sharpe_ratio(sharpe_qi, sharpe_spx, len(qi_returns), qi_returns.skew(),
                                                             qi_returns.kurt())

        # Sortino Ratio (Assuming risk-free rate is 2%)
        sortino_spx = sortino_ratio(spx_returns)
        sortino_qi = sortino_ratio(qi_returns)

        # Excess Return
        ex_return = excess_return(qi_returns, spx_returns)

        # Tracking Error
        track_err = tracking_error(qi_returns, spx_returns)

        # Information Ratio
        info_ratio = information_ratio(ex_return, track_err)

        # Regression Analysis
        alpha, beta, r_squared, percentage_errors, alpha_p_value, bp_p_value, dw_stat = regression_analysis(qi_returns,
                                                                                                            spx_returns)

        # ARMA Model Selection and Residual Analysis for Qi_Return
        best_order_qi, best_model_qi, _ = select_best_arma_model(qi_returns)

        # Check residuals of the best ARMA model
        p_value_qi = check_residuals(best_model_qi)

        # Calculate Sharpe Ratio from ARMA model
        arma_sharpe_ratio_qi = calculate_sharpe_from_arma_model(best_model_qi)

        # ARMA Model Selection and Residual Analysis for SPX_Return
        best_order_spx, best_model_spx, _ = select_best_arma_model(spx_returns)

        # Check residuals of the best ARMA model
        p_value_spx = check_residuals(best_model_spx)

        # Calculate Sharpe Ratio from ARMA model
        arma_sharpe_ratio_spx = calculate_sharpe_from_arma_model(best_model_spx)

        results[period_name] = {
            'Mean Annual Return SPX': round(mean_return_spx, 3),
            'Mean Annual Return Qi': round(mean_return_qi, 3),
            'Volatility Annualised SPX': round(vol_spx, 3),
            'Volatility Annualised Qi': round(vol_qi, 3),
            'Volatility Annualised ARMA-GARCH SPX': round(garch_vol_spx, 3),
            'Volatility Annualised ARMA-GARCH Qi': round(garch_vol_qi, 3),
            'Sharpe Ratio SPX': round(sharpe_spx, 3),
            'Sharpe Ratio Qi': round(sharpe_qi, 3),
            'Sharpe Ratio Qi Probabilistic': round(qi_probabilistic_sharpe, 3),
            'Sortino Ratio SPX': round(sortino_spx, 3),
            'Sortino Ratio Qi': round(sortino_qi, 3),
            'Excess Return': round(ex_return, 3),
            'Tracking Error': round(track_err, 3),
            'Information Ratio': round(info_ratio, 3),
            'Alpha': round(alpha, 3),
            'Alpha percentage error': round(percentage_errors[0], 3),
            'Beta': round(beta, 3),
            'Beta percentage error': round(percentage_errors[1], 3),
            'Alpha p-value': round(alpha_p_value, 3),
            'Alpha-Beta R-squared': round(r_squared, 3),
            'Alpha-Beta BP p-value': round(bp_p_value, 3),
            'Alpha-Beta DW Statistic': round(dw_stat, 3),
            'Sharpe Ratio ARMA SPX': round(arma_sharpe_ratio_spx, 3),
            'Sharpe Ratio ARMA Residuals p-value SPX': round(p_value_spx, 3),
            'Sharpe Ratio ARMA Qi': round(arma_sharpe_ratio_qi, 3),
            'Sharpe Ratio ARMA Residuals p-value Qi': round(p_value_qi, 3),
        }

    # Save results to a CSV file
    results_df = pd.DataFrame(results)
    results_df.to_csv("main_analysis/main_results.csv")
    print("Main analysis results saved to 'main_analysis/main_results.csv'.")

    return results_df


def visualise_raw_data(returns_df):
    """
    Visualise the raw index data over time.

    This function takes a DataFrame of daily returns and plots the raw index data
    over time. The result is a line plot showing the value of both the SPX and Qi
    indices over time.

    Parameters
    ----------
    returns_df : pd.DataFrame
        A DataFrame containing the daily returns of the SPX and Qi indices. The
        DataFrame should have columns 'SPX_Return' and 'Qi_Return'.

    Returns
    -------
    None
    """
    plt.figure(figsize=(12, 6))
    for col in ['SPX', 'Qi']:
        plt.plot(returns_df.index, returns_df[col], label=col, alpha=0.5)
    plt.title('Index Over Time')
    plt.xlabel('Date')
    plt.ylabel('Index value')
    plt.legend()
    save_plot("index_over_time.png")


def visualise_returns(returns_df):
    """
    Visualise the returns of the SPX and Qi indices over time.

    This function takes a DataFrame of daily returns and plots the returns of the
    SPX and Qi indices over time. The result is a line plot showing the returns of
    both the SPX and Qi indices over time.

    Parameters
    ----------
    returns_df : pd.DataFrame
        A DataFrame containing the daily returns of the SPX and Qi indices. The
        DataFrame should have columns 'SPX_Return' and 'Qi_Return'.

    Returns
    -------
    None
    """
    plt.figure(figsize=(12, 6))

    # Make a label map, so legend appears as we want it to
    label_map = {
        'SPX_Return': 'SPX',
        'Qi_Return': 'Qi'
    }

    for col in ['SPX_Return', 'Qi_Return']:
        plt.plot(returns_df.index, returns_df[col], label=label_map[col], alpha=0.5)

    plt.title('Returns Over Time')
    plt.xlabel('Date')
    plt.ylabel('Return')
    plt.legend()
    save_plot("returns_over_time.png")


def stationarity_check(returns_df):
    """
    Perform an Augmented Dickey-Fuller (ADF) test for stationarity on the series.
    
    Parameters
    ----------
    returns_df : pd.DataFrame
        A DataFrame containing the daily returns of the SPX and Qi indices. The
        DataFrame should have columns 'SPX_Return' and 'Qi_Return'.
    
    Returns
    -------
    list of dict
        A list of dictionaries, each containing the results of the ADF test for a
        given series. The dictionaries contain the keys 'Series', 'ADF Statistic',
        'p-value', and 'Stationary'. The last of these is 'Yes' if the p-value is
        less than or equal to 0.05 (i.e., the null hypothesis of non-stationarity
        can be rejected at the 5% level), and 'No' otherwise.
    """
    results = []
    for col in ['SPX_Return', 'Qi_Return']:
        result = adfuller(returns_df[col])
        results.append({
            "Series": col,
            "ADF Statistic": round(result[0], 3),
            "p-value": round(result[1], 3),
            "Stationary": "Yes" if result[1] <= 0.05 else "No"
        })
    return results


def normality_check(returns_df):
    """
    Perform a Kolmogorov-Smirnov test for normality on the series.

    Parameters
    ----------
    returns_df : pd.DataFrame
        A DataFrame containing the daily returns of the SPX and Qi indices. The
        DataFrame should have columns 'SPX_Return' and 'Qi_Return'.

    Returns
    -------
    list of dict
        A list of dictionaries, each containing the results of the Kolmogorov-Smirnov
        test for a given series. The dictionaries contain the keys 'Series',
        'KS Statistic', 'p-value', and 'Normal'. The last of these is 'Yes' if the
        p-value is greater than 0.05 (i.e., the null hypothesis of normality can be
        rejected at the 5% level), and 'No' otherwise.
    """
    results = []
    for col in ['SPX_Return', 'Qi_Return']:
        stat, p_value = kstest(returns_df[col], 'norm')
        results.append({
            "Series": col,
            "KS Statistic": round(stat, 3),
            "p-value": round(p_value, 3),
            "Normal": "Yes" if p_value > 0.05 else "No"
        })
    return results


def heteroskedasticity_check(returns_df):
    """
    Perform a Breusch-Pagan test for heteroskedasticity on SPX and Qi returns.

    This function tests for heteroskedasticity in the daily returns of the SPX
    and Qi indices using the Breusch-Pagan test. It fits a linear regression model
    to each return series and then applies the test to the residuals of the model.
    The function returns the Lagrange Multiplier (LM) statistic and p-value for
    each series, indicating whether heteroskedasticity is detected.

    Parameters
    ----------
    returns_df : pd.DataFrame
        A DataFrame containing the daily returns of the SPX and Qi indices. The
        DataFrame should have columns 'SPX_Return' and 'Qi_Return'.

    Returns
    -------
    list of dict
        A list of dictionaries, each containing the results of the Breusch-Pagan
        test for a given series. The dictionaries contain the keys 'Series', 'LM Statistic',
        'p-value', and 'Heteroskedasticity Detected', which is 'Yes' if the p-value
        is less than or equal to 0.05, and 'No' otherwise.
    """
    results = []
    for col in ['SPX_Return', 'Qi_Return']:
        data = pd.DataFrame({'y': returns_df[col]})
        data['x'] = np.arange(len(returns_df))
        model = sm.OLS(data['y'], sm.add_constant(data['x'])).fit()
        test = het_breuschpagan(model.resid, model.model.exog)
        results.append({
            "Series": col,
            "LM Statistic": round(test[0], 3),
            "p-value": round(test[1], 3),
            "Heteroskedasticity Detected": "Yes" if test[1] <= 0.05 else "No"
        })
    return results


def create_acf_pacf_plots(returns_df):
    """
    Create ACF and PACF plots for each column in the returns DataFrame.

    This function generates and saves Autocorrelation Function (ACF) and 
    Partial Autocorrelation Function (PACF) plots for each return series 
    in the provided DataFrame. The plots are saved as PNG files in the 
    'exploratory_analysis' directory with filenames indicating the column 
    name.

    Parameters
    ----------
    returns_df : pd.DataFrame
        A DataFrame containing the return series to be analyzed. Each column 
        represents a different series.

    Returns
    -------
    None
    """
    for col in returns_df.columns:
        plt.figure(figsize=(12, 6))

        # ACF plot
        plt.subplot(1, 2, 1)
        plot_acf(returns_df[col], lags=20, ax=plt.gca())
        plt.title(f"ACF")
        # The point corresponding to lag 1 gets cut off
        plt.ylim(-1, 1.1)

        # PACF plot
        plt.subplot(1, 2, 2)
        plot_pacf(returns_df[col], lags=20, ax=plt.gca())
        # The point corresponding to lag 1 gets cut off
        plt.ylim(-1, 1.1)
        plt.title(f"PACF")

        plt.tight_layout()

        plt.subplots_adjust(top=1.0)
        file_path = f"exploratory_analysis/acf_pacf_{col}.png"
        plt.savefig(file_path, bbox_inches="tight")
        plt.close()


def calculate_skew_kurtosis(data):
    """
    Calculate the skewness and kurtosis of a given DataFrame of returns.

    Parameters
    ----------
    data : pd.DataFrame
        A DataFrame containing the return series to be analyzed. Each column
        represents a different series.

    Returns
    -------
    pd.DataFrame
        A DataFrame with two columns, 'Skewness' and 'Kurtosis', containing the
        skewness and kurtosis of each column in the input DataFrame.
    """
    skewness = data.skew()
    kurt = data.kurt()
    results_df = pd.DataFrame({'Skewness': skewness, 'Kurtosis': kurt})
    return results_df


def arch_test_results(returns_df, output_path="exploratory_analysis/arch_results.csv"):
    """
    Perform Engle's ARCH test on a given DataFrame of returns.

    Parameters
    ----------
    returns_df : pd.DataFrame
        A DataFrame containing the return series to be analyzed. Each column
        represents a different series.
    output_path : str (optional)
        The path to save the results to. Defaults to
        "exploratory_analysis/arch_results.csv".

    Returns
    -------
    pd.DataFrame
        A DataFrame with the results of the ARCH test for each column in the
        input DataFrame. The columns of the returned DataFrame are:
        - Column: the name of the column in the input DataFrame
        - LM Statistic: the Lagrange multiplier statistic
        - p-value: the p-value of the test
        - F-Statistic: the F-statistic of the test
        - F-Test p-value: the p-value of the F-test
    """
    results = []

    for col in ['SPX_Return', 'Qi_Return']:
        arch_test = het_arch(returns_df[col], nlags=20)
        result = {
            "Column": col,
            "LM Statistic": arch_test[0],
            "p-value": arch_test[1],
            "F-Statistic": arch_test[2],
            "F-Test p-value": arch_test[3]
        }
        results.append(result)

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Save to CSV
    results_df.to_csv(output_path, index=False)

    return results_df


def perform_data_exploration():
    """
    Performs exploratory data analysis on a given DataFrame of returns.

    This function reads in data from a CSV file, visualises the raw data and returns,
    creates ACF and PACF plots, tests for stationarity, normality, and
    heteroskedasticity, performs an ARCH test for heteroskedasticity in volatility,
    calculates skewness and kurtosis, and saves the results to CSV files.

    The results are saved in the "exploratory_analysis" folder, with the following
    filenames:
    - "stationarity_results.csv"
    - "normality_results.csv"
    - "heteroskedasticity_results.csv"
    - "skew_kurtosis_results.csv"
    - "arch_results.csv"

    The function takes no parameters and returns nothing.

    """
    returns_df = read_in_data()

    # Visualise data
    visualise_raw_data(returns_df)
    visualise_returns(returns_df)

    # Create ACF/PACF plots
    create_acf_pacf_plots(returns_df)

    # Test for stationarity
    stationarity_results = stationarity_check(returns_df)

    # Test for normality using Kolmogorov-Smirnov test
    normality_results = normality_check(returns_df)

    # Test for heteroskedasticity using Breusch-Pagan test
    heteroskedasticity_results = heteroskedasticity_check(returns_df)

    # Perform ARCH test for heteroskedasticity in volatility
    arch_results = arch_test_results(returns_df)

    # Calculate skewness and kurtosis
    skew_kurtosis_results = calculate_skew_kurtosis(returns_df)

    # Combine all results into DataFrames
    stationarity_df = pd.DataFrame(stationarity_results)
    normality_df = pd.DataFrame(normality_results)
    heteroskedasticity_df = pd.DataFrame(heteroskedasticity_results)
    # regression_results, skew_kurtosis_results and arch_results is already a DataFrame

    # Save each to a CSV file
    stationarity_df.to_csv("exploratory_analysis/stationarity_results.csv", index=False)
    normality_df.to_csv("exploratory_analysis/normality_results.csv", index=False)
    heteroskedasticity_df.to_csv("exploratory_analysis/heteroskedasticity_results.csv", index=False)
    skew_kurtosis_results.to_csv("exploratory_analysis/skew_kurtosis_results.csv", index=True)
    arch_results.to_csv("exploratory_analysis/arch_results.csv", index=False)

    print("Exploratory data analysis results saved to CSV files in exploratory_analysis folder.")


def granger_causality_test(x_data, y_data, max_lag=20, output_folder="further_analysis"):
    """
    Perform Granger causality test between two time series datasets.

    This function conducts a Granger causality test to determine if one time series
    can predict another. It evaluates multiple lags and returns the significant results.
    The results are plotted and saved as a CSV file and a PNG plot.

    Parameters
    ----------
    x_data : pd.Series
        The first time series data to test for causality.
    y_data : pd.Series
        The second time series data to test for causality.
    max_lag : int, optional
        The maximum number of lags to consider. Default is 20.
    output_folder : str, optional
        The folder path to save the results. Default is "further_analysis".

    Returns
    -------
    dict
        A dictionary where keys are lag values and values are p-values for
        significant Granger causality results (p < 0.05).
    """
    results = grangercausalitytests(
        pd.concat([x_data, y_data], axis=1), maxlag=max_lag, verbose=False
    )

    chisq_p_values = {
        lag: test_result[0]['ssr_chi2test'][1] for lag, test_result in results.items()
    }

    # Get significant results (p < 0.05) before rounding
    significant_results = {lag: p for lag, p in chisq_p_values.items() if p < 0.05}

    rounded_significant_p_values = {lag: round(p, 3) for lag, p in significant_results.items()}

    # Save significant results to CSV
    significant_df = pd.DataFrame(
        list(rounded_significant_p_values.items()), columns=["Lag", "p-value"]
    )
    significant_df.sort_values('Lag', inplace=True)
    output_path_sig = os.path.join(output_folder, "granger_causality_significant_results.csv")
    significant_df.to_csv(output_path_sig, index=False)

    # Plot all p-values
    p_values_df = pd.DataFrame(list(chisq_p_values.items()), columns=["Lag", "p-value"])
    p_values_df.sort_values('Lag', inplace=True)

    plt.figure(figsize=(10, 6))
    plt.plot(p_values_df['Lag'], p_values_df['p-value'], marker='o')
    plt.axhline(y=0.05, color='red', linestyle='--', label='Significance Level (0.05)')
    plt.title('Granger Causality Test P-Values')
    plt.xlabel('Lag')
    plt.ylabel('P-Value')
    plt.xticks(range(1, max_lag + 1))
    plt.legend()
    plot_path = os.path.join(output_folder, "granger_causality_p_values.png")
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()

    return significant_results


def perform_var_impulse_response_analysis(returns_df, output_folder="further_analysis"):
    """
    Perform Vector Autoregression (VAR) analysis and compute the impulse response function (IRF).

    This function performs the following steps:
    1. Compute the Granger causality test to determine the significant lags.
    2. Fit a VAR model to the data using the determined lag order.
    3. Extract the model coefficients and p-values.
    4. Save the coefficients and p-values to a CSV file.
    5. Compute the impulse response function (IRF) of the model.
    6. Plot the IRF and save the plot to a PNG file.

    Parameters
    ----------
    returns_df : pd.DataFrame
        A DataFrame containing the daily returns of the SPX and Qi indices.
    output_folder : str (optional)
        The folder to save the results to. Defaults to "further_analysis".

    Returns
    -------
    irf : pd.Series
        The impulse response function of the model.
    """
    significant_lags = granger_causality_test(returns_df['Qi_Return'], returns_df['SPX_Return'])
    lag_order = max(significant_lags.keys()) if significant_lags else 1

    data = pd.concat([returns_df['Qi_Return'], returns_df['SPX_Return']], axis=1)
    data.columns = ['Qi_Return', 'SPX_Return']
    model = VAR(data)
    results = model.fit(lag_order)

    coeffs = results.params
    p_values = results.pvalues
    combined_data = pd.concat([coeffs, p_values.rename(lambda col: col + "_pvalue", axis=1)], axis=1)
    output_file = f"{output_folder}/var_coefficients_pvalues.csv"
    combined_data.T.to_csv(output_file, index=True)
    print(f"Model coefficients and p-values saved to {output_file}")

    irf = results.irf(10)
    # We are not testing a hypothesis, just examining the data, so set `orth=False`
    fig = irf.plot(orth=False)

    # Change subplot titles
    titles = [
        "Impact of Qi on Qi",
        "Impact of SPX on Qi",
        "Impact of Qi on SPX",
        "Impact of SPX on SPX"
    ]
    for ax, title in zip(fig.axes, titles):
        ax.set_title(title)

    save_plot("var_impulse_response.png", output_folder)

    return irf


def plot_cumulative_returns(returns_df, spx_col='SPX_Return', qi_col='Qi_Return',
                            output_folder="further_analysis", zoom_range=None):
    """
    Plot the cumulative returns of SPX and QI over time.

    Parameters
    ----------
    returns_df : pd.DataFrame
        A DataFrame containing the daily returns of the SPX and Qi indices. The
        DataFrame should have columns 'SPX_Return' and 'Qi_Return'.
    spx_col : str, optional
        The column name of the SPX returns. Defaults to 'SPX_Return'.
    qi_col : str, optional
        The column name of the Qi returns. Defaults to 'Qi_Return'.
    output_folder : str, optional
        The folder in which to save the plot. Defaults to "further_analysis".
    zoom_range : tuple, optional
        A tuple of two integers, representing the start and end dates (inclusive)
        for which to plot the cumulative returns. If provided, the plot will be
        zoomed in on this range. Otherwise, the entire range of returns will be
        plotted.

    Returns
    -------
    None
    """
    cumulative_spx = (1 + returns_df[spx_col]).cumprod() - 1
    cumulative_qi = (1 + returns_df[qi_col]).cumprod() - 1

    # Apply zoom range if provided
    if zoom_range is not None:
        start, end = zoom_range
        cumulative_spx = cumulative_spx.iloc[start:end]
        cumulative_qi = cumulative_qi.iloc[start:end]

    # Plot the cumulative returns
    plt.figure(figsize=(12, 6))
    plt.plot(cumulative_spx, label='Cumulative SPX Returns', color='blue')
    plt.plot(cumulative_qi, label='Cumulative QI Returns', color='orange')
    plt.title(
        'Cumulative Product of Returns of SPX and QI (Zoomed In)' if zoom_range else 'Cumulative Product of Returns of SPX and QI')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()

    # Save the plot
    plot_name = "cumulative_returns_zoomed.png" if zoom_range else "cumulative_returns.png"
    save_plot(plot_name, output_folder)


def volatility_weighted_var(returns, confidence_level=0.95, max_value=10, output_folder="further_analysis"):
    """
    Calculate volatility weighted VaR using the GARCH model.

    Parameters
    ----------
    returns : pd.Series
        A series of daily returns.
    confidence_level : float, optional
        The confidence level for the VaR calculation. Defaults to 0.95.
    max_value : int, optional
        The maximum number of autoregressive and moving average terms to
        consider, by default 10.
    output_folder : str, optional
        The folder in which to save the plot. Defaults to "further_analysis".

    Returns
    -------
    float
        The calculated VaR.

    Notes
    -----
    The VaR is calculated by fitting an ARMA-GARCH model to the data and then
    using the latest conditional volatility to adjust the historical returns.
    The percentile of the adjusted returns is then used to calculate the VaR.
    """
    best_aic = float('inf')
    best_model = None

    if optimise_aic == True:
        # Loop over combinations of p and q
        for p in range(max_value + 1):
            for q in range(max_value + 1):
                for lag in range(max_value + 1):
                    # Otherwise an error is raised `ValueError: One of p or o must be strictly positive`
                    if p == 0:
                        continue
                    if p == 0 and q == 0:
                        continue
                    garch_model = arch_model(returns, mean='AR', lags=lag, vol='Garch', p=p, q=q, dist='normal')
                    fitted_model = garch_model.fit(disp="off")
                    # Check AIC
                    aic = fitted_model.aic
                    if aic < best_aic:
                        best_aic = aic
                        best_model = fitted_model
    else:
        # The value for lag is informed by the rule of thumb T^1/4
        # The values for p and q are informed by the ACDF plot
        garch_model = arch_model(returns, mean='AR', lags=int(len(returns) ** 0.25), vol='Garch', p=14, q=14,
                                 dist='normal')
        best_model = garch_model.fit(disp="off")

    conditional_vols = best_model.conditional_volatility
    latest_vol = conditional_vols.iloc[-1]

    # Adjust historical returns
    vol_adjusted_returns = returns * (latest_vol / conditional_vols)

    # Calculate VaR as the percentile of adjusted returns
    var_vol_weighted = np.percentile(vol_adjusted_returns, (1 - confidence_level) * 100)

    # Make a graph
    plt.figure(figsize=(10, 6))
    plt.hist(returns, bins=100, alpha=0.7)
    plt.axvline(var_vol_weighted, color='r', linestyle='dashed',
                label=f'VaR ({confidence_level}): {var_vol_weighted:.2%}')
    plt.xlabel('Returns')
    plt.ylabel('Frequency')
    plt.title(f'Volatility Weighted VaR for Returns')
    plt.legend()
    save_plot("volatility_weighted_var.png", output_folder)

    return var_vol_weighted


def rolling_analysis(returns_df, rolling_window=252, risk_free_rate=0.02, output_folder="further_analysis"):
    """
    Calculate rolling metrics over the returns of the Qi and SPX portfolios.

    Parameters
    ----------
    returns_df : pd.DataFrame
        A DataFrame containing the daily returns of the Qi and SPX portfolios.
    rolling_window : int, optional
        The rolling window size (in days) for the calculations. Default is 252.
    risk_free_rate : float, optional
        The annual risk-free rate. Default is 0.02.
    output_folder : str, optional
        The folder to save the results to. Default is "further_analysis".

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the rolling metrics, with columns:
            - Rolling_Excess_Return: the excess return of Qi over SPX
            - Rolling_Tracking_Error: the tracking error of Qi to SPX
            - Rolling_Information_Ratio: the information ratio of Qi to SPX
            - Rolling_Sharpe_Qi: the Sharpe ratio of Qi
            - Rolling_Sharpe_SPX: the Sharpe ratio of SPX
    """
    results = pd.DataFrame(index=returns_df.index)

    # Rolling metrics
    results['Rolling_Excess_Return'] = (
            returns_df['Qi_Return'].rolling(rolling_window).mean() * 252 -
            returns_df['SPX_Return'].rolling(rolling_window).mean() * 252
    )
    results['Rolling_Tracking_Error'] = (
            (returns_df['Qi_Return'] - returns_df['SPX_Return'])
            .rolling(rolling_window).std() * np.sqrt(252)
    )
    results['Rolling_Information_Ratio'] = (
            results['Rolling_Excess_Return'] / results['Rolling_Tracking_Error']
    )
    results['Rolling_Sharpe_Qi'] = (
        returns_df['Qi_Return'].rolling(rolling_window).apply(
            lambda x: sharpe_ratio(pd.Series(x), risk_free_rate), raw=False
        )
    )
    results['Rolling_Sharpe_SPX'] = (
        returns_df['SPX_Return'].rolling(rolling_window).apply(
            lambda x: sharpe_ratio(pd.Series(x), risk_free_rate), raw=False
        )
    )
    output_path = os.path.join(output_folder, "rolling_results.csv")
    results.to_csv(output_path)

    return results


def further_analysis():
    """
    Conduct further financial analysis on the SPX and Qi returns.

    This function performs several analyses on the returns data, including:
    - Creating cumulative returns plots for the entire data and a zoomed-in version.
    - Performing VAR impulse response analysis.
    - Computing and saving the Value at Risk (VaR) for SPX and Qi to a CSV file.
    - Calculating rolling metrics such as excess return, tracking error, information ratio, and Sharpe ratios.
    - Plotting and saving figures for the rolling metrics over time.

    The results and plots are saved in the 'further_analysis' directory.
    """
    returns_df = read_in_data()

    # Create cumulative returns plot
    plot_cumulative_returns(returns_df)
    # Make a zoomed in version
    plot_cumulative_returns(returns_df, zoom_range=(1000, 1100))

    # Perform VAR impulse response analysis
    perform_var_impulse_response_analysis(returns_df)

    # Compute Value at Risk (VaR) for SPX and Qi
    value_at_risk_spx = volatility_weighted_var(returns_df['SPX_Return'])
    value_at_risk_qi = volatility_weighted_var(returns_df['Qi_Return'])

    results = {
        'Metric': ['VaR (SPX)', 'VaR (Qi)'],
        'Value': [round(value_at_risk_spx, 3), round(value_at_risk_qi, 3)]
    }
    results_df = pd.DataFrame(results)

    output_folder = "further_analysis"

    output_path = os.path.join(output_folder, "var_results.csv")
    results_df.to_csv(output_path, index=False)

    rolling_metrics = rolling_analysis(returns_df)

    # Plotting rolling metrics
    plt.figure(figsize=(12, 6))
    plt.plot(rolling_metrics.index, rolling_metrics['Rolling_Excess_Return'], label='Rolling Excess Return')
    plt.title('Rolling Excess Return Over Time')
    plt.xlabel('Date')
    plt.ylabel('Excess Return')
    plt.legend()
    save_plot("rolling_excess_return", "further_analysis")

    plt.figure(figsize=(12, 6))
    plt.plot(rolling_metrics.index, rolling_metrics['Rolling_Tracking_Error'], label='Rolling Tracking Error')
    plt.title('Rolling Tracking Error Over Time')
    plt.xlabel('Date')
    plt.ylabel('Tracking Error')
    plt.legend()
    save_plot("rolling_tracking_error", "further_analysis")

    plt.figure(figsize=(12, 6))
    plt.plot(rolling_metrics.index, rolling_metrics['Rolling_Information_Ratio'], label='Rolling Information Ratio')
    plt.title('Rolling Information Ratio Over Time')
    plt.xlabel('Date')
    plt.ylabel('Information Ratio')
    plt.legend()
    save_plot("rolling_information_ratio", "further_analysis")

    plt.figure(figsize=(12, 6))
    plt.plot(rolling_metrics.index, rolling_metrics['Rolling_Sharpe_Qi'], label='Rolling Sharpe Ratio (Qi)')
    plt.plot(rolling_metrics.index, rolling_metrics['Rolling_Sharpe_SPX'], label='Rolling Sharpe Ratio (SPX)')
    plt.title('Rolling Sharpe Ratios Over Time')
    plt.xlabel('Date')
    plt.ylabel('Sharpe Ratio')
    plt.legend()
    save_plot("rolling_sharpe_ratio", "further_analysis")


def install_requirements():
    """
    Installs all packages specified in the requirements.txt file using pip.

    This function will attempt to install all packages listed in the
    requirements.txt file using pip. If any of the installations fail,
    an error message will be displayed. If pip is not found in the
    system's PATH environment, a FileNotFoundError will be raised, and
    a message will be displayed asking the user to ensure pip is
    installed and added to their PATH environment.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    try:
        subprocess.check_call(["pip", "install", "-r", "requirements.txt"])
        print("All requirements installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while installing packages: {e}")
    except FileNotFoundError:
        print("Ensure that pip is installed and added to your PATH environment.")


if __name__ == "__main__":
    install_requirements()
    perform_data_exploration()
    main_analysis()
    further_analysis()
