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
    filepath = os.path.join(folder, filename)
    plt.savefig(filepath, bbox_inches="tight")
    plt.close()  # Close the plot to prevent it from showing inline


def read_in_data():
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
    cumulative_return = (1 + returns).prod()
    annual_return = cumulative_return ** (252 / len(returns)) - 1
    return annual_return


def annualised_volatility(returns):
    return returns.std() * np.sqrt(252)


def annualised_volatility_garch(returns, max_value=10):
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
    mean_return = returns.mean() * 252
    volatility = returns.std() * np.sqrt(252)
    # We need to deal with the case where the volatility is effectively zero
    if volatility < 1e-8:
        return np.nan
    return (mean_return - risk_free_rate) / volatility


def sortino_ratio(returns, risk_free_rate=0.02):
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
    residuals = model.resid
    lb_test = acorr_ljungbox(residuals, lags=[lags], return_df=True)
    p_value = lb_test['lb_pvalue'].values[0]
    return p_value


def calculate_sharpe_from_arma_model(model, risk_free_rate=0.02):
    mean_return = model.params.get('const', 0)
    mean_return_annualised = mean_return * 252

    sigma = np.sqrt(model.model.endog.var())
    sigma_annualised = sigma * np.sqrt(252)

    # Excess return
    excess_return = mean_return_annualised - risk_free_rate

    # Sharpe Ratio
    sharpe_ratio = excess_return / sigma_annualised

    return sharpe_ratio


# Calculate Standard Deviation of Sharpe Ratio (Std Dev SR)
def standard_deviation_sharpe_ratio(calculated_sharpe_ratio, num_obs, skewness, kurtosis):
    return np.sqrt(
        (1 - skewness * calculated_sharpe_ratio +
         (kurtosis - 1) / 4 * calculated_sharpe_ratio ** 2
         ) / (num_obs - 1)
    )


def probabilistic_sharpe_ratio(calculated_sharpe_ratio, bench_sharpe_ratio, num_obs, skewness, kurtosis):
    sr_diff = calculated_sharpe_ratio - bench_sharpe_ratio
    sr_vol = standard_deviation_sharpe_ratio(calculated_sharpe_ratio, num_obs, skewness, kurtosis)
    psr = norm.cdf(sr_diff / sr_vol)

    return psr


def excess_return(returns_qi, returns_spx):
    mean_return_qi = returns_qi.mean() * 252
    mean_return_spx = returns_spx.mean() * 252
    return mean_return_qi - mean_return_spx


def tracking_error(returns_qi, returns_spx):
    difference = returns_qi - returns_spx
    return difference.std() * np.sqrt(252)


def information_ratio(calculated_excess_return, calculated_tracking_err):
    return calculated_excess_return / calculated_tracking_err


def regression_analysis(returns_qi, returns_spx, risk_free_rate=0.02):
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
    plt.figure(figsize=(12, 6))
    for col in ['SPX', 'Qi']:
        plt.plot(returns_df.index, returns_df[col], label=col, alpha=0.5)
    plt.title('Indices Over Time')
    plt.xlabel('Date')
    plt.ylabel('Index value')
    plt.legend()
    save_plot("raw_returns_over_time.png")


def visualise_returns(returns_df):
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
    skewness = data.skew()
    kurt = data.kurt()
    results_df = pd.DataFrame({'Skewness': skewness, 'Kurtosis': kurt})
    return results_df


def arch_test_results(returns_df, output_path="exploratory_analysis/arch_results.csv"):
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
    # Read in data
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
    # Perform Granger causality test to determine significant lags
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


def plot_cumulative_returns(returns_df, spx_col='SPX_Return', qi_col='Qi_Return', output_folder="further_analysis"):
    # Calculate cumulative returns
    cumulative_spx = (1 + returns_df[spx_col]).cumprod() - 1
    cumulative_qi = (1 + returns_df[qi_col]).cumprod() - 1

    # Plot the cumulative returns
    plt.figure(figsize=(12, 6))
    plt.plot(cumulative_spx, label='Cumulative SPX Returns', color='blue')
    plt.plot(cumulative_qi, label='Cumulative QI Returns', color='orange')
    plt.title('Cumulative Product of Returns of SPX and QI')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    save_plot("cumulative_returns.png", output_folder)


def volatility_weighted_var(returns, confidence_level=0.95, max_value=10, output_folder="further_analysis"):
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
    # Read in returns data
    returns_df = read_in_data()

    # Create cumulative returns plot
    plot_cumulative_returns(returns_df)

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
