import itertools
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import kstest
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_breusch_godfrey, het_breuschpagan
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings('ignore')


def read_in_data():
    data = pd.read_csv(
        'SPX_Qi_TimeSeries.csv',
        parse_dates=['Date'],
        index_col='Date'
    )
    data.dropna(inplace=True)
    # Account for the fact that the data may be in the incorrect order
    data.sort_index(inplace=True)
    # The current returns data is not percentage based
    data['SPX_Return'] = data['SPX'].pct_change()
    data['Qi_Return'] = data['Qi'].pct_change()
    # The first row becomes a NaN
    data.dropna(inplace=True)
    return data


def select_returns(data, period):
    end_date = data.index.max()
    if period == '1 year':
        return data.loc[end_date - pd.DateOffset(years=1):end_date]
    elif period == '3 years':
        return data.loc[end_date - pd.DateOffset(years=3):end_date]
    elif period == '5 years':
        return data.loc[end_date - pd.DateOffset(years=5):end_date]
    elif period == 'inception':
        return data

    one_year = data.loc[end_date - pd.DateOffset(years=1):end_date]
    three_years = data.loc[end_date - pd.DateOffset(years=3):end_date]
    five_years = data.loc[end_date - pd.DateOffset(years=5):end_date]
    inception = data
    return one_year, three_years, five_years, inception


def mean_annual_return(returns):
    cumulative_return = (1 + returns).prod()
    annual_return = cumulative_return ** (252 / len(returns)) - 1
    return annual_return


def annualised_volatility(returns):
    return returns.std() * np.sqrt(252)


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
        print("No negative returns")
        return np.nan
    downside_std = negative_returns.std() * np.sqrt(252)
    # We need to deal with the case where the downside standard deviation is effectively zero
    if downside_std < 1e-8:
        print("Downside standard deviation is effectively zero")
        return np.nan
    print("Mean return:", mean_return, "Risk free rate:", risk_free_rate,"Downside standard deviation:", downside_std)
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
    print(f"Ljung-Box test p-value: {p_value}")
    if p_value > 0.05:
        print("Residuals appear to be white noise.")
    else:
        print("Residuals may not be white noise. Consider a different model.")


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


def simulate_returns(model, n_simulations=2000):
    # We use the ARMA model to simulate returns here
    simulated_returns = model.simulate(nsimulations=n_simulations)
    return simulated_returns


# def calculate_sortino_ratio_empirical(simulated_returns, risk_free_rate=0.02):
#     # Here, we calculate the Sortino ratio from the distribution created with the ARMA model
#     expected_return = np.mean(simulated_returns)
#     negative_returns = simulated_returns[simulated_returns < 0]
#     if len(negative_returns) == 0:
#         return np.nan
#     downside_std = negative_returns.std() * np.sqrt(252)
#     # We need to deal with the case, where the downside standard deviation is effectively zero
#     if downside_std < 1e-8:
#         return np.nan
#     sortino_ratio = (expected_return - risk_free_rate) / downside_std
#     return sortino_ratio


def excess_return(returns_qi, returns_spx):
    mean_return_qi = returns_qi.mean() * 252
    mean_return_spx = returns_spx.mean() * 252
    return mean_return_qi - mean_return_spx


def tracking_error(returns_qi, returns_spx):
    difference = returns_qi - returns_spx
    return difference.std() * np.sqrt(252)


def information_ratio(calculated_excess_return, calculated_tracking_err):
    return calculated_excess_return / calculated_tracking_err


def regression_analysis(returns_qi, returns_spx, risk_free_rate=0.02, maxlags=5):
    risk_free_rate_daily = risk_free_rate / 252
    # Could show a plot of this
    data = pd.concat([returns_qi, returns_spx], axis=1)
    # Calculate excess returns
    data['Excess_Qi_Return'] = data['Qi_Return'] - risk_free_rate_daily
    data['Excess_SPX_Return'] = data['SPX_Return'] - risk_free_rate_daily

    X = data['Excess_SPX_Return']
    y = data['Excess_Qi_Return']
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()#(cov_type='HAC', cov_kwds={'maxlags': maxlags})

    alpha_daily = model.params['const']
    alpha = (1 + alpha_daily) ** 252 - 1

    beta = model.params['Excess_SPX_Return']
    r_squared = model.rsquared

    coefficients = model.params
    errors = model.bse
    percentage_errors = []

    for coef_name, coef_value, error in zip(coefficients.index, coefficients, errors):
        percentage_error = (error / abs(coef_value)) * 100
        percentage_errors.append(percentage_error)

    alpha_p_value = model.pvalues['const']

    return alpha, beta, r_squared, percentage_errors, alpha_p_value


def main():
    data = read_in_data()
    end_date = data.index.max()
    one_year = data.loc[end_date - pd.DateOffset(years=1):end_date]
    three_years = data.loc[end_date - pd.DateOffset(years=3):end_date]
    five_years = data.loc[end_date - pd.DateOffset(years=5):end_date]
    inception = data

    # periods = {
    #     '1 Year': one_year,
    #     '3 Years': three_years,
    #     '5 Years': five_years,
    #     'Inception': inception
    # }

    periods = {
        '1 Year': one_year
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

        # Sharpe Ratio (Assuming risk-free rate is 2%)
        sharpe_spx = sharpe_ratio(spx_returns)
        sharpe_qi = sharpe_ratio(qi_returns)

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
        alpha, beta, r_squared, percentage_errors, alpha_p_value = regression_analysis(qi_returns, spx_returns)

        # ARMA Model Selection and Residual Analysis for Qi_Return
        best_order_qi, best_model, arma_results_df = select_best_arma_model(qi_returns)
        print(f"Best ARMA order for {period_name}: {best_order_qi}")

        # Check residuals of the best ARMA model
        check_residuals(best_model)

        # Calculate Sharpe Ratio from ARMA model
        arma_sharpe_ratio_qi = calculate_sharpe_from_arma_model(best_model)

        # # Simulate returns from the ARMA model and calculate Sortino Ratio
        # simulated_returns = simulate_returns(best_model)
        # sortino_ratio_arma_qi = calculate_sortino_ratio_empirical(simulated_returns)

        # ARMA Model Selection and Residual Analysis for SPX_Return
        best_order_spx, best_model, arma_results_df = select_best_arma_model(spx_returns)
        print(f"Best ARMA order for {period_name}: {best_order_spx}")

        # Check residuals of the best ARMA model
        check_residuals(best_model)

        # Calculate Sharpe Ratio from ARMA model
        arma_sharpe_ratio_spx = calculate_sharpe_from_arma_model(best_model)

        # # Simulate returns from the ARMA model and calculate Sortino Ratio
        # simulated_returns = simulate_returns(best_model)
        # sortino_ratio_arma_spx = calculate_sortino_ratio_empirical(simulated_returns)

        results[period_name] = {
            'Mean Annual Return SPX': round(mean_return_spx, 3),
            'Mean Annual Return Qi': round(mean_return_qi, 3),
            'Annualised Volatility SPX': round(vol_spx, 3),
            'Annualised Volatility Qi': round(vol_qi, 3),
            'Sharpe Ratio SPX': round(sharpe_spx, 3),
            'Sharpe Ratio Qi': round(sharpe_qi, 3),
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
            'R-squared': round(r_squared, 3),
            'ARMA Best Order Qi': best_order_qi,
            'ARMA Sharpe Ratio Qi': round(arma_sharpe_ratio_qi, 3),
            # 'ARMA Sortino Ratio Qi': round(sortino_ratio_arma_qi, 3),
            'ARMA Best Order SPX': best_order_spx,
            'ARMA Sharpe Ratio SPX': round(arma_sharpe_ratio_spx, 3),
            # 'ARMA Sortino Ratio SPX': round(sortino_ratio_arma_spx, 3)
        }

    return results


# main()

def perform_data_exploration():
    # Read in data
    returns_df = read_in_data()

    # Visualise data
    visualise_returns(returns_df)

    # Test for stationarity
    print("Testing for Stationarity:")
    stationarity_check(returns_df)

    # Test for normality using Kolmogorov-Smirnov test
    print("\nTesting for Normality:")
    normality_check(returns_df)

    # Test for heteroskedasticity using Breusch-Pagan test
    print("\nTesting for Heteroskedasticity:")
    heteroskedasticity_check(returns_df)

    # Validate regression assumptions
    print("\nValidating Regression Assumptions:")
    validate_regression_assumptions(returns_df)


def visualise_returns(returns_df):
    plt.figure(figsize=(12, 6))
    for col in ['SPX_Return', 'Qi_Return']:
        plt.plot(returns_df.index, returns_df[col], label=col, alpha=0.5)
    plt.title('Returns Over Time')
    plt.xlabel('Date')
    plt.ylabel('Return')
    plt.legend()
    plt.show()


def stationarity_check(returns_df):
    for col in ['SPX_Return', 'Qi_Return']:
        result = adfuller(returns_df[col].dropna())
        print(f"ADF Test for {col}:")
        print(f"  ADF Statistic: {result[0]:.3f}")
        print(f"  p-value: {result[1]:.3f}")
        if result[1] <= 0.05:
            print(f"  {col} is stationary.")
        else:
            print(f"  {col} is not stationary.")


def normality_check(returns_df):
    for col in ['SPX_Return', 'Qi_Return']:
        stat, p_value = kstest(returns_df[col].dropna(), 'norm')
        print(f"KS Test for {col}:")
        print(f"  KS Statistic: {stat:.3f}")
        print(f"  p-value: {p_value:.3f}")
        if p_value > 0.05:
            print(f"  {col} looks normal (fail to reject H0).")
        else:
            print(f"  {col} does not look normal (reject H0).")


def heteroskedasticity_check(returns_df):
    for col in ['SPX_Return', 'Qi_Return']:
        data = pd.DataFrame({'y': returns_df[col]})
        data['x'] = np.arange(len(returns_df))
        model = sm.OLS(data['y'], sm.add_constant(data['x'])).fit()
        test = het_breuschpagan(model.resid, model.model.exog)
        print(f"Breusch-Pagan Test for {col}:")
        print(f"  LM Statistic: {test[0]:.3f}, p-value: {test[1]:.3f}")
        if test[1] > 0.05:
            print(f"  No heteroskedasticity detected in {col}.")
        else:
            print(f"  Heteroskedasticity detected in {col}.")


def validate_regression_assumptions(returns_df):
    data = returns_df.dropna()
    X = sm.add_constant(data['SPX_Return'])
    y = data['Qi_Return']
    model = sm.OLS(y, X).fit()
    print(model.summary())

    residuals = model.resid

    # Test residuals for autocorrelation and heteroskedasticity
    dw_stat = sm.stats.durbin_watson(residuals)
    print(f"\nDurbin-Watson Statistic: {dw_stat:.3f}")
    bg_test = acorr_breusch_godfrey(model, nlags=5)
    print(f"Breusch-Godfrey Test p-value: {bg_test[3]:.3f}")

    # Test residuals for normality
    stat, p_value = kstest(residuals, 'norm')
    print(f"KS Test for Residuals:")
    print(f"  KS Statistic: {stat:.3f}, p-value: {p_value:.3f}")

    # Plot residuals
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, residuals)
    plt.title('Residuals from Regression')
    plt.xlabel('Date')
    plt.ylabel('Residual')
    plt.show()

    # Plot ACF and PACF of residuals
    plt.figure(figsize=(12, 6))
    plot_acf(residuals, lags=20, title='ACF of Residuals')
    plot_pacf(residuals, lags=20, title='PACF of Residuals')
    plt.show()


# if __name__ == "__main__":
#     SPX_Qi_TimeSeries = pd.read_csv("baskets/SPX_Qi_TimeSeries.csv")
#     pass
# perform_data_exploration()
print(main())