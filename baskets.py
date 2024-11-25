import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import anderson, jarque_bera, normaltest, probplot, shapiro
from statsmodels.formula.api import ols
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_breusch_godfrey, het_arch, het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
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


def sharpe_ratio(returns, risk_free_rate=2):
    mean_return = returns.mean() * 252
    volatility = returns.std() * np.sqrt(252)
    # We need to deal with the case where the volatility is effectively zero
    if volatility < 1e-8:
        return np.nan
    return (mean_return - risk_free_rate) / volatility


def sortino_ratio(returns, risk_free_rate=2):
    mean_return = returns.mean() * 252
    negative_returns = returns[returns < 0]
    downside_std = negative_returns.std() * np.sqrt(252)
    # We need to deal with the case, where the downside standard deviation is effectively zero
    if downside_std < 1e-8:
        return np.nan
    return (mean_return - risk_free_rate) / downside_std


def excess_return(returns_qi, returns_spx):
    mean_return_qi = returns_qi.mean() * 252
    mean_return_spx = returns_spx.mean() * 252
    return mean_return_qi - mean_return_spx


def tracking_error(returns_qi, returns_spx):
    difference = returns_qi - returns_spx
    return difference.std() * np.sqrt(252)


def information_ratio(calculated_excess_return, calculated_tracking_err):
    return calculated_excess_return / calculated_tracking_err


def regression_analysis(returns_qi, returns_spx):
    # Could show a plot of this
    data = pd.concat([returns_qi, returns_spx], axis=1)
    X = data['SPX_Return']
    y = data['Qi_Return']
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    # Should alpha be annualised?
    alpha_daily = model.params['const']
    alpha = (1 + alpha_daily) ** 252 - 1

    beta = model.params['SPX_Return']
    r_squared = model.rsquared
    return alpha, beta, r_squared


def main():
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

        # Annualized Volatility
        vol_spx = annualised_volatility(spx_returns)
        vol_qi = annualised_volatility(qi_returns)

        # Sharpe Ratio (Assuming risk-free rate is 0)
        sharpe_spx = sharpe_ratio(mean_return_spx, vol_spx)
        sharpe_qi = sharpe_ratio(mean_return_qi, vol_qi)

        # Sortino Ratio
        sortino_spx = sortino_ratio(spx_returns, mean_return_spx)
        sortino_qi = sortino_ratio(qi_returns, mean_return_qi)

        # Excess Return
        ex_return = excess_return(mean_return_qi, mean_return_spx)

        # Tracking Error
        track_err = tracking_error(qi_returns, spx_returns)

        # Information Ratio
        info_ratio = information_ratio(ex_return, track_err)

        # Regression Analysis
        alpha, beta, r_squared = regression_analysis(qi_returns, spx_returns)

        # Store Results
        results[period_name] = {
            'Mean Annual Return SPX': mean_return_spx,
            'Mean Annual Return Qi': mean_return_qi,
            'Annualized Volatility SPX': vol_spx,
            'Annualized Volatility Qi': vol_qi,
            'Sharpe Ratio SPX': sharpe_spx,
            'Sharpe Ratio Qi': sharpe_qi,
            'Sortino Ratio SPX': sortino_spx,
            'Sortino Ratio Qi': sortino_qi,
            'Excess Return': ex_return,
            'Tracking Error': track_err,
            'Information Ratio': info_ratio,
            'Alpha': alpha,
            'Beta': beta,
            'R-squared': r_squared
        }

    return results


# main()

def perform_eda():
    """
    Perform Exploratory Data Analysis (EDA) and statistical tests on returns data.

    Parameters:
    - returns_df: pandas DataFrame with columns ['SPX_Return', 'Qi_Return']
    """
    returns_df = read_in_data()
    # 1. Visualize data with time series plots
    plt.figure(figsize=(14, 6))
    plt.plot(returns_df.index, returns_df['SPX_Return'], label='SPX_Return')
    plt.plot(returns_df.index, returns_df['Qi_Return'], label='Qi_Return')
    plt.title('Daily Returns Over Time')
    plt.xlabel('Date')
    plt.ylabel('Return')
    plt.legend()
    plt.show()

    # 2. Identify anomalies or outliers using boxplots
    plt.figure(figsize=(10, 4))
    sns.boxplot(data=returns_df[['SPX_Return', 'Qi_Return']])
    plt.title('Boxplots of Returns')
    plt.show()

    # 3. Test for stationarity using ADF test
    print("ADF Test for SPX_Return:")
    adf_test(returns_df['SPX_Return'])
    print("\nADF Test for Qi_Return:")
    adf_test(returns_df['Qi_Return'])

    # 4. Assess normality
    for col in ['SPX_Return', 'Qi_Return']:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        sns.histplot(returns_df[col], kde=True)
        plt.title(f'Histogram of {col}')
        plt.subplot(1, 2, 2)
        probplot(returns_df[col].dropna(), dist="norm", plot=plt)
        plt.title(f'Q-Q Plot of {col}')
        plt.show()

        # Conduct normality tests
        print(f"Normality Tests for {col}:")
        normality_tests(returns_df[col])
        print("\n")

    # 5. Check for autocorrelation and heteroskedasticity
    for col in ['SPX_Return', 'Qi_Return']:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plot_acf(returns_df[col].dropna(), ax=plt.gca(), lags=20)
        plt.title(f'ACF of {col}')
        plt.subplot(1, 2, 2)
        plot_pacf(returns_df[col].dropna(), ax=plt.gca(), lags=20)
        plt.title(f'PACF of {col}')
        plt.show()

        print(f"Heteroskedasticity Tests for {col}:")
        heteroskedasticity_tests(returns_df[col])
        print("\n")

    # 6. Validate Regression Assumptions
    # Regress Qi_Return on SPX_Return
    data = returns_df.dropna()
    X = sm.add_constant(data['SPX_Return'])
    y = data['Qi_Return']
    model = sm.OLS(y, X).fit()
    print(model.summary())

    residuals = model.resid
    # Plot residuals
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, residuals)
    plt.title('Residuals from Regression')
    plt.xlabel('Date')
    plt.ylabel('Residual')
    plt.show()

    # Plot histogram and Q-Q plot of residuals
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(residuals, kde=True)
    plt.title('Histogram of Residuals')
    plt.subplot(1, 2, 2)
    probplot(residuals, dist="norm", plot=plt)
    plt.title('Q-Q Plot of Residuals')
    plt.show()

    # Normality test on residuals
    print("Normality Tests for Residuals:")
    normality_tests(residuals)

    # Autocorrelation of residuals
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plot_acf(residuals, ax=plt.gca(), lags=20)
    plt.title('ACF of Residuals')
    plt.subplot(1, 2, 2)
    plot_pacf(residuals, ax=plt.gca(), lags=20)
    plt.title('PACF of Residuals')
    plt.show()

    # Durbin-Watson test
    dw_stat = durbin_watson(residuals)
    print(f'Durbin-Watson statistic: {dw_stat}')

    # Breusch-Godfrey test for autocorrelation
    bg_test = acorr_breusch_godfrey(model, nlags=5)
    print(f'Breusch-Godfrey test p-value: {bg_test[3]}')

    # Heteroskedasticity tests on residuals
    print("Heteroskedasticity Tests on Residuals:")
    heteroskedasticity_tests(residuals)


# Helper functions
def adf_test(series):
    result = adfuller(series.dropna())
    labels = ['ADF Statistic', 'p-value', '# Lags Used', '# Observations']
    out = pd.Series(result[0:4], index=labels)
    for key, value in result[4].items():
        out[f'Critical Value ({key})'] = value
    print(out)
    if result[1] <= 0.05:
        print("The series is stationary.")
    else:
        print("The series is non-stationary.")


def normality_tests(series):
    # Shapiro-Wilk Test
    stat, p = shapiro(series)
    print(f'Shapiro-Wilk Test: Statistics={stat:.3f}, p={p:.3f}')
    # D'Agostino's K^2 Test
    stat, p = normaltest(series)
    print(f'D\'Agostino\'s K^2 Test: Statistics={stat:.3f}, p={p:.3f}')
    # Jarque-Bera Test
    stat, p = jarque_bera(series)
    print(f'Jarque-Bera Test: Statistics={stat:.3f}, p={p:.3f}')
    # Anderson-Darling Test
    result = anderson(series)
    print('Anderson-Darling Test:')
    print(f'Statistic: {result.statistic:.3f}')
    for i in range(len(result.critical_values)):
        sl, cv = result.significance_level[i], result.critical_values[i]
        if result.statistic < cv:
            print(f'Significance Level {sl}%, Critical Value {cv:.3f}: Data looks normal (Fail to Reject H0)')
        else:
            print(f'Significance Level {sl}%, Critical Value {cv:.3f}: Data does not look normal (Reject H0)')


def heteroskedasticity_tests(series):
    # Create a DataFrame for the test
    data = pd.DataFrame({'y': series})
    data['x'] = np.arange(len(series))
    model = ols('y ~ x', data=data).fit()
    # Breusch-Pagan test
    bp_test = het_breuschpagan(model.resid, model.model.exog)
    labels = ['LM Statistic', 'LM-Test p-value', 'F-Statistic', 'F-Test p-value']
    bp_results = dict(zip(labels, bp_test))
    print('Breusch-Pagan Test:')
    for key, value in bp_results.items():
        print(f'{key}: {value}')
    # Engle's ARCH test
    arch_test = het_arch(series)
    arch_results = dict(zip(labels, arch_test))
    print('Engle\'s ARCH Test:')
    for key, value in arch_results.items():
        print(f'{key}: {value}')

# if __name__ == "__main__":
#     SPX_Qi_TimeSeries = pd.read_csv("baskets/SPX_Qi_TimeSeries.csv")
#     pass
perform_eda()