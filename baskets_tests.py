import pytest
import numpy as np
import pandas as pd
from baskets import (
    mean_annual_return,
    annualised_volatility,
    sharpe_ratio,
    sortino_ratio,
    excess_return,
    tracking_error,
    information_ratio,
)

def test_mean_annual_return():
    # Test with a constant daily return
    returns = pd.Series([0.001] * 252)  # 0.1% daily return for one year
    expected_return = (1 + 0.001) ** 252 - 1
    calculated_return = mean_annual_return(returns)
    assert np.isclose(calculated_return, expected_return, atol=1e-6)

def test_annualised_volatility():
    # Test with zero volatility
    returns = pd.Series([0.0] * 252)
    expected_volatility = 0.0
    calculated_volatility = annualised_volatility(returns)
    assert calculated_volatility == expected_volatility

    # Test with known volatility
    returns = pd.Series(np.random.normal(0, 0.01, 252))
    expected_volatility = returns.std() * np.sqrt(252)
    calculated_volatility = annualised_volatility(returns)
    assert np.isclose(calculated_volatility, expected_volatility, atol=1e-6)

def test_sharpe_ratio():
    returns = pd.Series([0.001] * 252)
    risk_free_rate = 0.0
    mean_return = returns.mean() * 252
    volatility = returns.std() * np.sqrt(252)
    expected_sharpe = (mean_return - risk_free_rate) / volatility
    calculated_sharpe = sharpe_ratio(returns, risk_free_rate)
    # Since volatility is zero, Sharpe ratio should be infinite
    assert np.isnan(calculated_sharpe)

    # Test with varying returns
    returns = pd.Series(np.random.normal(0.001, 0.02, 252))
    calculated_sharpe = sharpe_ratio(returns, risk_free_rate)
    assert not np.isnan(calculated_sharpe)

def test_sortino_ratio_no_negative_returns():
    returns = pd.Series([0.001] * 252)
    risk_free_rate = 0.0
    # Since there are no negative returns, downside deviation is zero
    calculated_sortino = sortino_ratio(returns, risk_free_rate)
    assert np.isnan(calculated_sortino)

def test_sortino_ratio_with_negative_returns():
    returns = pd.Series([0.001] * 126 + np.random.normal(-0.002, 0.001, 126).tolist())
    risk_free_rate = 0.0
    mean_return = returns.mean() * 252
    negative_returns = returns[returns < 0]
    downside_std = negative_returns.std() * np.sqrt(252)
    expected_sortino = (mean_return - risk_free_rate) / downside_std
    calculated_sortino = sortino_ratio(returns, risk_free_rate)
    assert np.isclose(calculated_sortino, expected_sortino, atol=1e-6)

def test_excess_return():
    returns_qi = pd.Series([0.0015] * 252)
    returns_spx = pd.Series([0.001] * 252)
    expected_excess = (returns_qi.mean() - returns_spx.mean()) * 252
    calculated_excess = excess_return(returns_qi, returns_spx)
    assert np.isclose(calculated_excess, expected_excess, atol=1e-6)

def test_tracking_error():
    returns_qi = pd.Series([0.0015] * 252)
    returns_spx = pd.Series([0.001] * 252)
    difference = returns_qi - returns_spx
    expected_tracking_error = difference.std() * np.sqrt(252)
    calculated_tracking_error = tracking_error(returns_qi, returns_spx)
    assert np.isclose(calculated_tracking_error, expected_tracking_error, atol=1e-6)

def test_information_ratio():
    annualised_excess_return = 0.05
    tracking_error = 0.10
    expected_ir = annualised_excess_return / tracking_error
    calculated_ir = information_ratio(annualised_excess_return, tracking_error)
    assert np.isclose(calculated_ir, expected_ir, atol=1e-6)
