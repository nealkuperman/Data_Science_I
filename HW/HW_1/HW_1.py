#%% Packages
import ISLP 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as smf
from lin_reg_plots import LinearRegDiagnostic

# %%
def calc_leverage(x):
    """
    Observations with high leverage have an unusual value for xi. High leverage observations tend to have
    a sizable impact on the estimated regression line.   

    SLR
    hi = 1/n + (xi - x_bar)^2 / sum((xi - x_bar)^2)

    MLR

    Hii = diag(H) 
        H = X^T (X^T X)^-1 X
    """

    H = x @ np.linalg.inv(x.T @ x) @ x.T
    leverage_points = np.diag(H)
    return leverage_points

def find_high_leverage_points(x):
    """
    Observations with high leverage have an unusual value for xi. High leverage observations tend to have

    hii > 2p/n
        p = number of predictors
        n = number of observations
    """
    leverage_points = calc_leverage(x)
    high_leverage_points = leverage_points > 2 * x.shape[1]/x.shape[0]
    index = np.where(high_leverage_points)[0]
    return index, leverage_points[index]

def calc_standard_error(x, y):
    """
    Standard error of the estimated regression coefficients.

    SE(beta_j) = sqrt(MSE / (n - p - 1))
        MSE = mean squared error
        n = number of observations
        p = number of predictors
    """


def calc_cooks_distance(x, y):
    """
    Cook's distance is a measure of the influence of an observation on the estimated regression coefficients.
    
    D_i = (e_i^2 / (p * MSE)) * (h_ii / (1 - h_ii)^2)
    
    where:
        e_i = residual (y_i - y_hat_i)
        h_ii = leverage (diagonal of hat matrix)
        p = number of predictors (including intercept)
        MSE = mean squared error = SS_res / (n - p)
    """
    n, p = x.shape
    
    # Fit model
    beta = np.linalg.inv(x.T @ x) @ x.T @ y
    y_hat = x @ beta
    residuals = y - y_hat
    
    # MSE
    SS_res = np.sum(residuals**2)
    MSE = SS_res / (n - p)
    
    # Leverage
    h = calc_leverage(x)
    
    # Cook's distance
    cooks_d = (residuals**2 / (p * MSE)) * (h / (1 - h)**2)
    
    return cooks_d

rng = np.random.default_rng(10)
x1 = rng.uniform(0, 1, size=100)
x2 = 0.5 * x1 + rng.normal(size=100) / 10
y = 2 + 2 * x1 + 0.3 * x2 + rng.normal(size=100)

x1 = np.concatenate([x1, [0.1]])
x2 = np.concatenate([x2, [0.8]])
X = np.column_stack((np.ones(len(x1)), x1, x2))

y = np.concatenate([y, [6]])

leverage_points = calc_leverage(X)
print(leverage_points)

index, high_leverage_points = find_high_leverage_points(X)
print(index)
print(high_leverage_points)


# %%
