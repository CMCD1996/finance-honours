# Imports useful python packages
# Essential
import numpy as np
from numpy.core.fromnumeric import transpose # Arithmetic operations
import pandas as pd # Data analysis package
import matplotlib.pyplot as plt # Simple plotting
import sklearn as skl # Simple statistical models 
import tensorflow as tf # Tensorflow (https://www.tensorflow.org/)
import csv as csv # read and write csvs
# Additional
import random as rd # random functionality
import saspy as sas # Use saspy functionality in python
import seaborn as sb # Imports seaborn library for use
import wrds as wrds# Wharton Research Data Services API
import pydatastream as pds # Thomas Reuters Datastream API
import yfinance as yf # Yahoo Finance API
import datetime as dt # Manipulate datetime values
import statsmodels.api as sm # Create Stats functionalities
import linearmodels as lp # Ability to use PooledOLS
from sklearn.linear_model import LinearRegression
from stargazer.stargazer import Stargazer
import finance_byu as fin # Python Package for Fama-MacBeth Regressions
from statsmodels.regression.rolling import RollingOLS # Use factor loadings
from stargazer.stargazer import Stargazer
import sympy as sy # convert latex code
import scipy as sc # Scipy packages
import tabulate as tb # Create tables in python
import itertools as it # Find combinations of lists

# Writes functions


def ranking_function(type):
    """ Ranking function to produce charts for demonstration purposes

    Args:
        type ([type]): String for desired ranking functions
    """
    if type == 'linear':
        # Creates an ordered, random array of proxy returns (%)
        num = 100
        returns_uniform =np.sort(np.arange(-10,10,0.2))
        print('returns',returns_uniform)
        print('returns size',np.size(returns_uniform))
        returns = np.sort(np.random.uniform(low=-10.0, high=10.0, size=(num,)))
        base = np.zeros(num)
        ones = np.ones(num)
        # Creates rank array
        rank = np.array(list(range(1,len(returns)+ 1)))
        # Create weights
        weights  = returns/transpose(ones)
        print('weights',weights)
        print('Sum of weights', np.sum(weights))
        weights  = weights*returns
        print('weights',weights)
        print('Sum of weights', np.sum(weights))
        # Fit monotonic functions between curves

        # Plots the functions
        plt.plot(rank,returns, 'r.', rank,base, 'k.')
        plt.legend('Returns','Baseline')
        plt.xlabel('Rank')
        plt.ylabel('Return (%)')
        plt.title('Ranking: Monotonic Functions')
        plt.savefig('results/plots/monotonic-ranking.png')
    return

# Implements project script
ranking_function('linear')

