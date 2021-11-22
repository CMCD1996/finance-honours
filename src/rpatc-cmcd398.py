# Imports useful python packages
# Analytical
import sympy as sym # Symbolic package for calculus
# Essential
import numpy as np
from numpy.core.fromnumeric import transpose # Arithmetic operations
import pandas as pd # Data analysis package
import dask as ds # Data importing for very large software packages.
import matplotlib.pyplot as plt # Simple plotting
import sklearn as skl # Simple statistical models 
import tensorflow as tf # Tensorflow (https://www.tensorflow.org/)
import csv as csv # read and write csvs
import os # change/manipulate operating systems
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

# Functions to prepare and inspect data
def convert_data(data_location, data_destination):
    """ Converts dta  format to a series of 100k line csvs

    Args:
        data_location (str): directory of dta file
        data_destination (str): 
    """
    # Converts dta file to chunks
    dflocation = data_destination
    data = pd.read_stata(data_location, chunksize=100000)
    num = 1
    for chunk in data:
        # Saves chunck to seperate csvs given dataset size
        df = pd.DataFrame()
        df = df.append(chunk)
        df.to_csv(dflocation + str(num) +'.csv')
        num_convert = num*100000
        print('Number of rows converted: ',num_convert)
        num = num + 1

def process_data(csv_location):
    """ Function to create 
    """
    '/Users/connor/Google Drive/Documents/University/Courses/2020-21/Finance 788/finance-honours/data/dataframe-columns.txt'
    # Get test loading one dataframe
    num_csvs = list(range(1,29,1))
    print(num_csvs)
    df = pd.read_csv(csv_location + "1.csv")
    # Show frame information
    show_info = False
    if show_info == True: 
        # Prints df head, info, columns
        print('information on Dataframe')
        print(df.info())
        print('Dataframe Head')
        print(df.head())
        print('Dataframe Columns')
        print(df.columns)
        # Saves columns as list in txt file
        np.savetxt(r'/Users/connor/Google Drive/Documents/University/Courses/2020-21/Finance 788/finance-honours/data/dataframe-columns.txt', df.columns, fmt='%s')
    # Pre-process dataframe for suitability
    
    
    # Uses data
    return

def neural_network():
    """[summary]
    """
    return

def sass_access(data_location):
    # Two files are accessed once for reference
    # sascfg_personal is a configuration file for accessing SAS Ondemand Academic Packages
    '/opt/anaconda3/lib/python3.7/site-packages/saspy'
    # SAS User credientials for granting access
    '/Users/connor/.authinfo'
    # Enable SAS Connection
    session = sas.SASsession()
    # Create sass dataset
    # sass_data = session.submit('''proc import out= fulldata datafile = "/Users/connor/Google Drive/Documents/University/Courses/2020-21/Finance 788/finance-honours/data/combined_predictors_filtered_us.dta";
    # run; ''',results = 'TEXT')
    # proc contents data=hsb2;
    # run;
    # data = sas_session.sasdata2dataframe(data_location)
    # print(data.head())
    # print(data.tail())
    # print(data.info())
    return

# Writes functions
def analytical_analysis():
    # Test simple functionality
    print(sym.sqrt(8))
    theta, x = sym.symbols('O X')
    return


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

# Variables
# File paths
# data_location = '/Users/connor/Google Drive/Documents/University/Courses/2020-21/Finance 788/finance-honours/data/combined_predictors_filtered_us.dta'
data_source = 'data/combined_predictors_filtered_us.dta'
csv_location = '/Volumes/Seagate/dataframes/'

# Binary (Set to True or False depending on the functions to run)
source_data = False
use_sass = False
analytical = False
rank_functions = False
create_data_pipelines = True

# Executes functions
# Calls convert data
if source_data == True:
    convert_data(data_source,csv_location)

# Creates processing pipelines for TensorFlow
if create_data_pipelines == True:
    process_data(csv_location)
# sass_access(data_location)

# Do analytical function
# analytical_analysis('Test')

# Creates monotonic ranking function plots
# ranking_function('linear')

