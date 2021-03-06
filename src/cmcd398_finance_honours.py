#################################################################################
# Information
#################################################################################
# Author; Connor Robert McDowall
# Linux Terminal Call on Google Cloud Platform Virtual Machine Instance
# via SSH connection
# 1) cd finance-honours
# 2) cd src
# 3) python cmcd398-finance-honours.py
#################################################################################
# Module Imports
#################################################################################
# System
import psutil as ps  # Monitor CPU usage
import nvidia_smi  # Monitor GPU usage
import io  # Manipulate strings writing
import os  # Change/manipulate operating systems
import datetime as dt  # Manipulate datetime values
import random as rd  # Random functionality
import csv as csv  # Read and write csvs
import itertools as it  # Create iterators for efficient looping
# Analytical
from pandas.core.base import NoNewAttributesMixin
import sympy as sym  # Symbolic package for calculus
# Machine Learning/AI/Statistics
import numpy as np
from numpy.core.fromnumeric import transpose  # Arithmetic operations
import pandas as pd  # Data analysis package
import dask as ds  # Data importing for very large software packages.
import seaborn as sb  # Imports seaborn library for use
import sklearn as skl  # Simple statistical models
from sklearn.model_selection import train_test_split
import tensorflow as tf  # Tensorflow (https://www.tensorflow.org/)
from tensorflow.keras import layers
from tensorflow.python.eager.def_function import run_functions_eagerly
from tensorflow.python.ops.gen_array_ops import split  # Find combinations of lists
# Keras backend functions to design custom metrics
import tensorflow.keras.backend as K
import linearmodels as lm  # Ability to use PooledOLS
from statsmodels.regression.rolling import RollingOLS  # Use factor loadings
from keras.callbacks import Callback  # Logging training performance
import neptune.new as neptune
from neptunecontrib.monitoring.keras import NeptuneMonitor
import joblib  # Store model history
import pickle
# APIs
import wrds as wrds  # Wharton Research Data Services API
import pydatastream as pds  # Thomas Reuters Datastream API
import yfinance as yf  # Yahoo Finance API
import finance_byu as fin  # Python Package for Fama-MacBeth Regressions
import saspy as sas  # Use saspy functionality in python
import statsmodels.api as sm  # Create Stats functionalities
# Formatting/Graphing
import tabulate as tb  # Create tables in python
import pydot as pyd  # Dynamically generate graphs
import matplotlib.pyplot as plt  # Simple plotting
import scipy as sc  # Scipy packages
# Stargazor package to lm latex tables
from stargazer.stargazer import Stargazer
#################################################################################
# Function Calls
#################################################################################
# System Functions
#################################################################################


def monitor_memory_usage(units, cpu=False, gpu=False):
    """ Function to monitor both CPU & GPU memory consumption

    Args:
        units (int): Memory units (0 = Bytes, 1 = KB, 2 = MB, 3 = GB, 4 = TB, 5 = PB)
        cpu (bool, optional): CPU Information. Defaults to False.
        gpu (bool, optional): GPU Information. Defaults to False.
    """
    # Set unit conversion for readability
    convertor = (1024**units)
    # Shows CPU information using psutil
    if cpu:
        cpu_f = (ps.virtual_memory().available)/convertor
        cpu_t = (ps.virtual_memory().total)/convertor
        cpu_u = (ps.virtual_memory().used)/convertor
        cpu_fp = (ps.virtual_memory().available *
                  100 / ps.virtual_memory().total)
        print("CPU - Memory : ({:.2f}% free): {}(total), {} (free), {} (used)".format(
            cpu_fp, cpu_t, cpu_f, cpu_u))
        # Shows GPU information using nvidia-ml-py3
    if gpu:
        print("GPU Memory Summary")
        nvidia_smi.nvmlInit()
        deviceCount = nvidia_smi.nvmlDeviceGetCount()
        for i in range(deviceCount):
            # Gets device handle
            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
            # Uses handle to get GPU device info
            info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
            # Prints GPU information
            print("GPU - Device {}: {}, Memory : ({:.2f}% free): {}(total), {} (free), {} (used)".format(i, nvidia_smi.nvmlDeviceGetName(
                handle), 100*info.free/info.total, info.total/convertor, info.free//convertor, info.used/convertor))
        nvidia_smi.nvmlShutdown()
    return


def reconfigure_gpu(restrict_tf, growth_memory):
    """ Reconfigures GPU to either restrict the numner of GPU
        or enable allocated GPU to grow on use oppose to allocating
        all memory

    Args:
        restrict_tf (bool): True/False to restrict number of GPUs
        growth_memory (bool): True/False to enable contuous
    """
    # Check the number of GPUs avaiable to Tensorflow and in use
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    # Limit tf to a specfic set of GO devices
    gpus = tf.config.list_physical_devices('GPU')
    # Restrict TensorFlow to only use the first GPU
    if gpus and restrict_tf:
        try:
            tf.config.set_visible_devices(gpus[0], 'GPU')
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,",
                  len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)
    # Limit GPU Memory Growth
    if gpus and growth_memory:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(
                logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    return


def configure_training_ui(project, api_token):
    """ Configures Neptune.ai API, integrated with Github,
    to record and monitor dashboard performance

    Args:
        project (str): Name of Neptune.ai project
        api_token (str): API token to authenticate account

    Returns:
        var: Neptune callback configuration
    """
    # Monitor Keras loss using callback
    # https://app.neptune.ai/common/tf-keras-integration/e/TFK-35541/dashboard/metrics-b11ccc73-9ac7-4126-be1a-cf9a3a4f9b74
    # Initialise neptune with credientials
    run = neptune.init(project=project, api_token=api_token)
    # Define the custom class for the function

    class NeptuneCallback(Callback):
        def __init__(self, run=None, base_namespace=None, batch=None, epoch=None):
            self.run = run
            self.base_namespace = base_namespace
            self.batch = batch
            self.epoch = epoch

        def on_batch_end(self, batch, logs=None):
            batch = self.batch
            run = self.run
            for metric_name, metric_value in logs.items():
                run[f"{metric_name}"].log(metric_value)

        def on_epoch_end(self, epoch, logs=None):
            epoch = self.epoch
            for metric_name, metric_value in logs.items():
                run[f"{metric_name}"].log(metric_value)
    # Find the call back
    neptune_cbk = NeptuneCallback(
        run=run, base_namespace='metrics', batch=None, epoch=None)
    return neptune_cbk


def set_gpus(manual_GPU_device_placement=False):
    """ Set GPU configuration

    Args:
        manual_GPU_device_placement (bool, optional): Mnnual place CPU. Defaults to False.

    Returns:
        [sys]: GPU Device configuration
    """
    gpu = tf.config.list_physical_devices('GPU')
    print(gpu)
    print('Tensorflow GPU: {}'.format(len(gpu)))
    # Set the log device of the GPU
    tf.debugging.set_log_device_placement(True)
    if manual_GPU_device_placement:
        return tf.device('/GPU:0')
    else:
        return

#################################################################################
# Data Processing
#################################################################################


def partition_data(data_location, data_destination):
    """ Converts dta  format to a series of 100k line csvs

    Args:
        data_location (str): directory to source dta file
        data_destination (str): directory to store csvs
    """

    # Converts dta file to chunks
    dflocation = data_destination
    data = pd.read_stata(data_location, chunksize=100000)
    num = 1
    for chunk in data:
        # Saves chunck to seperate csvs given dataset size
        df = pd.DataFrame()
        df = df.append(chunk)
        df.to_csv(dflocation + str(num) + '.csv')
        num_convert = num*100000
        print('Number of rows converted: ', num_convert)
        num = num + 1
    return


def create_dataframes(csv_location, multi_csv):
    """ Creates dataframes

    Args:
        csv_location (str): directory of csvs
        multi_csv (bool): True/False for loading multiple csvs

    Returns:
        dataframe: Returns dataframe after convert the csv file
    """
    # Creates list of dataframes
    num_csvs = list(range(1, 29, 1))
    if multi_csv == False:
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
            np.savetxt(
                r'/home/connormcdowall/finance-honours/data/raw-columns.txt', df.columns, fmt='%s')
        # Save summary statistics to dataframe
        # Changing the script location
        data_stats = df.describe().round(3)
        data_stats.T.to_latex('results/tables/subset-summary-statistics.txt')
        return df
        # Pre-process dataframe for suitability (Remove empty rows, columns etc.)
    else:
        df_list = []
        for num in num_csvs:
            df = pd.read_csv(csv_location + str(num) + ".csv")
            # Append all the dataframes after reading the csv
            df_list.append(df)
            # Concantenate into one dataframe
            df = pd.concat(df_list)
        # Save summary statistics to dataframe
        data_stats = df.describe().round(4)
        data_stats.T.to_latex('results/tables/subset-summary-statistics.txt')
        return df


def sass_access(dataframe):
    """ Remote access to SAS functionalities

    Args:
        dataframe (dataframe): Data to convert to SAS datafile
    """
    # Two files are accessed once for reference
    # sascfg_personal is a configuration file for accessing SAS Ondemand Academic Packages
    '/opt/anaconda3/lib/python3.7/site-packages/saspy'
    # SAS User credientials for granting access
    '/Users/connor/.authinfo'
    # Enable SAS Connection
    session = sas.SASsession()
    # Create sass data
    data = session.dataframe2sasdata(dataframe)
    # Display summary statistics for the data
    data.means()
    return


def replace_nan(df, replacement_method):
    """ Replace/Remove nan files in a dataframe

    Args:
        df (dataframe): Pandas Dataframe
        replacement_method (int): Specify replacement methods
        {0: remove rows with nan values;
        {1: remove columns with nan values;
        {2: fill nan with column mean;
        {3: fill nan with column median}

    Returns:
        dataframe: Updated pandas dataframe
    """
    nan_total = df.isnull().sum().sum()
    print('Number of nan values before processing: ', nan_total)
    if nan_total > 0:
        # Replace dataframe level nan (rows or columns)
        # Replacement methods (0: remove rows with nan values, medium, remove, none)
        if replacement_method == 0:
            df.dropna(axis=0, how='any', inplace=True)
        # Caution: Change to dataframe-columns.txt and features list required (Do not use)
        if replacement_method == 1:
            df.dropna(axis=1, how='any', inplace=True)
        # Replace column level nan
        for column in df.columns:
            if df[column].isnull().sum() > 0:
                if replacement_method == 2:
                    df[column].fillna(df[column].mean(), inplace=True)
                elif replacement_method == 3:
                    df[column].fillna(df[column].median(), inplace=True)
    nan_total = df.isnull().sum().sum()
    print('Number of nan values after processing: ', nan_total)
    return df


def reduce_mem_usage(props):
    """ Function reducing the memory size of a dataframe from Kaggle
        https://www.kaggle.com/arjanso/reducing-dataframe-memory-size-by-65

    Args:
        props (dataframe): Pandas Dataframe

    Returns:
        props (dataframe): Resized Pandas Dataframe
    """
    # Begin the resizing function
    start_mem_usg = props.memory_usage().sum() / 1024**2
    print("Memory usage of properties dataframe is :", start_mem_usg, " MB")
    NAlist = []  # Keeps track of columns that have missing values filled in.
    for col in props.columns:
        if props[col].dtype != object:  # Exclude strings

            # Print current column type
            print("******************************")
            print("Column: ", col)
            print("dtype before: ", props[col].dtype)

            # make variables for Int, max and min
            IsInt = False
            mx = props[col].max()
            mn = props[col].min()

            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(props[col]).all():
                NAlist.append(col)
                props[col].fillna(mn-1, inplace=True)

            # test if column can be converted to an integer
            asint = props[col].fillna(0).astype(np.int64)
            result = (props[col] - asint)
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True

            # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        props[col] = props[col].astype(np.uint8)
                    elif mx < 65535:
                        props[col] = props[col].astype(np.uint16)
                    elif mx < 4294967295:
                        props[col] = props[col].astype(np.uint32)
                    else:
                        props[col] = props[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        props[col] = props[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        props[col] = props[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        props[col] = props[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        props[col] = props[col].astype(np.int64)

            # Make float datatypes 32 bit
            else:
                props[col] = props[col].astype(np.float32)

            # Print new column type
            print("dtype after: ", props[col].dtype)
            print("******************************")

    # Print final result
    print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = props.memory_usage().sum() / 1024**2
    print("Memory usage is: ", mem_usg, " MB")
    print("This is ", 100*mem_usg/start_mem_usg, "% of the initial size")
    return props, NAlist


def resizing_dataframe(dataframe, resizing_options):
    """ Resizes the dataframe to control number of factors
    (fullset) or original ~178, remove mircro and nano size groups,
    and optimise variable type by reducing float64 types to float32.

    Args:
        dataframe (df): Data in dataframe format
        resizing_options (list): List of True/False statements
        to control sizing statements.

    Returns:
        df: Resized dataframe
    """
    print(dataframe.head())
    # Remove both micro
    if resizing_options[0]:
        print('Reducing number of size_grp entries')
        indexNames = dataframe[(dataframe['size_grp'] == 'micro') | (
            dataframe['size_grp'] == 'nano')].index
        dataframe.drop(indexNames, inplace=True)
        print(dataframe.info())
        monitor_memory_usage(units=3, cpu=True, gpu=True)
    # Reduce the number of factors to the original ~178 from JKP
    if resizing_options[1]:
        print('Reducing number of factors to original ~178 from JKP')
        # Extract new columns to the dataframe
        new_columns = []
        list_of_columns = '/home/connormcdowall/finance-honours/data/178-factors.txt'
        file = open(list_of_columns, 'r')
        lines = file.readlines()
        for line in lines:
            line = line.rstrip('\n')
            new_columns.append(line)
        # Only collect column in both lists
        cols = dataframe.columns
        extract_columns = []
        for column in new_columns:
            if column in cols:
                extract_columns.append(column)
        # Extract the old columns
        dataframe = dataframe[extract_columns]
        # Rewrite new working file for numerical encoding
        file = open(
            "/home/connormcdowall/finance-honours/data/working-columns.txt", "r+")
        file.truncate(0)
        file.close()
        textfile = open(
            "/home/connormcdowall/finance-honours/data/working-columns.txt", "w")
        for element in extract_columns:
            textfile.write(element + "\n")
        textfile.close()
        monitor_memory_usage(units=3, cpu=True, gpu=True)
    # Optimises Variable Type
    if resizing_options[2]:
        print('Optimise variable type configuration')
        dataframe, NAlist = reduce_mem_usage(dataframe)
        monitor_memory_usage(units=3, cpu=True, gpu=True)
    return dataframe


def create_original_list_of_columns(dataframe):
    """ Gets the original dataframe list

    Args:
        dataframe (df): Pandas dataframe
    """
    new_columns = []
    list_of_columns = '/home/connormcdowall/finance-honours/data/178-factors.txt'
    file = open(list_of_columns, 'r')
    lines = file.readlines()
    for line in lines:
        line = line.rstrip('\n')
        new_columns.append(line)
    # Only collect column in both lists
    cols = dataframe.columns
    extract_columns = []
    for column in new_columns:
        if column in cols:
            extract_columns.append(column)
    # Rewrite new working file for numerical encoding
    file = open(
        "/home/connormcdowall/finance-honours/data/working-columns.txt", "r+")
    file.truncate(0)
    file.close()
    textfile = open(
        "/home/connormcdowall/finance-honours/data/working-columns.txt", "w")
    for element in extract_columns:
        textfile.write(element + "\n")
    textfile.close()
    return


def split_vm_dataset(data_vm_directory, create_statistics, split_new_data, create_validation_set):
    """ Splits the dta dataset into training, testing, and validation sets

    Args:
        data_vm_directory (str): Directory locating dta file (combined factors)
        create_statistics (bool): True/False to create summary statistics
        split_new_data (bool): True/False to split the data into training/testing
        create_validation_set (bool): Treu/False (nested) to create validation set
    """
    # Create Dataframe from the entire dataset
    # Create summary statisitics for the entire dataset
    if create_statistics == True:
        # Read data into one dataframe on python
        total_df = pd.read_stata(
            data_vm_directory + 'combined_predictors_filtered_us.dta')
        data_stats = total_df.describe().round(4)
        data_stats.T.to_latex('results/tables/summary-statistics.txt')
    # Create training and testing dataframes for Tensorflow
    if split_new_data == True:
        train_df = pd.DataFrame()
        test_df = pd.DataFrame()
        total_df = pd.read_stata(
            data_vm_directory + 'combined_predictors_filtered_us.dta', chunksize=100000)
        for chunk in total_df:
            train_df = train_df.append(chunk[chunk["train"] == 1])
            test_df = test_df.append(chunk[chunk["test"] == 1])
        # Split training set into training and validation
        if create_validation_set == True:
            train_new_df, val_df = train_test_split(train_df, test_size=0.2)
            print(train_df.info())
            print(val_df.info())
            train_new_df.to_stata(data_vm_directory + 'train.dta')
            print('Completed: Training Set')
            val_df.to_stata(data_vm_directory + 'val.dta')
            print('Completed: Validation Set')
        else:
            train_df.to_stata(data_vm_directory + 'train.dta')
            print('Completed: Training Set')
        print(test_df.info())
        test_df.to_stata(data_vm_directory + 'test.dta')
        print('Completed: Testing Set')
    return


def process_vm_dataset(data_vm_dta, size_of_chunks, resizing_options, save_statistics=False, sample=False):
    """ This script processes the training and testing datasets for Tensorflow
    following the classify structured data with feature columns tutorial

    Args:
        data_vm_dta (str): Directory
        size_of_chunks (int): Size of chunks e.g., 10000
        resizing_options ([type]): [description]
        save_statistics (bool, optional): Save Statistics. Defaults to False.
        sample (bool, optional): Process a smaller set of memory. Defaults to False.

    Returns:
        df: Complete dataset
    """
    # Load the test and train datasets into dataframes in chunks
    # df = pd.read_stata(data_vm_dta)
    subset = pd.read_stata(data_vm_dta, chunksize=size_of_chunks)
    df_full = pd.DataFrame()
    predict_df = pd.DataFrame()
    # Uses loop count to create a prediction set
    loop_count = 0
    for df in subset:
        # Adds one to loop count
        loop_count = loop_count + 1
        print('Number of instances: ', len(df))
        print('Excess Return')
        print(df['ret_exc'])
        # Find the dtypes of the dataframe and save them to a data column
        if save_statistics:
            # Saves dtypes for column dataframe
            np.savetxt(
                r'/home/connormcdowall/finance-honours/results/statistics/factor-types.txt', df.dtypes, fmt='%s')
            # Saves information on missing values in the dataframe
            np.savetxt(
                r'/home/connormcdowall/finance-honours/results/statistics/missing-values.txt', df.isna().sum(), fmt='%s')
        # Gets unique list of size_grp
        size_grp_list = list(df['size_grp'].unique())
        # Converts month to integrer format (Need for regressions with predictions)
        print(df['mth'].head())
        for index, row in df.iterrows():
            # Gets datetime value
            datetime = row['mth']
            # Sets year and month values from datetime
            year = datetime.year
            month = datetime.month
            if month < 10:
                month_str = '0'+str(month)
            else:
                month_str = str(month)
            # Concatenates new value and converst to int
            new_mth = int(str(year) + month_str)
            # Sets new month value
            df.at[index, 'mth'] = new_mth
        # Sets mth column to int type
        df['mth'] = df['mth'].astype(int)
        # Sets the second subset to a prediction set (only valide when using a sample)
        if loop_count == 1:
            df_full = df_full.append(df)
        else:
            predict_df = predict_df.append(df)
        # Prints memory usage after the process
        monitor_memory_usage(units=3, cpu=True, gpu=True)
        if sample and loop_count == 2:
            # Process nan options in the dataframes
            df_full = replace_nan(df_full, replacement_method=3)
            predict_df = replace_nan(predict_df, replacement_method=3)
            # Resizes the dataframes based on memory options
            df_full = resizing_dataframe(
                dataframe=df_full, resizing_options=resizing_options)
            predict_df = resizing_dataframe(
                dataframe=predict_df, resizing_options=resizing_options)
            # Saves the prediction dataframe to file
            predict_df.to_stata(
                '/home/connormcdowall/finance-honours/data/dataframes/active_prediction.dta')
            # Print size and shape of dataframe
            print('The dataframe has {} entries with {} rows and {} columns.'.format(
                df_full.size, df_full.shape[0], df_full.shape[1]))
            return df_full
    # Prints size categories in dataframe
    size_grp_list = list(df['size_grp'].unique())
    print('List of size_grp variables')
    print(size_grp_list)
    # Checks Nan in dataframe
    df_full = replace_nan(df_full, replacement_method=3)
    # Memory resizing to prevent excessive memory consumption
    df_full = resizing_dataframe(
        dataframe=df_full, resizing_options=resizing_options)
    # Print size and shape of dataframe
    print('The dataframe has {} entries with {} rows and {} columns.'.format(
        df_full.size, df_full.shape[0], df_full.shape[1]))
    # Prints memory usage after the process
    monitor_memory_usage(units=3, cpu=True, gpu=True)
    return df_full


def save_df_statistics(df, frame_set, statistics_location, data_location):
    """ Save dataframe summary statistics

    Args:
        df (df): Dataframe
        frame_set (str): name of frame
        statistics_location (str): directory to store stats
        data_location (str): directory to store file
    """
    # Sets file paths
    description_file = statistics_location + '/' + frame_set + '-description.txt'
    information_file = statistics_location + '/' + frame_set + '-information.txt'
    datatype_file = statistics_location + '/' + frame_set + '-datatypes.txt'
    data_file = data_location + '/' + 'active_' + frame_set + '.dta'
    # Truncates/clears the information datafiles
    file = open(information_file, "r+")
    file.truncate(0)
    file.close()
    # Saves a descrption of the dataframe
    data_stats_1 = df.describe().round(3)
    data_stats_1.T.to_latex(description_file)
    # Saves the high level overview of the dataframe
    data_stats_2 = df.info(verbose=False)
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    with open(information_file, "w", encoding="utf-8") as f:
        f.write(s)
    # Saves datatypes of the dataframe
    np.savetxt(datatype_file, df.dtypes, fmt='%s')
    # Saves the dataframe to dta files for the regressions
    df.to_stata(data_file)
    return


def convert_datetime_to_int(dataframe, column_name):
    """Convert datetime formats to int

    Args:
        dataframe (df): Dataframe
        column_name (str): column name to convert

    Returns:
        df: Updated dataframe
    """
    # Creates new month column in dataframe
    dataframe[column_name] = int(
        str(dataframe[column_name].year) + str(dataframe[column_name].month))
    print(dataframe[column_name].head())
    return dataframe


def create_fama_factor_models(model_name, selected_losses, factor_location, prediction_location, dependant_column, regression_dictionary, realised_returns=False):
    """ Creates pricing models and regressions from predictions

    Args:
        model_name (Str): Name of run
        selected_losses (Str): Name of loss function
        factor_location (Str): Factors Directory location
        prediction_location (Str): Prediction Directory location
        dependant_column (Str): dependant variable
        regression_dictionary (Str): Checks for regressions
        realised_returns (bool, optional): use realised values. Defaults to False.
    """
    # Note: uses permo and mth to create multiple index for panel regressions
    # permno is the permanent unique firm identifier
    # Reads in all the pandas dataframes
    factors_df = pd.read_csv(factor_location)
    # Divide factors in dataframe by 100 to convert to decimals from percentages
    factors_df[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
               ] = factors_df[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']].div(100)
    # Creates arrays for metric storage
    hp_means = []
    hp_sharpes = []
    hp_treynors = []
    hp_regressions = []
    predictability_regressions = []
    # Sets up actual
    hedge_actual = pd.DataFrame(columns=['mth', 'hedge_returns'])
    # Regress booleans
    regress_prediction = True
    regress_actual = True

    if regress_prediction:
        # Sets actual returns subset
        for loss in selected_losses:
            regression_df = pd.read_csv(
                prediction_location + model_name + '-' + loss + '.csv')
            if realised_returns:
                loss = 'realised-excess-returns'
                dependant_column = 'ret_exc_lead1m'
            # Creates new dataframes for hedge (predicted and return)
            hedge_returns = pd.DataFrame(columns=['mth', 'hedge_returns'])
            hedge_actual = pd.DataFrame(columns=['mth', 'hedge_returns'])
            # Creates portfolio returns via groupings
            monthly_groups = regression_df.groupby("mth")
            for month, subset_predictions in monthly_groups:
                # Sort the predicted returns in the sub_predictiosn set
                subset_predictions.sort_values(
                    by=[dependant_column], ascending=False, inplace=True)
                # Reset the index of this dorted dataframe for forming the hedge portfolio
                subset_predictions.reset_index(drop=True, inplace=True)
                # Calculates decile 1 (Top 10%)
                decile_length = len(subset_predictions[dependant_column])/3
                # print('decile_length: ', decile_length)
                top_decile = range(0, (int(decile_length - 1)))
                bottom_decile = range((int(2*decile_length)),
                                      (int(3 * decile_length-1)))
                # Calculates Hedge Portfolio Return (Decile 1 - Decile 10)
                top_decile_mean = subset_predictions['ret_exc_lead1m'].iloc[top_decile].mean(
                    axis=0)
                bottom_decile_mean = subset_predictions['ret_exc_lead1m'].iloc[bottom_decile].mean(
                    axis=0)
                hp_mean = top_decile_mean - bottom_decile_mean
                # Forms the hedge portfolio and sets to new row
                new_row = {'mth': int(month), 'hedge_returns': hp_mean}
                # Stores the hedge portfolio return for the month in another dataframe
                hedge_returns = hedge_returns.append(
                    new_row, ignore_index=True)

            # Renames 'Date'  column to 'mth'
            factors_df.rename(columns={'Date': 'mth'}, inplace=True)
            # Convert mth dataframe column to the same dtype (float64)
            regression_df['mth'] = regression_df['mth'].astype(np.float64)
            factors_df['mth'] = factors_df['mth'].astype(np.float64)
            hedge_returns['mth'] = hedge_returns['mth'].astype(np.float64)
            # Merges hedge returns with factors
            hedge_returns = hedge_returns.merge(
                factors_df, how='inner', on='mth')
            # Adds the factors to the regression dataframe via merge
            regression_df = regression_df.merge(
                factors_df, how='inner', on='mth')
            # Resets the index on both size_grp and mth
            data = regression_df.set_index(['permno', 'mth'])
            # Do Panel Regressions to determine model predictability of predicted returns
            exog_vars = ['predict']
            # exog = sm.add_constant(data[exog_vars])
            exog = data[exog_vars]
            fb = lm.PooledOLS(data['ret_exc_lead1m'], exog).fit(
                cov_type='clustered', cluster_entity=True, cluster_time=True)
            with open('/home/connormcdowall/finance-honours/results/tables/pooled-ols/accuracy/' + model_name + '-' + loss + '.txt', 'w') as f:
                f.truncate(0)
                print(fb.summary.as_latex(), file=f)
                f.close()
            # Uses stats models to perform standard linear regressions
            predict_regress = sm.OLS(data['ret_exc_lead1m'], exog).fit(
                cov_type='HAC', cov_kwds={'maxlags': 6})
            if loss in ['mean_squared_error', 'custom_mse', 'custom_hp']:
                predictability_regressions.append(predict_regress)

            print(hedge_returns['hedge_returns'])
            print(hedge_returns[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']])

            # Get regression for asset pricing models
            if regression_dictionary['capm'] == True:
                # Uses linear models to perform CAPM regressions (Panel Regressions)
                capm_exog_vars = ['Mkt-RF']
                capm_exog = sm.add_constant(data[capm_exog_vars])
                capm_fb = lm.PooledOLS(
                    data['ret_exc_lead1m'], capm_exog).fit(cov_type='clustered', cluster_entity=True, cluster_time=True)
                with open('/home/connormcdowall/finance-honours/results/tables/pooled-ols/capm/' + model_name + '-' + loss + '-capm.txt', 'w') as f:
                    f.truncate(0)
                    print(capm_fb.summary.as_latex(), file=f)
                    f.close()
                # Uses stats models to perform standard linear regressions
                capm_hp_exog = sm.add_constant(hedge_returns[capm_exog_vars])
                capm_hp = sm.OLS(hedge_returns['hedge_returns'], capm_hp_exog).fit(
                    cov_type='HAC', cov_kwds={'maxlags': 6})
            if regression_dictionary['ff3'] == True:
                # Uses linear models to perform FF3 regression (Panel Regressions)
                ff3_exog_vars = ['Mkt-RF', 'SMB', 'HML']
                ff3_exog = sm.add_constant(data[ff3_exog_vars])
                ff3_fb = lm.PooledOLS(data['ret_exc_lead1m'],
                                      ff3_exog).fit(cov_type='clustered', cluster_entity=True, cluster_time=True)
                with open('/home/connormcdowall/finance-honours/results/tables/pooled-ols/ff3/' + model_name + '-' + loss + '-ff3.txt', 'w') as f:
                    f.truncate(0)
                    print(ff3_fb.summary.as_latex(), file=f)
                    f.close()
                # Uses stats models to perform standard linear regressions
                ff3_hp_exog = sm.add_constant(hedge_returns[ff3_exog_vars])
                ff3_hp = sm.OLS(hedge_returns['hedge_returns'], ff3_hp_exog).fit(
                    cov_type='HAC', cov_kwds={'maxlags': 6})
            if regression_dictionary['ff4'] == True:
                # Uses linear models to perform FF4 (Carhart) regression (Panel Regressions)
                ff4_exog_vars = ['Mkt-RF', 'SMB', 'HML', 'RMW']
                ff4_exog = sm.add_constant(data[ff4_exog_vars])
                ff4_fb = lm.PooledOLS(data['ret_exc_lead1m'],
                                      ff4_exog).fit(cov_type='clustered', cluster_entity=True, cluster_time=True)
                with open('/home/connormcdowall/finance-honours/results/tables/pooled-ols/ff4/' + model_name + '-' + loss + '-ff4.txt', 'w') as f:
                    f.truncate(0)
                    print(ff4_fb.summary.as_latex(), file=f)
                    f.close()
                print(ff4_fb)
                # Uses stats models to perform standard linear regressions
                ff4_hp_exog = sm.add_constant(hedge_returns[ff4_exog_vars])
                ff4_hp = sm.OLS(hedge_returns['hedge_returns'], ff4_hp_exog).fit(
                    cov_type='HAC', cov_kwds={'maxlags': 6})
            if regression_dictionary['ff5'] == True:
                # Uses linear model to perform FF5 regression (Panel Regressions)
                ff5_exog_vars = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
                ff5_exog = sm.add_constant(data[ff5_exog_vars])
                ff5_fb = lm.PooledOLS(data['ret_exc_lead1m'],
                                      ff5_exog).fit(cov_type='clustered', cluster_entity=True, cluster_time=True)
                with open('/home/connormcdowall/finance-honours/results/tables/pooled-ols/ff5/' + model_name + '-' + loss + '-ff5.txt', 'w') as f:
                    f.truncate(0)
                    print(ff5_fb.summary.as_latex(), file=f)
                    f.close()
                # Uses stats models to perform standard linear regressions
                ff5_hp_exog = sm.add_constant(hedge_returns[ff5_exog_vars])
                ff5_hp = sm.OLS(hedge_returns['hedge_returns'], ff5_hp_exog).fit(
                    cov_type='HAC', cov_kwds={'maxlags': 6})

            # Extract the metrics from loss function
            hp_mean = np.asscalar(
                hedge_returns[['hedge_returns']].mean(axis=0))
            print('Hedge Portfolio Mean for {} is {}'.format(loss, hp_mean))
            hp_sharpe_ratio = np.asscalar((hedge_returns[['hedge_returns']].mean(
                axis=0)/hedge_returns[['hedge_returns']].std(axis=0)))
            print('Hedge Portfolio Sharpe Ratio for {} is {}'.format(
                loss, hp_sharpe_ratio))
            hp_teynor = np.asscalar((hedge_returns[['hedge_returns']].mean(
                axis=0) / capm_hp.params[1]))
            print('Hedge Portfolio treynor for {} is {}'.format(loss, hp_teynor))
            hp_means.append(hp_mean)
            print('Hedge portfolio means are: ', hp_means)
            hp_sharpes.append(hp_sharpe_ratio)
            print('Hedge portfolio sharpe ratios are: ', hp_sharpes)
            hp_treynors.append(hp_teynor)
            print('Hedge portfolio means are: ', hp_treynors)
            hp_regress = sm.OLS(hedge_returns['hedge_returns'], hedge_returns['hedge_returns']).fit(
                cov_type='HAC', cov_kwds={'maxlags': 6})
            hp_regressions.append(hp_regress)

            # Creates tables for comparison using the stargazor package
            hp_stargazer = Stargazer([capm_hp, ff3_hp, ff4_hp, ff5_hp])
            with open('/home/connormcdowall/finance-honours/results/tables/hedge-portfolio-ols/' + model_name + '-' + loss + '.txt', 'w') as f:
                # Deletes existing text
                f.truncate(0)
                print(hp_stargazer.render_latex(), file=f)
        # Uses stargazer and arrays
            if realised_returns:
                return

    if regress_actual:
        # Calculates actual portfolio metrics
        # Calculate standard portfolio return not on the metric
        regression_actual_df = pd.read_csv(
            prediction_location + model_name + '-' + 'mean_squared_error' + '.csv')
        # print(regression_actual_df.head())
        monthly_groups = regression_actual_df.groupby("mth")
        for month, subset_predictions in monthly_groups:
            # Sort the predicted returns in the sub_predictiosn set
            subset_predictions.sort_values(
                by=['ret_exc_lead1m'], ascending=False, inplace=True)
            # Reset the index of this dorted dataframe for forming the hedge portfolio
            subset_predictions.reset_index(drop=True, inplace=True)
            # Calculates decile 1 (Top 10%)
            decile_length = len(subset_predictions['ret_exc_lead1m'])/3
            top_decile = [0, int(decile_length - 1)]
            bottom_decile = [int(2*decile_length), int(3*decile_length-1)]
            # Calculates decile means
            top_decile_mean = subset_predictions['ret_exc_lead1m'].iloc[top_decile[0]: top_decile[1]].mean(
                axis=0)
            bottom_decile_mean = subset_predictions['ret_exc_lead1m'].iloc[bottom_decile[0]: bottom_decile[1]].mean(
                axis=0)
            hp_mean = top_decile_mean - bottom_decile_mean
            # Forms the hedge portfolio and sets to new row
            new_row = {'mth': int(month), 'hedge_returns': hp_mean}
            # Stores the hedge portfolio return for the month in another dataframe
            hedge_actual = hedge_actual.append(new_row, ignore_index=True)
        hedge_actual.to_csv(
            '/home/connormcdowall/finance-honours/results/predictions/cmcd398-finance-honours-hedge-actual.csv')
        # Renames 'Date'  column to 'mth'
        factors_df.rename(columns={'Date': 'mth'}, inplace=True)
        # Converts mth dataframe column to the same dtype (float64)
        regression_actual_df['mth'] = regression_actual_df['mth'].astype(
            np.float64)
        hedge_actual['mth'] = hedge_actual['mth'].astype(np.float64)
        print(hedge_actual.head())
        # Merges hedge returns with factors
        hedge_actual = hedge_actual.merge(factors_df, how='inner', on='mth')

        # Uses statsmodels to get market approximations
        capm_actual_exog = sm.add_constant(hedge_actual[['Mkt-RF']])
        ff3_actual_exog = sm.add_constant(
            hedge_actual[['Mkt-RF', 'SMB', 'HML']])
        ff4_actual_exog = sm.add_constant(
            hedge_actual[['Mkt-RF', 'SMB', 'HML', 'RMW']])
        ff5_actual_exog = sm.add_constant(
            hedge_actual[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']])

        capm_actual_hp = sm.OLS(hedge_actual['hedge_returns'], capm_actual_exog).fit(
            cov_type='HAC', cov_kwds={'maxlags': 6})
        ff3_actual_hp = sm.OLS(hedge_actual['hedge_returns'], ff3_actual_exog).fit(
            cov_type='HAC', cov_kwds={'maxlags': 6})
        ff4_actual_hp = sm.OLS(hedge_actual['hedge_returns'], ff4_actual_exog).fit(
            cov_type='HAC', cov_kwds={'maxlags': 6})
        ff5_actual_hp = sm.OLS(hedge_actual['hedge_returns'], ff5_actual_exog).fit(
            cov_type='HAC', cov_kwds={'maxlags': 6})

        # Save the model to file
        hp_stargazer = Stargazer(
            [capm_actual_hp, ff3_actual_hp, ff4_actual_hp, ff5_actual_hp])
        with open('/home/connormcdowall/finance-honours/results/tables/hedge-portfolio-ols/' + model_name + '-' + 'omlhpex' + '-actual.txt', 'w') as f:
            # Deletes existing text
            f.truncate(0)
            print(hp_stargazer.render_latex(), file=f)

        # Sets up arrays to append
        hp_means_actual = []
        hp_sharpes_actual = []
        hp_treynors_actual = []

        # Extract the metrics from loss function
        hp_mean_actual = np.asscalar(
            hedge_actual[['hedge_returns']].mean(axis=0))
        print('Traditional Hedge Portfolio Mean for {} is {}'.format(
            loss, hp_mean_actual))
        hp_sharpe_ratio_actual = np.asscalar((hedge_actual[['hedge_returns']].mean(
            axis=0)/hedge_actual[['hedge_returns']].std(axis=0)))
        print('Actual Hedge Portfolio Sharpe Ratio for {} is {}'.format(
            loss, hp_sharpe_ratio_actual))
        hp_treynor_actual = np.asscalar((hedge_actual[['hedge_returns']].mean(
            axis=0) / capm_actual_hp.params[1]))
        print('Actual Hedge Portfolio treynor for {} is {}'.format(
            loss, hp_treynor_actual))

        # Append to array
        hp_means_actual.append(hp_mean_actual)
        print('Actual Hedge portfolio means are: ', hp_means)
        hp_sharpes_actual.append(hp_sharpe_ratio_actual)
        print('Actual Hedge portfolio sharpe ratios are: ', hp_sharpes)
        hp_treynors_actual.append(hp_treynor_actual)
        print('Actual Hedge portfolio treynors are: ', hp_treynors)

    if regress_prediction:
        # Creates metrics dataframes and saves to latex variable
        metrics_df_cols = ['Loss Function', 'HP Mean',
                           'Sharpe Ratio', 'Treynor Ratio']
        metrics_df = pd.DataFrame(columns=metrics_df_cols)

        # Set name if just
        traditional_sort = 'traditional_sort'

        # Append to dataframe
        hp_means.append(hp_means_actual[0])
        hp_sharpes.append(hp_sharpes_actual[0])
        hp_treynors.append(hp_treynors_actual[0])
        selected_losses.append(traditional_sort[0])

        # Create the metrics dataframe
        metrics_df['HP Mean'] = hp_means
        metrics_df['Loss Function'] = selected_losses
        metrics_df['Sharpe Ratio'] = hp_sharpes
        metrics_df['Treynor Ratio'] = hp_treynors

        # Only extract the most relevant factors
        truncate = True
        if truncate:
            metrics_df = metrics_df[(metrics_df['Loss Function'] == 'mean_squared_error') | (
                metrics_df['Loss Function'] == 'custom_mse') | (metrics_df['Loss Function'] == 'custom_hp')]
        with open('/home/connormcdowall/finance-honours/results/tables/metrics/' + model_name + '-calculations-metrics.txt', 'w') as f:
            # Deletes existing text
            f.truncate(0)
            print(metrics_df.to_latex(index=False), file=f)
            f.close()

        # Create new sharelatex regression columns
        hp_metric_stargazer = Stargazer(predictability_regressions)
        with open('/home/connormcdowall/finance-honours/results/tables/metrics/' + model_name + '-regression-metrics.txt', 'w') as f:
            # Deletes existing text
            f.truncate(0)
            print(hp_metric_stargazer.render_latex(), file=f)
            f.close()
    return


def sort_data_chronologically(data_directory, size_of_chunks, set_top_500=False):
    """Processes the dataset to the following chronilogical order
        Training: Before 1990 [1861,1989]
        Validation: Between 1990 and 1999 [1990,1999]
        Testing: After 2000 [2000,2021]

    Args:
        data_directory (str): Directory of stored data on the virtual instance
        size_of_chunks (int): Chunk_size for reading pandas dataframes
        set_top_500 (bool, optional): Select top 500 equities. Defaults to False.
    """
    dataframes = ['train.dta', 'test.dta', 'val.dta']
    # Each dataframe in turn
    for dataframe in dataframes:
        subset = pd.read_stata(data_directory + dataframe,
                               chunksize=size_of_chunks)
        # Create new dataframe
        df_full = pd.DataFrame()
        for df in subset:
            # Monitor memeory usage
            monitor_memory_usage(units=3, cpu=True, gpu=True)
            # Converts mth from datetime to int
            for index, row in df.iterrows():
                # Gets datetime value
                datetime = row['mth']
                # Sets year and month values from datetime
                year = datetime.year
                month = datetime.month
                if month < 10:
                    month_str = '0'+str(month)
                else:
                    month_str = str(month)
                # Concatenates new value and converst to int
                new_mth = int(str(year) + month_str)
                # Sets new month value
                df.at[index, 'mth'] = new_mth
            # Sets mth column to int type
            df['mth'] = df['mth'].astype(int)
            # Monitor memeory usage
            monitor_memory_usage(units=3, cpu=True, gpu=True)
            # Removes nans
            df = replace_nan(df, replacement_method=3)
            # Resizes the dataframe (Resize dataframe)
            df = resizing_dataframe(df, resizing_options=[True, True, True])
            df_full = df_full.append(df)
            # Prints list of unique months
            print(df_full.info(verbose=True))
            print(sorted(df_full['mth'].unique()))
        # Saves the formatted dataframe to file
        df_full.to_stata(data_directory + 'sorted_' + dataframe)
    # Inititalises new chronological dataframes
    first_loop = True
    for dataframe in dataframes:
        df = pd.read_stata(data_directory + 'sorted_' + dataframe)
        column_names = df.columns
        print(df.head())
        monitor_memory_usage(units=3, cpu=True, gpu=True)
        train_subset = df[(df["mth"] <= 198912)]
        print('Training Subset')
        print(train_subset.head())
        val_subset = df[(df["mth"] > 198912) & (df["mth"] <= 199912)]
        print('Validation Subset')
        print(val_subset.head())
        test_subset = df[(df["mth"] > 199912)]
        print('Testing Subset')
        print(test_subset.head())
        # Concatenate dataframes while ignoring the appendix
        if first_loop == True:
            train_chronological = pd.DataFrame(
                train_subset, columns=column_names)
            val_chronological = pd.DataFrame(val_subset, columns=column_names)
            test_chronological = pd.DataFrame(
                test_subset, columns=column_names)
            first_loop = False
        else:
            train_chronological = pd.concat(
                [train_chronological, train_subset], ignore_index=True)
            val_chronological = pd.concat(
                [val_chronological, val_subset], ignore_index=True)
            test_chronological = pd.concat(
                [test_chronological, test_subset], ignore_index=True)
    # Drops the Level_0 column
    train_chronological = train_chronological.drop(
        columns=['level_0'])
    val_chronological = val_chronological.drop(columns=['level_0'])
    test_chronological = test_chronological.drop(
        columns=['level_0'])
    # Prints the new dataframes
    print('Training')
    print(train_chronological.info(verbose=True))
    print(sorted(train_chronological['mth'].unique()))
    print('Validation')
    print(val_chronological.info(verbose=True))
    print(sorted(val_chronological['mth'].unique()))
    print('Testing')
    print(test_chronological.info(verbose=True))
    print(sorted(test_chronological['mth'].unique()))
    # Saves the chronological files to file
    train_chronological.to_stata(data_directory + 'active_train.dta')
    val_chronological.to_stata(data_directory + 'active_val.dta')
    test_chronological.to_stata(data_directory + 'active_test.dta')
    return


def convert_txt_to_tex(fp_in, fp_out, replace_text=False, replacement_text=None):
    """Convert text files to latex format

    Args:
        fp_in (str): Path in
        fp_out (str): Path out
        replace_text (bool, optional): Replace text in file. Defaults to False.
        replacement_text ([type], optional): Text to replace. Defaults to None.
    """
    texdoc = []  # a list of string representing the latex document in python
    # Set base directory
    with open(fp_in, 'r') as fin:
        lines = fin.readlines()
        for line in lines:
            print(line)
            if replace_text:
                try:
                    line = line.replace('(1)', 'CAPM')
                except:
                    pass
                try:
                    line = line.replace('(2)', 'FF3')
                except:
                    pass
                try:
                    line = line.replace('(3)', 'FF4')
                except:
                    pass
                try:
                    line = line.replace('(4)', 'FF5')
                except:
                    pass
                try:
                    texdoc.append(line.replace(
                        'Dependent variable', replacement_text))
                except:
                    pass
            else:
                texdoc.append(line)
    fin.close()
    print(texdoc)
    # write back the new document
    with open(fp_out, 'w') as fout:
        fout.truncate(0)
        for i in range(len(texdoc)):
            fout.write(texdoc[i])
    fout.close()
    return


def execute_conversion_options(model_name, selected_losses, hp_ols=False, pooled_ols=False, true_excess_returns=False):
    """Do all the text to tex conversion.

    Args:
        model_name (Str): 
        selected_losses ([type]): [description]
        hp_ols (bool, optional): Do HP OLS. Defaults to False.
        pooled_ols (bool, optional): Do pooled OLS. Defaults to False.
        true_excess_returns (bool, optional): Do realised OLS. Defaults to False.
    """
    base_directory_in = '/home/connormcdowall/finance-honours/results/tables/'
    base_directory_out = '/home/connormcdowall/finance-honours/results/tex/'
    factor_models = ['capm', 'ff3', 'ff4', 'ff5']
    replacement_set_losses = ['mean squared error', 'custom mse',
                              'custom sharpe', 'custom sharpe mse', 'custom information', 'custom hp']
    replace_count = 0
    if true_excess_returns:
        selected_losses.append('realised-excess-returns')
        replacement_set_losses.append('realised excess returns')
    for loss in selected_losses:
        replacement_text = 'One Month Lead Excess Portfolio Return using ' + \
            replacement_set_losses[replace_count]
        # Get model accruracy measures
        if loss in ['mean_squared_error', 'custom_mse', 'custom_hp']:
            fp_in = base_directory_in + 'pooled-ols/accuracy/' + \
                model_name + '-' + loss + '.txt'
            fp_out = base_directory_out + model_name + '-' + loss + '-accuracy.tex'
            convert_txt_to_tex(fp_in, fp_out, replace_text=False,
                               replacement_text=replacement_text)
        if hp_ols:
            fp_in = base_directory_in + 'hedge-portfolio-ols/' + \
                model_name + '-' + loss + '.txt'
            fp_out = base_directory_out + model_name + '-' + loss + '.tex'
            convert_txt_to_tex(fp_in, fp_out, replace_text=True,
                               replacement_text=replacement_text)
        if pooled_ols:
            for factor in factor_models:
                fp_in = base_directory_in + 'pooled-ols/' + factor + \
                    '/' + model_name + '-' + loss + '-' + factor + '.txt'
                fp_out = base_directory_out + model_name + '-' + loss + '-' + factor + '.tex'
                convert_txt_to_tex(
                    fp_in, fp_out, replace_text=True, replacement_text=replacement_text)
        replace_count = replace_count + 1
    # Get the metrics file
    fp_in = base_directory_in + 'metrics/' + \
        model_name + '-calculations-metrics.txt'
    fp_out = base_directory_out + model_name + '-calculations-metrics.tex'
    convert_txt_to_tex(fp_in, fp_out, replace_text=False,
                       replacement_text=replacement_text)
    # Get the regression summary
    fp_in = base_directory_in + 'metrics/' + \
        model_name + '-regression-metrics.txt'
    fp_out = base_directory_out + model_name + '-regression-metrics.tex'
    convert_txt_to_tex(fp_in, fp_out, replace_text=False,
                       replacement_text=replacement_text)
    return
#################################################################################
# Machine Learning
#################################################################################
# Utility method to use pandas dataframe to create a tf.data dataset
# Adapted from https://www.tensorflow.org/tutorials/structured_data/feature_columns#use_pandas_to_create_a_dataframe
# Adapted from https://www.tensorflow.org/tutorials/structured_data/preprocessing_layers


def download_test_data():
    """Download test data

    Returns:
        [list]]: List of different dataframes (Total, training, validation, testing)
    """
    dataset_url = 'http://storage.googleapis.com/download.tensorflow.org/data/petfinder-mini.zip'
    csv_file = 'datasets/petfinder-mini/petfinder-mini.csv'
    tf.keras.utils.get_file('petfinder_mini.zip', dataset_url,
                            extract=True, cache_dir='.')
    dataframe = pd.read_csv(csv_file)

    # Creates the target variable for the assignment
    dataframe['target'] = np.where(dataframe['AdoptionSpeed'] == 4, 0, 1)
    # Drop unused features.
    dataframe = dataframe.drop(columns=['AdoptionSpeed', 'Description'])
    # Split the dataset into training, validation and testing sets
    train, val, test = np.split(dataframe.sample(
        frac=1), [int(0.8*len(dataframe)), int(0.9*len(dataframe))])
    # Returns the dataframe and the three subsets
    return dataframe, train, val, test


def create_feature_lists(list_of_columns, categorical_assignment):
    """ Creates required feature lists of normalisation and encoding

    Args:
        list_of_columns ([type]): [description]
        categorical_assignment ([type]): [description]

    Returns:
        numerical_features: 
        categorical_features: 
    """
    # Assignn variables
    categorical_features = []
    numerical_features = []
    file = open(list_of_columns, 'r')
    lines = file.readlines()
    for line in lines:
        line = line.rstrip('\n')
        if line in categorical_assignment:
            categorical_features.append(line)
        else:
            numerical_features.append(line)
    # Returns numerical and categorical features
    return numerical_features, categorical_features


def shuffle_columns(df, column_name):
    """Shuffles columns to front of the dataframe

    Args:
        df (df): Dataframe
        column_name (Str): Column name

    Returns:
        [df]: Dataframe
    """
    column_to_insert = df[column_name]
    df.drop(labels=[column_name], axis=1, inplace=True)
    df.insert(0, column_name, column_to_insert)
    return df


def create_tf_dataset(dataframe, target_column, shuffle=True, batch_size=32):
    """Set target variable and converts dataframe to tensorflow dataset

    Args:
        df (dataframe): dataframe
        target_column (str): Column used to predict for labels
        shuffle (bool, optional): [description]. Defaults to True.
        batch_size (int, optional): Sets batch size. Defaults to 32.

    Returns:
        [ds]: Tensorflow dataset
    """
    df = dataframe.copy()
    print('Dataframe Before')
    print(list(df.columns))
    # Returns the labels and drop columns from dataframe
    labels = df.pop(target_column)
    # List of columns to be shifted columns
    shift_columns = ['beta_60m_x']
    for col in shift_columns:
        df = shuffle_columns(df, col)
    print('Dataframe After')
    print(list(df.columns))
    # This call
    df = {key: value[:, tf.newaxis] for key, value in dataframe.items()}
    ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=(int(len(dataframe)/30)))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(batch_size)
    print('Create Dataset: Successful')
    return ds


def get_normalization_layer(name, dataset):
    """Get normalisation error

    Args:
        name ([type]): [description]
        dataset ([type]): [description]

    Returns:
        [type]: [description]
    """
    # Create a Normalization layer for the feature.
    # Layer Normalization normalizes each feature of the activations
    # to zero mean and unit variance.
    normalizer = layers.Normalization(axis=None)
    # Prepare a Dataset that only yields the feature.
    feature_ds = dataset.map(lambda x, y: x[name])
    # Learn the statistics of the data.
    normalizer.adapt(feature_ds)
    return normalizer


def get_category_encoding_layer(name, dataset, dtype, max_tokens=None):
    """ Get encoding layer for categorical variables

    Args:
        name (str): [description]
        dataset (ds): Tensroflow dataset
        dtype (str): Datatype for encoded variable
        max_tokens (int, optional): Number of max tokens. Defaults to None.

    Returns:
        [lamdba]: lambda function for the encoded feature
    """
    # Create a layer that turns strings into integer indices.
    if dtype == 'string':
        index = layers.StringLookup(max_tokens=max_tokens)
    # Otherwise, create a layer that turns integer values into integer indices.
    else:
        index = layers.IntegerLookup(max_tokens=max_tokens)
    # Prepare a `tf.data.Dataset` that only yields the feature.
    feature_ds = dataset.map(lambda x, y: x[name])
    # Learn the set of possible values and assign them a fixed integer index.
    index.adapt(feature_ds)
    # Encode the integer indices.
    encoder = layers.CategoryEncoding(num_tokens=index.vocabulary_size())
    # Apply multi-hot encoding to the indices. The lambda function captures the
    # layer, so you can use them, or include them in the Keras Functional model later.
    return lambda feature: encoder(index(feature))


def encode_tensor_flow_features(train_df, val_df, test_df, target_column, numerical_features, categorical_features, categorical_dictionary, size_of_batch=256):
    """Encodes tensorflow features

    Args:
        train_df (df): Training dataframe
        val_df (df): Validation dataframe
        test_df (df): Testign dataframe
        target_column (str): Target column for prediction
        numerical_features (list): List of numerical features
        categorical_features (list): List of categorical features
        categorical_dictionary (dict): Dictionary of categorical features
        size_of_batch (int, optional): Batch size. Defaults to 256.

    Returns:
        all_features (tf.layer): Tensorflow layer of all features
        all_inputs (list): Tensorflow layer of all inputs
        train_dataset (ds): Training Dataset
        val_dataset (ds): Validation Dataset
        test_dataset (ds): Testing Dataset
    """
    # Creates the dataset
    train_dataset = create_tf_dataset(
        train_df, target_column, shuffle=False, batch_size=size_of_batch)
    val_dataset = create_tf_dataset(
        val_df, target_column, shuffle=False, batch_size=size_of_batch)
    test_dataset = create_tf_dataset(
        test_df, target_column, shuffle=False, batch_size=size_of_batch)

    # print lengths of dataset
    print('Length of train dataset: ', len(list(train_dataset)))
    print('Length of val dataset: ', len(list(val_dataset)))
    print('Length of test dataset: ', len(list(test_dataset)))
    # Display a set of batches
    [(train_features, label_batch)] = train_dataset.take(1)

    # Initilise input and encoded feature arrays
    all_inputs = []
    encoded_features = []
    numerical_count = 0
    categorical_count = 0

    # Encode the categorical features
    for header in categorical_features:
        try:
            print('Start: ', header)
            categorical_col = tf.keras.Input(
                shape=(1,), name=header, dtype=categorical_dictionary[header])
            print('Processing: Input Categorical Column')
            encoding_layer = get_category_encoding_layer(name=header,
                                                         dataset=train_dataset,
                                                         dtype=categorical_dictionary[header],
                                                         max_tokens=5)
            print('Processing: Sourced Encoding Layer')
            encoded_categorical_col = encoding_layer(categorical_col)
            print('Processing: Encoded Categorical Column')
            all_inputs.append(categorical_col)
            encoded_features.append(encoded_categorical_col)
            print('Passed: ', header)
            categorical_count = categorical_count + 1
            print('Number of Categorical Features Encoded: ', categorical_count)
        except RuntimeError as e:
            print(e)
        # Monitor memory usage
        monitor_memory_usage(units=3, cpu=True, gpu=True)
    # Normalise the numerical features
    for header in numerical_features:
        try:
            print('Start: ', header)
            numeric_col = tf.keras.Input(shape=(1,), name=header)
            print('Processing: Input Numeric Column')
            normalization_layer = get_normalization_layer(
                header, train_dataset)
            print('Processing: Sourced Normalization Layer')
            encoded_numeric_col = normalization_layer(numeric_col)
            print('Processing: Encoded Numerical Column')
            all_inputs.append(numeric_col)
            encoded_features.append(encoded_numeric_col)
            print('Passed: ', header)
            numerical_count = numerical_count + 1
            print('Number of Numerical Features Encoded: ', numerical_count)
        except RuntimeError as e:
            print(e)
        # Monitor memory usage
        monitor_memory_usage(units=3, cpu=True, gpu=True)
    # Concatenate all encoded layers
    all_features = tf.keras.layers.concatenate(encoded_features)
    # Try to pickle both all inputs and all encoded features
    try:
        with open('/home/connormcdowall/finance-honours/results/models/features/all-inputs', 'wb') as f:
            pickle.dump(all_inputs, f)
            f.close()
            print('Complete: Pickled all inputs')
    except:
        print('Complete: Pickled all inputs')
    try:
        with open('/home/connormcdowall/finance-honours/results/models/features/all-features', 'wb') as f:
            pickle.dump(all_features, f)
            f.close()
            print('Complete: Pickled all features')
    except:
        print('Complete: Pickled all features')
    print('All Features')
    print(all_features)
    print('Encoding: Successful')
    # Monitor memory usage
    monitor_memory_usage(units=3, cpu=True, gpu=True)
    # Try saving tensorflow dataset

    return all_features, all_inputs, train_dataset, val_dataset, test_dataset


def build_tensor_flow_model(train_dataset, val_dataset, test_dataset, model_name, all_features, all_inputs, selected_optimizer, selected_losses, selected_metrics, finance_configuration=True):
    """Builds tensorflow neural networks

    Args:
        train_dataset (ds): Training dataset
        val_dataset (ds): Validation dataset
        test_dataset (ds): Testing dataset
        model_name (str): Run name
        all_features ([type]): [description]
        all_inputs ([type]): [description]
        selected_optimizer (str): optimizer to use
        selected_losses (str): Loss function to use
        selected_metrics (list): List of sleection metrics to use
        finance_configuration (bool, optional): Run the complex configuration options. Defaults to True.

    Returns:
        model (tf.model): Tensorflow model
        loss (float): Loss metric
        accuracy (float): Accuracy metric
    """
    # Information pertaining to the tf.keras.layers.dense function
    if finance_configuration:
        # Note: The combination of optimizer. loss function and metric must be compatible
        # https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense
        # Generalised Artificial Neural Network
        # Input features (One per feature)
        # Hidden Layers (1-5)
        # Neurons per input layer (10-100)
        # Output neurons (1 per prediction dimension)
        # Hidden activations (Relum Tanh, Sigmoid)
        # Output layer (sigmoid)

        # List of activation functions:
        # 'linear' = Linear activation function (final layer)
        # 'relu' = Rectified linear unit activation
        # 'sigmoid' = Sigmoid activation function, sigmoid(x) = 1 / (1 + exp(-x)).
        # 'softmax' = Softmax converts a vector of values to a probability distribution
        # 'softplus' = Softplus activation function, softplus(x) = log(exp(x) + 1)
        # 'softsign' = Softsign activation function, softsign(x) = x / (abs(x) + 1).
        # 'tanh' = Hyperbolic tangent activation function.
        # 'selu' = Scaled Exponential Linear Unit (SELU) activation function is defined as:
        #   if x > 0: return scale * x
        #   if x < 0: return scale * alpha * (exp(x) - 1)
        # 'elu' = The exponential linear unit (ELU) with alpha > 0 is:
        # x if x > 0 and alpha * (exp(x) - 1) if x < 0
        # Note: The ELU hyperparameter alpha controls the value to which an ELU saturates
        # for negative net inputs. ELUs diminish the vanishing gradient effect.
        # https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dropout
        # Dropout layer to randomly set input units to zero with a deterministic rate
        # during each step of training to help prevent overfitting. Note:
        # inputs not set to zero are scaled by 1/(1-rate) so the sum of all inputs is unchanged.

        # Configure the neural network layers
        print('Start: Configuration of Deep Network Layers')
        # Binary variables to control network construction
        complex_model = True
        # Simple configuration, only a handful of layers
        if complex_model:
            # Initial Layer
            layer_1 = tf.keras.layers.Dense(
                32, activation="relu")(all_features)
            # Dropout layer
            layer_2 = tf.keras.layers.Dropout(
                rate=0.5, noise_shape=None, seed=None)(layer_1)
            layer_3 = tf.keras.layers.Dense(64, activation='relu')(layer_2)
            layer_4 = tf.keras.layers.Dense(128, activation='relu')(layer_3)
            # Creates the output layer
            output = tf.keras.layers.Dense(1, activation='linear')(layer_4)
            print('End: Configuration of Deep Network Layers')
            # Configure the model (https://www.tensorflow.org/api_docs/python/tf/keras/Model)
            model = tf.keras.Model(all_inputs, output)
            print('Model Summary')
            print(model.summary)
        # Deploy a sequential model
        else:
            # Initial Layer
            x = tf.keras.layers.Dense(
                units=32, activation="relu", use_bias=True,
                kernel_initializer='glorot_uniform',
                bias_initializer='zeros', kernel_regularizer=None,
                bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
                bias_constraint=None)(all_features)
            # Dropout layer
            x = tf.keras.layers.Dropout(
                rate=0.5, noise_shape=None, seed=None)(x)
            # Creates the output layer
            output = tf.keras.layers.Dense(1)(x)
            print('End: Configuration of Deep Network Layers')
            # Configure the model (https://www.tensorflow.org/api_docs/python/tf/keras/Model)
            model = tf.keras.Model(all_inputs, output)
        # Initilises optimizer variables
        lr = 0.001
        eps = 1e-07
        rh = 0.95
        mom = 0.0
        b0 = 0.0
        b1 = 0.9
        b2 = 0.999
        iav = 0.1
        lrp = -0.5
        l1rs = 0.0
        l2rs = 0.0
        l2srs = 0.0
        ams = False
        cen = False
        nes = False
        #################################################################################
        # Optimizer (https://www.tensorflow.org/api_docs/python/tf/keras/optimizers)
        #################################################################################
        if selected_optimizer == 'Adagrad':
            opt = tf.keras.optimizers.Adagrad(
                learning_rate=lr, initial_accumulator_value=iav, epsilon=eps, name='Adagrad')
        if selected_optimizer == 'Adadelta':
            opt = tf.keras.optimizers.Adadelta(
                learning_rate=lr, rho=rh, epsilon=eps, name='Adadelta')
        if selected_optimizer == 'Adam':
            opt = tf.keras.optimizers.Adam(
                learning_rate=lr, beta_1=b1, beta_2=b2, epsilon=eps, amsgrad=ams, name='Adam')
        if selected_optimizer == 'Adamax':
            opt = tf.keras.optimizers.Adamax(
                learning_rate=lr, beta_1=b1, beta_2=b2, epsilon=eps, name='Adamax')
        if selected_optimizer == 'Ftrl':
            opt = tf.keras.optimizers.Ftrl(
                learning_rate=lr, learning_rate_power=lrp, initial_accumulator_value=iav,
                l1_regularization_strength=l1rs, l2_regularization_strength=l2rs,
                name='Ftrl', l2_shrinkage_regularization_strength=l2srs, beta=b0)
        if selected_optimizer == 'Nadam':
            opt = tf.keras.optimizers.Nadam(
                learning_rate=lr, beta_1=b1, beta_2=b2, epsilon=eps, name='Nadam')
        if selected_optimizer == 'RMSprop':
            opt = tf.keras.optimizers.RMSprop(
                learning_rate=lr, rho=rh, momentum=mom, epsilon=eps, centered=cen, name='RMSprop')
        if selected_optimizer == 'SGD':
            opt = tf.keras.optimizers.SGD(
                learning_rate=lr, momentum=mom, nesterov=nes, name='SGD')
        #################################################################################
        # Metrics
        #################################################################################
        # Metric variables
        metrics_list = []
        meaniou_num_classes = 2

        def mean_metric_wrapper_function(y_true, y_pred):
            return tf.cast(tf.math.equal(y_true, y_pred), tf.float32)
        # Must be the same size as predictions
        mean_relative_error_normalizer = [1, 2, 3, 4]
        recall = 0.5  # A scalar value in range [0, 1]
        precision = 0.5  # A scalar value in range [0, 1]
        specificity = 0.5  # A scalar value in range [0, 1]
        sensitivity = 0.5  # A scalar value in range [0, 1]
        # Metric Classes
        if 'Auc' in selected_metrics:
            metrics_list.append(tf.keras.metrics.AUC(
                num_thresholds=200, curve='ROC',
                summation_method='interpolation', name=None, dtype=None,
                thresholds=None, multi_label=False, num_labels=None, label_weights=None,
                from_logits=False))
        if 'accuracy' in selected_metrics:
            met = metrics_list.append(tf.keras.metrics.Accuracy(
                name='accuracy', dtype=None))
        if 'binary_accuracy' in selected_metrics:
            metrics_list.append(tf.keras.metrics.BinaryAccuracy(
                name='binary_accuracy', dtype=None, threshold=0.5))
        if 'binary_crossentropy' in selected_metrics:
            metrics_list.append(tf.keras.metrics.BinaryCrossentropy(
                name='binary_crossentropy', dtype=None, from_logits=False,
                label_smoothing=0))
        if 'categorical_accuracy' in selected_metrics:
            metrics_list.append(tf.keras.metrics.CategoricalAccuracy(
                name='categorical_accuracy', dtype=None))
        if 'categorical_crossentropy' in selected_metrics:
            metrics_list.append(tf.keras.metrics.CategoricalCrossentropy(
                name='categorical_crossentropy', dtype=None, from_logits=False,
                label_smoothing=0))
        if 'categorical_hinge' in selected_metrics:
            metrics_list.append(tf.keras.metrics.CategoricalHinge(
                name='categorical_hinge', dtype=None))
        if 'cosine_similarity' in selected_metrics:
            metrics_list.append(tf.keras.metrics.CosineSimilarity(
                name='cosine_similarity', dtype=None, axis=-1))
        if 'Fn' in selected_metrics:
            metrics_list.append(tf.keras.metrics.FalseNegatives(
                thresholds=None, name=None, dtype=None))
        if 'Fp' in selected_metrics:
            metrics_list.append(tf.keras.metrics.FalsePositives(
                thresholds=None, name=None, dtype=None))
        if 'hinge' in selected_metrics:
            metrics_list.append(tf.keras.metrics.Hinge(
                name='hinge', dtype=None))
        if 'kullback_leibler_divergence' in selected_metrics:
            metrics_list.append(tf.keras.metrics.KLDivergence(
                name='kullback_leibler_divergence', dtype=None))
        if 'logcosh' in selected_metrics:
            metrics_list.append(tf.keras.metrics.LogCoshError(
                name='logcosh', dtype=None))
        if 'mean' in selected_metrics:
            metrics_list.append(tf.keras.metrics.Mean(
                name='mean', dtype=None))
        if 'mean_absolute_error' in selected_metrics:
            metrics_list.append(tf.keras.metrics.MeanAbsoluteError(
                name='mean_absolute_error', dtype=None))
        if 'mean_absolute_percentage_error' in selected_metrics:
            metrics_list.append(tf.keras.metrics.MeanAbsolutePercentageError(
                name='mean_absolute_percentage_error', dtype=None))
        if 'meaniou' in selected_metrics:
            metrics_list.append(tf.keras.metrics.MeanIoU(
                num_classes=meaniou_num_classes, name=None, dtype=None))
        if 'mean_metric_wrapper' in selected_metrics:
            metrics_list.append(tf.keras.metrics.MeanMetricWrapper(
                fn=mean_metric_wrapper_function, name=None, dtype=None))
        if 'mean_relative_error' in selected_metrics:
            metrics_list.append(tf.keras.metrics.MeanRelativeError(
                normalizer=mean_relative_error_normalizer, name=None, dtype=None))
        if 'mean_squared_error' in selected_metrics:
            metrics_list.append(tf.keras.metrics.MeanSquaredError(
                name='mean_squared_error', dtype=None))
        if 'mean_squared_logarithmic_error' in selected_metrics:
            metrics_list.append(tf.keras.metrics.MeanSquaredLogarithmicError(
                name='mean_squared_logarithmic_error', dtype=None))
        if 'mean_tensor' in selected_metrics:
            metrics_list.append(tf.keras.metrics.MeanTensor(
                name='mean_tensor', dtype=None, shape=None))
        if 'metric' in selected_metrics:
            metrics_list.append(tf.keras.metrics.Metric(
                name=None, dtype=None))
        if 'poisson' in selected_metrics:
            metrics_list.append(tf.keras.metrics.Poisson(
                name='poisson', dtype=None))
        if 'precision' in selected_metrics:
            metrics_list.append(tf.keras.metrics.Precision(
                thresholds=None, top_k=None, class_id=None, name=None, dtype=None))
        if 'precision_at_recall' in selected_metrics:
            metrics_list.append(tf.keras.metrics.PrecisionAtRecall(
                recall, num_thresholds=200, class_id=None, name=None, dtype=None))
        if 'recall' in selected_metrics:
            metrics_list.append(tf.keras.metrics.Recall(
                thresholds=None, top_k=None, class_id=None, name=None, dtype=None))
        if 'recall_at_precision' in selected_metrics:
            metrics_list.append(tf.keras.metrics.RecallAtPrecision(
                precision, num_thresholds=200, class_id=None, name=None, dtype=None))
        if 'root_mean_squared_error' in selected_metrics:
            metrics_list.append(tf.keras.metrics.RootMeanSquaredError(
                name='root_mean_squared_error', dtype=None))
        if 'sensitivity_at_specificity' in selected_metrics:
            metrics_list.append(tf.keras.metrics.SensitivityAtSpecificity(
                specificity, num_thresholds=200, class_id=None, name=None, dtype=None))
        if 'sparse_categorical_accuracy' in selected_metrics:
            metrics_list.append(tf.keras.metrics.SparseCategoricalAccuracy(
                name='sparse_categorical_accuracy', dtype=None))
        if 'sparse_top_k_categorical_accuracy' in selected_metrics:
            metrics_list.append(tf.keras.metrics.SparseTopKCategoricalAccuracy(
                k=5, name='sparse_top_k_categorical_accuracy', dtype=None))
        if 'specificty_at_sensitivity' in selected_metrics:
            metrics_list.append(tf.keras.metrics.SpecificityAtSensitivity(
                sensitivity, num_thresholds=200, class_id=None, name=None, dtype=None))
        if 'squared_hinge' in selected_metrics:
            metrics_list.append(tf.keras.metrics.SquaredHinge(
                name='squared_hinge', dtype=None))
        if 'sum' in selected_metrics:
            metrics_list.append(tf.keras.metrics.Sum(
                name='sum', dtype=None))
        if 'top_k_categorical_accuracy' in selected_metrics:
            metrics_list.append(tf.keras.metrics.TopKCategoricalAccuracy(
                k=5, name='top_k_categorical_accuracy', dtype=None))
        if 'Tn' in selected_metrics:
            metrics_list.append(tf.keras.metrics.TrueNegatives(
                thresholds=None, name=None, dtype=None))
        if 'Tp' in selected_metrics:
            metrics_list.append(tf.keras.metrics.TruePositives(
                thresholds=None, name=None, dtype=None))
        # Custom Metrics
        if 'custom_mse_metric' in selected_metrics:
            metrics_list.append(custom_mse_metric)
        if 'custom_sharpe_metric' in selected_metrics:
            metrics_list.append(custom_sharpe_metric)
        if 'custom_information_metric' in selected_metrics:
            metrics_list.append(custom_information_metric)
        if 'custom_hp_metric' in selected_metrics:
            metrics_list.append(custom_hp_metric)
        if 'custom_capm_metric' in selected_metrics:
            metrics_list.append(custom_mse_metric())
        #################################################################################
        # Loss weights
        #################################################################################
        # Optional list or dictionary specifying scalar coefficients (Python floats) to
        # weight the loss contributions of different model outputs. The loss value that
        # will be minimized by the model will then be the weighted sum of all individual
        # losses, weighted by the loss_weights coefficients. If a list, it is expected
        # to have a 1:1 mapping to the model's outputs. If a dict, it is expected to map
        # output names (strings) to scalar coefficients.
        lw = None
        #################################################################################
        # Weighted Metrics
        #################################################################################
        # List of metrics to be evaluated and weighted by sample_weight or class_weight
        # during training and testing.
        wm = None
        #################################################################################
        # Run eagerly
        #################################################################################
        # Bool. Defaults to False. If True, this Model's logic will not be wrapped in a
        # tf.function. Recommended to leave this as None unless your Model cannot be run
        # inside a tf.function. run_eagerly=True is not supported when using
        # tf.distribute.experimental.ParameterServerStrategy.
        regly = None
        #################################################################################
        # Steps_per_execution
        #################################################################################
        # Int. Defaults to 1. The number of batches to run during each tf.function call.
        # Running multiple batches inside a single tf.function call can greatly improve
        # performance on TPUs or small models with a large Python overhead. At most,
        # one full epoch will be run each execution. If a number larger than the size
        # of the epoch is passed, the execution will be truncated to the size of the
        # epoch. Note that if steps_per_execution is set to N, Callback.on_batch_begin
        # and Callback.on_batch_end methods will only be called every N batches
        # (i.e. before/after each tf.function execution).
        spe = None
        #################################################################################
        # Losses
        #################################################################################
        models = []
        losses = []
        metrics_storage = []
        other_metrics_storage = []
        for selected_loss in selected_losses:
            # Change back as only building and testing three models (tabed)
            if selected_loss in ['mean_squared_error', 'custom_mse', 'custom_hp']:
                # Loss variables
                red = 'auto'
                flt = True
                ls = 0.0
                ax = -1
                dta = 1.0
                # Loss classes
                if selected_loss == 'binary_crossentropy':
                    lf = tf.keras.losses.BinaryCrossentropy(
                        from_logits=flt, label_smoothing=ls, axis=ax, reduction=red, name='binary_crossentropy')
                if selected_loss == 'categorical_crossentropy':
                    lf = tf.keras.losses.CategoricalCrossentropy(
                        from_logits=flt, label_smoothing=ls, axis=ax, reduction=red, name='categorical_crossentropy')
                if selected_loss == 'cosine_similarity':
                    lf = tf.keras.losses.CosineSimilarity(
                        axis=-1, reduction=red, name='cosine_similarity')
                if selected_loss == 'hinge':
                    lf = tf.keras.losses.Hinge(reduction=red, name='hinge')
                if selected_loss == 'huber_loss':
                    lf = tf.keras.losses.Huber(
                        delta=dta, reduction=red, name='huber_loss')
                # loss = y_true * log(y_true / y_pred)
                if selected_loss == 'kl_divergence':
                    lf = tf.keras.losses.KLDivergence(
                        reduction=red, name='kl_divergence')
                # logcosh = log((exp(x) + exp(-x))/2), where x is the error y_pred - y_true.
                if selected_loss == 'log_cosh':
                    lf = tf.keras.losses.LogCosh(
                        reduction=red, name='log_cosh')
                if selected_loss == 'loss':
                    lf = tf.keras.losses.Loss(reduction=red, name=None)
                # loss = abs(y_true - y_pred)
                if selected_loss == 'mean_absolute_error':
                    lf = tf.keras.losses.MeanAbsoluteError(
                        reduction=red, name='mean_absolute_error')
                # loss = 100 * abs(y_true - y_pred) / y_true
                if selected_loss == 'mean_absolute_percentage_error':
                    lf = tf.keras.losses.MeanAbsolutePercentageError(
                        reduction=red, name='mean_absolute_percentage_error')
                # loss = square(y_true - y_pred)
                if selected_loss == 'mean_squared_error':
                    lf = tf.keras.losses.MeanSquaredError(
                        reduction=red, name='mean_squared_error')
                # loss = square(log(y_true + 1.) - log(y_pred + 1.))
                if selected_loss == 'mean_squared_logarithmic_error':
                    lf = tf.keras.losses.MeanSquaredLogarithmicError(
                        reduction=red, name='mean_squared_logarithmic_error')
                # loss = y_pred - y_true * log(y_pred)
                if selected_loss == 'poisson':
                    lf = tf.keras.losses.Poisson(reduction=red, name='poisson')
                if selected_loss == 'sparse_categorical_crossentropy':
                    lf = tf.keras.losses.SparseCategoricalCrossentropy(
                        from_logits=flt, reduction=red, name='sparse_categorical_crossentropy')
                # loss = square(maximum(1 - y_true * y_pred, 0))
                if selected_loss == 'squared_hinge':
                    lf = tf.keras.losses.SquaredHinge(
                        reduction=red, name='squared_hinge')
                # Custom Losses
                if selected_loss == 'custom_mse':
                    lf = custom_mse(extra_tensor=None, reduction=red,
                                    name='custom_mse')
                if selected_loss == 'custom_hp':
                    lf = custom_hp(extra_tensor=None, reduction=red,
                                   name='custom_hp')
                if selected_loss == 'custom_sharpe':
                    lf = custom_sharpe(extra_tensor=None, reduction=red,
                                       name='custom_sharpe')
                if selected_loss == 'custom_information':
                    lf = custom_information(extra_tensor=None, reduction=red,
                                            name='custom_information')
                if selected_loss == 'custom_treynor':
                    lf = custom_treynor(extra_tensor=None, reduction=red,
                                        name='custom_treynor')
                if selected_loss == 'custom_hp_mse':
                    lf = custom_hp_mse(extra_tensor=None, reduction=red,
                                       name='custom_hp_mse')
                if selected_loss == 'custom_sharpe_mse':
                    lf = custom_sharpe_mse(extra_tensor=None, reduction=red,
                                           name='custom_sharpe_mse')
                if selected_loss == 'custom_information_mse':
                    lf = custom_information_mse(extra_tensor=None, reduction=red,
                                                name='custom_information_mse')
                #################################################################################
                # Compiler
                #################################################################################
                # Compiler variables
                # Establishes the compiler
                print('Start: Model Compilation')
                print('Metrics List', metrics_list)
                model.compile(
                    optimizer=opt, loss=lf, metrics=metrics_list, loss_weights=lw,
                    weighted_metrics=wm, run_eagerly=regly, steps_per_execution=spe)
                print('End: Model Compilation')
                #################################################################################
                # Visualise model (https://www.tensorflow.org/api_docs/python/tf/keras/utils/plot_model)
                #################################################################################
                # Visualisation variables
                to_file = '/home/connormcdowall/finance-honours/results/plots/tensorflow-visualisations/' + \
                    model_name + '.png'
                show_shapes = True
                show_dtype = False
                show_layer_names = True
                rankdir = 'TB'  # TB (Top Bottom), LR (Left Right)
                expand_nested = False
                dpi = 96
                layer_range = None
                show_layer_activations = False
                # Creates a plot of the model
                tf.keras.utils.plot_model(model, to_file, show_shapes, show_dtype,
                                          show_layer_names, rankdir, expand_nested, dpi, layer_range, show_layer_activations)
                # Prints a summary of the model
                print('Model Summary')
                print(model.summary())
                #################################################################################
                # Model.fit (https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit)
                #################################################################################
                # Fit variables
                x_train = train_dataset
                y = None  # If x is a dataset, generator, or keras.utils.Sequence instance, y should
                # not be specified (since targets will be obtained from x).
                batch_size = None  # Defaults to 32
                # Integer. Number of epochs to train the model. An epoch is an iteration over  - Try running model with model epochs.
                eps = 30
                # the entire x and y data provided (unless the steps_per_epoch flag is set to something other than None).
                verbose = 'auto'
                callbacks = None
                validation_split = 0.0  # Not support when x is a dataset
                validation_data = val_dataset
                # Ignored when x is a generator or an object of tf.data.Dataset (This case)
                shuffle = True
                # Optional dictionary mapping class indices (integers) to a
                class_weight = None
                # continued: weight (float) value, used for weighting the loss function (during training only)
                sample_weight = None  # This argument is not supported when x is a dataset
                # Integer. Epoch at which to start training (useful for resuming a previous training run).
                initial_epoch = 0
                # If x is a tf.data dataset, and 'steps_per_epoch' is None, the epoch will run until the input dataset is exhausted.
                steps_per_epoch = None
                # Only relevant if validation_data is provided and is a tf.data dataset.
                validation_steps = None
                # Continued: If 'validation_steps' is None, validation will run until the validation_data dataset is exhausted.
                # Do not specify the validation_batch_size if your data is in the form of datasets
                validation_batch_size = None
                validation_freq = 1
                # Integer. Used for generator or keras.utils.Sequence input only.
                max_queue_size = 10
                # Continued: Maximum size for the generator queue. If unspecified, max_queue_size will default to 10.
                # Integer. Used for generator or keras.utils.Sequence input only (Not this case)
                workers = 1
                # Boolean. Used for generator or keras.utils.Sequence input only.
                use_multiprocessing = False
                # Sets Neptune ai
                project = "connormcdowall/finance-honours"
                api_token = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI4YzBmOTFlNS0zZTFiLTQyNDUtOGFjZi1jZGI0NDY4ZGVkOTQifQ=="
                # Configures neptune ai
                neptune_cbk = configure_training_ui(project, api_token)
                # Fit the model
                print('Start: Model Fitting')
                # Sets up early stopping callback to accompany neptune.ai callback
                early_stop_callback = tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss', mode="min", patience=5, restore_best_weights=True)
                # Fits model
                history = model.fit(x=x_train, batch_size=128, epochs=eps,
                                    verbose='auto', validation_data=val_dataset, callbacks=[neptune_cbk, early_stop_callback])
                print('End: Model Fitting')
                # Creates learning curves
                try:
                    create_learning_curves(
                        model_name, selected_loss, history, from_load_file=False)
                    print('Successfully printed learning curves')
                except:
                    print('Failed to print learning curves')
                # Save the model history
                try:
                    with open('/home/connormcdowall/finance-honours/results/models/history/' +
                              model_name + '-' + selected_loss, 'wb') as f:
                        pickle.dump(history.history, f)
                        f.close()
                        print(
                            'Complete: Pickled history')
                except:
                    print('Incomplete: Failed to pickle hisotry')
                #################################################################################
                # Model.evaluate (https://www.tensorflow.org/api_docs/python/tf/keras/Model#evaluate)
                #################################################################################
                # Evaluation variables
                x_test = test_dataset
                # Only use if target variables not specified in dataset, must align with x.
                y = None
                batch_size = None  # Defaults to 32
                # 0 or 1. Verbosity mode. 0 = silent, 1 = progress bar.
                verb = 1
                sample_weight = None  # Optional, This argument is not supported when x is a dataset
                steps = None  # If x is a tf.data dataset and steps is None, 'evaluate' will run until the dataset is exhausted
                callbacks = None
                mqs = 10  # Max queue size. If unspecified, max_queue_size will default to 10
                workers = 1  # Integer. Used for generator or keras.utils.Sequence
                # use_multiprocessing, boolean. Used for generator or keras.utils.Sequence input only.
                ump = False
                # Continued: If True, use process-based threading. If unspecified, use_multiprocessing will default to False.
                rd = False  # If True, loss and metric results are returned as a dict,
                # with each key being the name of the metric. If False, they are returned as a list.
                # Model evaluation
                print('Start: Model Evaluation')
                loss, metrics, *other_metrics = model.evaluate(x_test, batch_size=None, verbose=verb, steps=None, callbacks=None,
                                                               max_queue_size=mqs, workers=1, use_multiprocessing=ump, return_dict=rd)
                #################################################################################
                print('End: Model Evaluation')
                print("Loss: ", loss)
                print("Metric Descriptions: ", model.metrics_names)
                print("Metric Values: ", metrics)
                # Save the model
                model.save(
                    '/home/connormcdowall/finance-honours/results/models/tensorflow/'+model_name + '-' + selected_loss + '-active')
                # Monitor memory usage
                monitor_memory_usage(units=3, cpu=True, gpu=True)
                models.append(model)
                losses.append(loss)
                metrics_storage.append(metrics)
                other_metrics_storage.append(other_metrics)
            # Return the model, loss and accuracy
        return models, losses, metrics_storage, other_metrics_storage
    else:
        # Exemplar implementation prior to finance adaptation
        # Set up neural net layers
        x = tf.keras.layers.Dense(32, activation="relu")(all_features)
        x = tf.keras.layers.Dropout(rate=0.5, noise_shape=None, seed=None)(x)
        output = tf.keras.layers.Dense(1)(x)
        # Configure the model
        model = tf.keras.Model(all_inputs, output)
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.BinaryCrossentropy(
                          from_logits=True),
                      metrics=["accuracy"])
        # Visualise the model via a connectivity graph
        tf.keras.utils.plot_model(model, show_shapes=True, rankdir="LR")
        # Train the model
        model.fit(train_dataset, epochs=10, validation_data=val_dataset)
        # Test the model
        loss, accuracy = model.evaluate(test_dataset)
        print("Loss: ", loss)
        print("Accuracy: ", accuracy)
        # Save the model
        model.save('results/plots/tensorflow-models/'+model_name+'.pb')
        # Return the model, loss and accuracy
        return model, loss, accuracy


def create_tensorflow_models(data_vm_directory, list_of_columns, categorical_assignment, target_column, chunk_size, resizing_options, batch_size, model_name, selected_optimizer, selected_losses, selected_metrics, split_data=False, trial=False, sample=False):
    """Creates the tensorflow models combining all the analysis

    Args:
        data_vm_directory (str): Directory of source data
        list_of_columns (str): Directory to txt file with list of columns
        categorical_assignment (dict): Dictionary of features to be categorical
        target_column (str): Target column
        chunk_size (int): Chunk size 
        resizing_options (list): List of boolean variables for resizgin the dataset
        batch_size (int): Batch size for creating tensor slices
        model_name (str): Run name
        selected_optimizer (str): optimizer to use
        selected_losses (str): Loss function to use
        selected_metrics (list): List of sleection metrics to use
        split_data (bool, optional): Boolean to split the original dataset. Defaults to False.
        trial (bool, optional): Boolean to taek a smaller dataset. Defaults to False.
        sample (bool, optional): Boolean to take an even smaller dataset. Defaults to False.
    """
    # Prints memory usage before analysis
    monitor_memory_usage(units=3, cpu=True, gpu=True)
    # Reset working textfile if resizing used for numerical encoding
    # Clear the working file
    file = open(
        "/home/connormcdowall/finance-honours/data/working-columns.txt", "r+")
    file.truncate(0)
    file.close()
    # Set boolean to control which dataset
    # Tranfer file lines
    with open("/home/connormcdowall/finance-honours/data/dataframe-columns.txt", "r") as f1:
        with open("/home/connormcdowall/finance-honours/data/working-columns.txt", "w") as f2:
            for line in f1:
                f2.write(line)
    # Split the initial vm dataset
    if split_data:
        split_vm_dataset(data_vm_directory, create_statistics=False,
                         split_new_data=True, create_validation_set=True)
    # Creates the training, validation and testing dataframes
    use_chronological_data = True
    if use_chronological_data:
        train_df = pd.read_stata(data_vm_directory + 'active_train.dta')
        print('Training Set')
        print(train_df.head())
        print()
        val_df = pd.read_stata(data_vm_directory + 'active_val.dta')
        print('Validation Set')
        print(val_df.head())
        test_df = pd.read_stata(data_vm_directory + 'active_test.dta')
        print('Testing Set')
        print(test_df.head())
        print('Dataframe Columns')
        print(train_df.columns)
    else:
        test_df = process_vm_dataset(data_vm_directory + 'test.dta', chunk_size,
                                     resizing_options, save_statistics=False, sample=sample)
        train_df = process_vm_dataset(data_vm_directory + 'train.dta',
                                      chunk_size, resizing_options, save_statistics=False, sample=sample)
        val_df = process_vm_dataset(data_vm_directory + 'val.dta', chunk_size,
                                    resizing_options, save_statistics=False, sample=sample)
    # Recreate smaller factors list
    small_factors = True
    if small_factors:
        create_original_list_of_columns(train_df)
    # Use trial to test the dataframe when functions not as large
    if trial:
        # Trial run takes 5% of dataframe produced from processed vm datasets
        test_df, test_discard_df = train_test_split(test_df, test_size=0.95)
        train_df, train_discard_df = train_test_split(train_df, test_size=0.95)
        val_df, val_discard_df = train_test_split(val_df, test_size=0.95)
    # Drop level_0 again from the datasets
    train_df = train_df.drop(columns=['level_0'])
    val_df = val_df.drop(columns=['level_0'])
    test_df = test_df.drop(columns=['level_0'])
    # Replace any NaNs in the dataset
    train_df = replace_nan(train_df, replacement_method=3)
    val_df = replace_nan(val_df, replacement_method=3)
    test_df = replace_nan(test_df, replacement_method=3)
    print('NaNs in Training Set: ', train_df.isna().sum().sum())
    print('NaNs in Validation Set: ', val_df.isna().sum().sum())
    print('NaNs in Testing Set: ', test_df.isna().sum().sum())
    # Saves descriptions, information, and datasets of the created dataframes
    statistics_location = '/home/connormcdowall/finance-honours/results/statistics'
    data_location = '/home/connormcdowall/finance-honours/data/dataframes'
    save_df_statistics(test_df, 'test', statistics_location, data_location)
    save_df_statistics(train_df, 'train', statistics_location, data_location)
    save_df_statistics(val_df, 'validation',
                       statistics_location, data_location)

    # Prints test of the dataframe
    print(train_df.info(verbose=True))
    print(val_df.info(verbose=True))
    print(test_df.info(verbose=True))
    # Create feature lists for deep learning
    numerical_features, categorical_features = create_feature_lists(
        list_of_columns, categorical_assignment)
    # Creates the categorical dictonary (must specify the variables types of each)
    categorical_dictionary = dict.fromkeys(categorical_features, 'string')
    category_dtypes = {'size_grp': 'string', 'permno': 'int32', 'permco': 'int32', 'crsp_shrcd': 'int8',
                       'crsp_exchcd': 'int8', 'adjfct': 'float64', 'sic': 'float64', 'ff49': 'float64'}
    for key in category_dtypes:
        categorical_dictionary[key] = category_dtypes[key]
    # Encodes the tensorflow matrix
    require_encoding = True
    if require_encoding:
        all_features, all_inputs, train_dataset, val_dataset, test_dataset = encode_tensor_flow_features(
            train_df, val_df, test_df, target_column, numerical_features, categorical_features, categorical_dictionary, size_of_batch=batch_size)
    else:
        print('Loads datasets and encoded features from file')
    # Note: Keep Stochastic Gradient Descent as Optimizer for completeness
    # Buids tensorflow model
    model, loss, metrics, other_metrics = build_tensor_flow_model(train_dataset, val_dataset, test_dataset, model_name,
                                                                  all_features, all_inputs, selected_optimizer, selected_losses, selected_metrics, finance_configuration=True)
    return


def create_learning_curves(model_name, selected_loss, model_history=None, from_load_file=True):
    """Creates learning curves to model training losses

    Args:
        model_name (str): Run name
        selected_loss (str): Selected loss function
        model_history (str, optional): Load a model history. Defaults to None.
        from_load_file (bool, optional): Load from a file instead. Defaults to True.
    """
    # Set destination dictionary
    destination_directory = '/home/connormcdowall/finance-honours/results/plots/learning-curves/'
    history_path = '/home/connormcdowall/finance-honours/results/models/history/'
    path = history_path + model_name + '-' + selected_loss
    if from_load_file:
        with open(path, "rb") as f:
            model_history = pickle.load(f)
            f.close()
    else:
        model_history = model_history.history
    if selected_loss in ['mean_squared_error', 'custom_mse', 'custom_hp']:
        path = history_path + model_name + '-' + selected_loss
        # model = pickle.load(open(path, "rb"))
        # Create learning curves
        plt.plot(model_history['loss'], label=' Training')
        plt.plot(model_history['val_loss'], label='Validation')
        plt.title(selected_loss)
        plt.ylabel('Losses')
        plt.xlabel('No. epoch')
        plt.legend(loc="upper left")
        plt.savefig(destination_directory + model_name +
                    '-' + selected_loss + '.png')
        plt.clf()
        return


def make_tensorflow_predictions(model_name, model_directory, selected_losses, dataframe_location, custom_objects):
    """ Makes tensorflo predictions

    Args:
        model_name (str): Run name
        model_directory (str): Model directory
        selected_losses (str): Loss function
        dataframe_location (str): Directory to the government
        custom_objects (list): List of custom objects in the tensorflow model
    """
    # Initialises new dataframe
    column_names = ['size_grp', "mth", "predict", 'ret_exc_lead1m', 'permno']
    model_locations = []
    # Sets model directory based on loss
    for loss in selected_losses:
        model_filepath = model_directory + '-' + loss
        model_locations.append(model_filepath)
    # Loads the dictionary
    df = pd.read_stata(dataframe_location)
    df = replace_nan(df, replacement_method=3)
    # Displays key dataframe information
    print('Dataframe Information')
    print(df.head())
    print(df.info(verbose=False))
    print('The length of the dataframe is: ', len(df['ret_exc_lead1m']))
    # Convert dataframe row to dictionary with column headers (section)
    dataframe_dictionary = df.to_dict(orient="records")
    # Starts fo loop to loop through every model
    print('Starting: Loss Function Predictions')
    # Loads all models
    print('Loading Tensorflow Models...')
    mse_tf_model = tf.keras.models.load_model(
        filepath=model_locations[0], custom_objects=custom_objects)
    print('Loaded: Mean Square Error (Tensorflow)')
    mse_model = tf.keras.models.load_model(
        filepath=model_locations[1], custom_objects=custom_objects)
    print('Loaded: Mean Square Error')
    sharpe_model = tf.keras.models.load_model(
        filepath=model_locations[2], custom_objects=custom_objects)
    print('Loaded: Sharpe Ratio')
    sharpe_mse_model = tf.keras.models.load_model(
        filepath=model_locations[3], custom_objects=custom_objects)
    print('Loaded: Sharpe Ratio (MSE)')
    information_model = tf.keras.models.load_model(
        filepath=model_locations[4], custom_objects=custom_objects)
    print('Loaded: Information Ratio')
    hp_model = tf.keras.models.load_model(
        filepath=model_locations[5], custom_objects=custom_objects)
    print('Loaded: Hedge Portfolio')
    models = [mse_tf_model, mse_model, sharpe_model,
              sharpe_mse_model, information_model, hp_model]
    # Creates learning curves plots

    # Resets df predictions dataframe
    print('Setting Prediction Dataframe')
    mse_df_predictions = pd.DataFrame(columns=column_names)
    mse_tf_df_predictions = pd.DataFrame(columns=column_names)
    sharpe_df_predictions = pd.DataFrame(columns=column_names)
    sharpe_mse_df_predictions = pd.DataFrame(columns=column_names)
    information_df_predictions = pd.DataFrame(columns=column_names)
    hp_df_predictions = pd.DataFrame(columns=column_names)
    # hp_mse_df_predictions = pd.DataFrame(columns=column_names)
    # Stores dataframes in an array
    df_predictions = [mse_df_predictions,
                      mse_tf_df_predictions,
                      sharpe_df_predictions,
                      sharpe_mse_df_predictions,
                      information_df_predictions,
                      hp_df_predictions]
    # Initialises row count
    row_count = 0
    count = 0
    check = 0
    # Make predictions from the entire dataframe at once
    all_predictions = True
    if all_predictions:
        # Converts dataframe to dataset
        ds = create_tf_dataset(
            df, target_column='ret_exc_lead1m', shuffle=False, batch_size=256)
        print('Completed: Creating Dataset')
        for i in range(len(models)):
            predictions = models[i].predict(ds)
            print('Completed: Mean Square Error - Tensorflow')
            print(predictions)
            print('The total number of predictions is {}.'.format(len(predictions)))
            # Creates new dataframe at once
            scaler_predictions = []
            for column in column_names:
                if column == 'predict':
                    # Assigns the predictions
                    for j in range(len(predictions)):
                        scaler_predictions.append(np.asscalar(predictions[j]))
                    df_predictions[i][column] = scaler_predictions
                else:
                    df_predictions[i][column] = df[column].to_list()
            print('New Dataframe')
            print(mse_df_predictions.head())
            # Saves the model to file
            df_predictions[i].to_csv('/home/connormcdowall/finance-honours/results/predictions/' +
                                     model_name + '-' + selected_losses[i] + '.csv')
        return
    # Makes predictions per row on the dataframe (Computationally and temporally expensive)
    print('Start to make predictions per row')
    for row in dataframe_dictionary:
        # Suspect the this convert to tensor function is time intensive
        input_dict = {name: tf.convert_to_tensor(
            [value]) for name, value in row.items()}
        # Makes the model predictions
        mse_predictions = mse_model.predict(input_dict)
        mse_tf_predictions = mse_tf_model.predict(input_dict)
        sharpe_predictions = sharpe_model.predict(input_dict)
        sharpe_mse_predictions = sharpe_mse_model.predict(input_dict)
        information_predictions = information_model.predict(input_dict)
        hp_predictions = hp_model.predict(input_dict)
        # hp_mse_predictions = hp_mse_model.predict(input_dict)
        # Stores predictions in an array
        predictions = [mse_predictions[0],
                       mse_tf_predictions[0],
                       sharpe_predictions[0],
                       sharpe_mse_predictions[0],
                       information_predictions[0],
                       hp_predictions[0]]
        count = count + 1
        for i in range(len(predictions)):
            # Adds prediction value to prediction df
            new_df_row = {'size_grp': row['size_grp'], "mth": int(row['mth']),
                          "predict": np.asscalar(predictions[i]), 'ret_exc_lead1m': row['ret_exc_lead1m'], 'permno': row['permno']}
            df_predictions[i] = df_predictions[i].append(
                new_df_row, ignore_index=True)
        row_count = row_count + 1
        print('Completed row {} for all Loss functions.'.format(row_count))
        # Use the count to make sure the function is working properly (remove once tested)
        # Sets up incremental saving given the size of the dataframe
        checkpoints = [1000, 5000, 10000, 50000, 100000,
                       200000, 300000, 400000, 500000, len(df)]
        if row_count == checkpoints[check]:
            print('Saving {} predictions for all models.'.format(
                checkpoints[check]))
            for j in range(len(df_predictions)):
                print(df_predictions[j].info(verbose=True))
                print(df_predictions[j].head())
            # Saves the model predictions to file (model_locations and selected losses alogn for these purposes)
                df_predictions[j].to_csv('/home/connormcdowall/local-data/' +
                                         model_name + '-' + selected_losses[j] + '-' + str(checkpoints[check]) + '-rows.csv')
            check = check + 1
    print('Completed: Loss Function Predictions')
    return


def perform_tensorflow_model_inference(model_name, sample):
    """ Perform evaluations from model (must be configured)

    Args:
        model_name ([type]): [description]
        sample ([type]): [description]

    Returns:
        [type]: [description]
    """
    reloaded_model = tf.keras.models.load_model(model_name)
    input_dict = {name: tf.convert_to_tensor(
        [value]) for name, value in sample.items()}
    predictions = reloaded_model.predict(input_dict)
    print('Predction; ', predictions)
    # prob = tf.nn.sigmoid(predictions[0])
    return predictions


def implement_test_data(dataframe, train, val, test, full_implementation=False):
    """[summary]

    Args:
        dataframe (df): Dataframe
        train (df): Training dataframe
        val (df): Validation dataframe
        test (df): Testing dataframe
        full_implementation (bool, optional): Implement a full implementation. Defaults to False.
    """
    # Sets the batch size
    target_column = 'target'
    batch_size = 5
    train_ds = create_tf_dataset(
        train, target_column, shuffle=True, batch_size=batch_size)
    # See arrangement of the data
    [(train_features, label_batch)] = train_ds.take(1)
    print('Every feature:', list(train_features.keys()))
    print('A batch of ages:', train_features['Age'])
    print('A batch of targets:', label_batch)
    # Test the get_normalisation function
    photo_count_col = train_features['PhotoAmt']
    layer = get_normalization_layer('PhotoAmt', train_ds)
    layer(photo_count_col)
    # Test the get category encoding layer function
    test_type_col = train_features['Type']
    test_type_layer = get_category_encoding_layer(name='Type',
                                                  dataset=train_ds,
                                                  dtype='string')
    test_type_layer(test_type_col)
    test_age_col = train_features['Age']
    test_age_layer = get_category_encoding_layer(name='Age',
                                                 dataset=train_ds,
                                                 dtype='int64',
                                                 max_tokens=5)
    test_age_layer(test_age_col)
    # Continues with a full implementation if necessary
    if full_implementation:
        print("Continues with full implementation")
        numerical_features = ['PhotoAmt', 'Fee']
        categorical_features = ['Age', 'Type', 'Color1', 'Color2', 'Gender', 'MaturitySize',
                                'FurLength', 'Vaccinated', 'Sterilized', 'Health', 'Breed1']
        # Create categorical type dictionary
        categorical_dictionary = dict.fromkeys(categorical_features, 'string')
        categorical_dictionary["Age"] = 'int64'
        model_name = 'pets_test'
        selected_optimizer = 'adam'
        selected_loss = 'binary_crossentropy'
        selected_metrics = ['accuracy']
        all_features, all_inputs, train_dataset, val_dataset, test_dataset = encode_tensor_flow_features(
            train, val, test, target_column, numerical_features, categorical_features, categorical_dictionary, size_of_batch=256)
        model, loss, metrics = build_tensor_flow_model(train_dataset, val_dataset, test_dataset, model_name,
                                                       all_features, all_inputs, selected_optimizer, selected_loss, selected_metrics, finance_configuration=False)
        # Test model inference
        sample = {
            'Type': 'Cat',
                    'Age': 3,
                    'Breed1': 'Tabby',
                    'Gender': 'Male',
                    'Color1': 'Black',
                    'Color2': 'White',
                    'MaturitySize': 'Small',
                    'FurLength': 'Short',
                    'Vaccinated': 'No',
                    'Sterilized': 'No',
                    'Health': 'Healthy',
                    'Fee': 100,
                    'PhotoAmt': 2,
        }
        prob = perform_tensorflow_model_inference(
            'results/plots/tensorflow-models/'+model_name+'.pb', sample)
    else:
        print('Test functions complete')
    return


def reinforement_learning(model, env, target_vec):
    """ Exammple to reinforcement learning

    Args:
        model (tf.model): Configured model
        env (env): Reinforcement learning environment
        target_vec (): 
    """
    discount_factor = 0.95
    eps = 0.5
    eps_decay_factor = 0.999
    num_episodes = 500

    # Implements Q Leanring Approach
    for i in range(num_episodes):
        state = env.reset()
        eps *= eps_decay_factor
        done = False
        while not done:
            if np.random.random() < eps:
                action = np.random.randint(0, env.action_space.n)
            else:
                action = np.argmax(
                    model.predict(np.identity(env.observation_space.n)[state:state + 1]))
            new_state, reward, done, _ = env.step(action)
            target = reward + discount_factor * \
                np.max(model.predict(np.identity(
                    env.observation_space.n)[new_state:new_state + 1]))
            target_vector = model.predict(
                np.identity(env.observation_space.n)[state:state + 1])[0]
            target_vector[action] = target
            model.fit(
                np.identity(env.observation_space.n)[state:state + 1],
                target_vec.reshape(-1, env.action_space.n),
                epochs=1, verbose=0)
            state = new_state
    return
#################################################################################
# Custom Loss Functions, Metrics and Autodiff Testing
#################################################################################
# Loss Functions
#################################################################################
# Key:
# 0 = Matrix of Parameters (Theta)
# X = Feature Matrix
# f_(0)(X) = Target (e.g., Excess Returns)
# V = All-Ones=Vector

# Use Tensorlow backend functions
# Symbolic Tensors do not work in function calls as require eager tensors.
# Subsequently, must create custom class with call function
# Utilisation of function closure to pass multiple inputs into the function.


class CustomLossFunctionExample(tf.keras.losses.Loss):
    """Example Custom Loss Function Class

    Args:
        tf (env): Tensorflow keras losses environment
    """
    # Example from Youtube (https://www.youtube.com/watch?v=gcwRjM1nZ4o)

    def __init__(self):
        """Initialisation of a custom loss function template
        """
        # Initialise the function
        super().__init__()

    def call(self, y_true, y_pred):
        """Call for loss function

        Args:
            y_true (tf): Tensor of realisations
            y_pred (tf): Tensor of predictions

        Returns:
            : Loss
        """
        mse = tf.reduce_mean(tf.square(y_true, y_pred))
        rmse = tf.math.sqrt(mse)
        return rmse / tf.reduce_mean(tf.square(y_true)) - 1


class custom_mse(tf.keras.losses.Loss):
    """Custom Mean Squared Error Function Class

    Args:
        tf (env): Tensorflow keras losses environment
    """

    def __init__(self, extra_tensor=None, reduction=tf.keras.losses.Reduction.AUTO, name='custom_mse'):
        """Initialisation of custom mse portfolio function

        Args:
            extra_tensor (tensor, optional): Original tensor format. Defaults to None.
            reduction (red, optional): Reduction. Defaults to tf.keras.losses.Reduction.AUTO.
            name (str, optional): name of function. Defaults to 'custom_hp'.
        """
        super().__init__(reduction=reduction, name=name)
        self.extra_tensor = extra_tensor

    def call(self, y_true, y_pred):
        """Call for mse loss function

        Args:
            y_true (tf): Tensor of realisations
            y_pred (tf): Tensor of predictions

        Returns:
            : Loss
        """
        extra_tensor = self.extra_tensor
        loss = K.mean(K.square(y_pred - y_true))
        return loss


class custom_hp(tf.keras.losses.Loss):
    """Custom Hedge Portfolio Function Class

    Args:
        tf (env): Tensorflow keras losses environment
    """

    def __init__(self, extra_tensor=None, reduction=tf.keras.losses.Reduction.AUTO, name='custom_hp'):
        """Initialisation of custom hedge portfolio function

        Args:
            extra_tensor (tensor, optional): Original tensor format. Defaults to None.
            reduction (red, optional): Reduction. Defaults to tf.keras.losses.Reduction.AUTO.
            name (str, optional): name of function. Defaults to 'custom_hp'.
        """
        super().__init__(reduction=reduction, name=name)
        self.extra_tensor = extra_tensor

    def call(self, y_true, y_pred):
        """Call for hp loss function

        Args:
            y_true (tf): Tensor of realisations
            y_pred (tf): Tensor of predictions

        Returns:
            : Loss
        """
        extra_tensor = self.extra_tensor
        # Calculates sum over vector tensors
        y_true_sum = K.sum(y_true)
        y_pred_sum = K.sum(y_pred)
        #
        y_true_weights = (y_true/y_true_sum)
        y_pred_weights = (y_pred/y_pred_sum)
        # Transpose the weights
        y_true_transposed = K.transpose(y_true_weights)
        y_pred_transposed = K.transpose(y_pred_weights)
        # Multiply by the weights
        y_true_loss = K.dot(y_true_transposed, y_true)
        y_pred_loss = K.dot(y_pred_transposed, y_pred)
        loss = -1*(y_pred_loss)
        return loss


class custom_sharpe(tf.keras.losses.Loss):
    """Custom Sharpe Ratio Function Class

    Args:
        tf (env): Tensorflow keras losses environment
    """

    def __init__(self, extra_tensor=None, reduction=tf.keras.losses.Reduction.AUTO, name='custom_sharpe'):
        """Initialisation of custom sharpe portfolio function

        Args:
            extra_tensor (tensor, optional): Original tensor format. Defaults to None.
            reduction (red, optional): Reduction. Defaults to tf.keras.losses.Reduction.AUTO.
            name (str, optional): name of function. Defaults to 'custom_sharpe'.
        """
        super().__init__(reduction=reduction, name=name)
        self.extra_tensor = extra_tensor

    def call(self, y_true, y_pred):
        """Call for sharpe loss function

        Args:
            y_true (tf): Tensor of realisations
            y_pred (tf): Tensor of predictions

        Returns:
            : Loss
        """
        extra_tensor = self.extra_tensor
        sr_pred = -1*(K.mean(y_pred)/K.std(y_pred))
        sr_true = -1*(K.mean(y_true)/K.std(y_true))
        return sr_pred


class custom_information(tf.keras.losses.Loss):
    """Custom Information Function Class

    Args:
        tf (env): Tensorflow keras losses environment
    """

    def __init__(self, extra_tensor=None, reduction=tf.keras.losses.Reduction.AUTO, name='custom_information'):
        """Initialisation of custom information portfolio function

        Args:
            extra_tensor (tensor, optional): Original tensor format. Defaults to None.
            reduction (red, optional): Reduction. Defaults to tf.keras.losses.Reduction.AUTO.
            name (str, optional): name of function. Defaults to 'custom_information'.
        """
        super().__init__(reduction=reduction, name=name)
        self.extra_tensor = extra_tensor

    def call(self, y_true, y_pred):
        """Call for information loss function

        Args:
            y_true (tf): Tensor of realisations
            y_pred (tf): Tensor of predictions

        Returns:
            : Loss
        """
        extra_tensor = self.extra_tensor
        loss = -1*((K.mean(y_pred) - K.mean(y_true))/K.std(y_pred - y_true))
        return loss


class custom_treynor(tf.keras.losses.Loss):
    """Custom Treynor Ratio Function Class

    Args:
        tf (env): Tensorflow keras losses environment
    """

    def __init__(self, extra_tensor=None, reduction=tf.keras.losses.Reduction.AUTO, name='custom_treynor'):
        """Initialisation of custom trynor portfolio function

        Args:
            extra_tensor (tensor, optional): Original tensor format. Defaults to None.
            reduction (red, optional): Reduction. Defaults to tf.keras.losses.Reduction.AUTO.
            name (str, optional): name of function. Defaults to 'custom_treynor'.
        """
        super().__init__(reduction=reduction, name=name)
        self.extra_tensor = extra_tensor

    def call(self, y_true, y_pred):
        """Call for treynor loss function

        Args:
            y_true (tf): Tensor of realisations
            y_pred (tf): Tensor of predictions

        Returns:
            : Loss
        """
        extra_tensor = self.extra_tensor
        loss = K.mean(K.square(y_pred - y_true))
        return loss


class custom_hp_mse(tf.keras.losses.Loss):
    """Custom Hedge Portfolio + Mean Squared Error Loss Function Class

    Args:
        tf (env): Tensorflow keras losses environment
    """

    def __init__(self, extra_tensor=None, reduction=tf.keras.losses.Reduction.AUTO, name='custom_hp_mse'):
        """Initialisation of custom hedge portfolio mse portfolio function

        Args:
            extra_tensor (tensor, optional): Original tensor format. Defaults to None.
            reduction (red, optional): Reduction. Defaults to tf.keras.losses.Reduction.AUTO.
            name (str, optional): name of function. Defaults to 'custom_hp_mse'.
        """
        super().__init__(reduction=reduction, name=name)
        self.extra_tensor = extra_tensor

    def call(self, y_true, y_pred):
        """Call for hp mse loss function

        Args:
            y_true (tf): Tensor of realisations
            y_pred (tf): Tensor of predictions

        Returns:
            : Loss
        """
        extra_tensor = self.extra_tensor
        # Calculates sum over vector tensors
        y_true_sum = K.sum(y_true)
        y_pred_sum = K.sum(y_pred)
        #
        y_true_weights = (y_true/y_true_sum)
        y_pred_weights = (y_pred/y_pred_sum)
        # Transpose the weights
        y_true_transposed = K.transpose(y_true_weights)
        y_pred_transposed = K.transpose(y_pred_weights)
        # Multiply by the weights
        y_true_loss = K.dot(y_true_transposed, y_true)
        y_pred_loss = K.dot(y_pred_transposed, y_pred)
        loss = K.mean(K.square(y_true_loss-y_pred_loss))
        return loss


class custom_sharpe_mse(tf.keras.losses.Loss):
    """Custom Sharpe + Mean Squared Error Function Class

    Args:
        tf (env): Tensorflow keras losses environment
    """

    def __init__(self, extra_tensor=None, reduction=tf.keras.losses.Reduction.AUTO, name='custom_sharpe_mse'):
        """Initialisation of custom sharpe mse portfolio function

        Args:
            extra_tensor (tensor, optional): Original tensor format. Defaults to None.
            reduction (red, optional): Reduction. Defaults to tf.keras.losses.Reduction.AUTO.
            name (str, optional): name of function. Defaults to 'custom_sharpe_mse'.
        """
        super().__init__(reduction=reduction, name=name)
        self.extra_tensor = extra_tensor

    def call(self, y_true, y_pred):
        """Call for sharpe mse loss function

        Args:
            y_true (tf): Tensor of realisations
            y_pred (tf): Tensor of predictions

        Returns:
            : Loss
        """
        extra_tensor = self.extra_tensor
        sr_pred_loss = -1*(K.mean(y_pred)/K.std(y_pred))
        sr_true_loss = -1*(K.mean(y_true)/K.std(y_true))
        loss = K.mean(K.square(sr_pred_loss - sr_true_loss))
        return loss


class custom_information_mse(tf.keras.losses.Loss):
    """Custom Information + Mean Squared Error Function Class

    Args:
        tf (env): Tensorflow keras losses environment
    """

    def __init__(self, extra_tensor=None, reduction=tf.keras.losses.Reduction.AUTO, name='custom_information_mse'):
        """Initialisation of custom information mse portfolio function

        Args:
            extra_tensor (tensor, optional): Original tensor format. Defaults to None.
            reduction (red, optional): Reduction. Defaults to tf.keras.losses.Reduction.AUTO.
            name (str, optional): name of function. Defaults to 'custom_information_mse'.
        """
        super().__init__(reduction=reduction, name=name)
        self.extra_tensor = extra_tensor

    def call(self, y_true, y_pred):
        """Call for information mse loss function

        Args:
            y_true (tf): Tensor of realisations
            y_pred (tf): Tensor of predictions

        Returns:
            : Loss
        """
        extra_tensor = self.extra_tensor
        loss = -1*((K.mean(y_pred) - K.mean(y_true))/K.std(y_pred - y_true))
        return loss

#################################################################################
# Metrics
#################################################################################


@ tf.function
def custom_mse_metric(y_pred, y_true):
    """Call for mse metric

    Args:
        y_true (tf): Tensor of realisations
        y_pred (tf): Tensor of predictions

    Returns:
        : Loss
    """
    metric = K.mean(K.square(y_pred - y_true))
    return metric


@ tf.function
def custom_hp_metric(y_true, y_pred):
    """Call for hp metric

    Args:
        y_true (tf): Tensor of realisations
        y_pred (tf): Tensor of predictions

    Returns:
        : Loss
    """
    y_true_sum = K.sum(y_true)
    y_pred_sum = K.sum(y_pred)
    # Calculate the weights
    y_true_weights = (y_true/y_true_sum)
    y_pred_weights = (y_pred/y_pred_sum)
    # Transpose the weights
    y_true_transposed = K.transpose(y_true_weights)
    y_pred_transposed = K.transpose(y_pred_weights)
    # Multiply by the weights
    y_true_loss = K.dot(y_true_transposed, y_true)
    y_pred_loss = K.dot(y_pred_transposed, y_pred)
    mean = (y_pred_loss)
    return mean


@ tf.function
def custom_sharpe_metric(y_true, y_pred):
    """Call for sharpe metric

    Args:
        y_true (tf): Tensor of realisations
        y_pred (tf): Tensor of predictions

    Returns:
        : Loss
    """
    # Finds Sharpe ratios of both true and predicted returns
    sr_pred = -1*(K.mean(y_pred)/K.std(y_pred))
    sr_true = -1*(K.mean(y_true)/K.std(y_true))
    # Finds MSE between predited and true MSE
    loss = K.mean(K.square(sr_true - sr_pred))
    return loss


@ tf.function
def custom_information_metric(y_true, y_pred):
    """Call for information metric

    Args:
        y_true (tf): Tensor of realisations
        y_pred (tf): Tensor of predictions

    Returns:
        : Loss
    """
    loss = -1*((K.mean(y_pred) - K.mean(y_true))/K.std(y_pred - y_true))
    return loss


@ tf.function
def custom_capm_metric(factors):
    """ Call for CAPM metric

    Args:
        factors (tensor): Tensor of factors
    """
    def capm_metric(y_pred, y_true):
        return K.mean(K.square(y_pred - y_true)) + K.mean(factors)
    return capm_metric


class CustomSharpeMetric(tf.keras.metrics.Metric):
    """Custom Sharpe Metric 

    Args:
        tf (env): Tensorflow keras metrics environment
    """

    def __init__(self, num_classes=None, batch_size=None,
                 name='sharpe_ratio', **kwargs):
        """Initialisation for custom metric

        Args:
            num_classes (int, optional): Number of classes. Defaults to None.
            batch_size (int, optional): Batch size. Defaults to None.
            name (str, optional): name of string. Defaults to 'sharpe_ratio'.
        """
        super().__init__(name=name, **kwargs)
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.hedge_portflio_mean = self.add_weight(
            name=name, initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Update state position

        Args:
            y_true (tf): Tensor of realisations
            y_pred (tf): Tensor of predictions
            sample_weight (int, optional): Sample weights. Defaults to None.
        """
        # Returns the index of the maximum values along the last axis in y_true (Last layer)
        y_true = K.argmax(y_true, axis=-1)
        # Returns the index of the maximum values along the last axis in y_true (Last layer)
        y_pred = K.argmax(y_pred, axis=-1)
        # Flattens a tensor to reshape to a shape equal to the number of elements contained
        # Removes all dimensions except for one.
        y_true = K.flatten(y_true)
        # Defines the metric for assignment
        true_poss = K.sum(K.cast((K.equal(y_true, y_pred)), dtype=tf.float32))
        self.hedge_portflio_mean.assign_add(true_poss)

    def result(self):
        """Result return

        Returns:
            Metric: 
        """
        return self.hedge_portflio_mean

#################################################################################
# Autodiff Testing
#################################################################################
# Information:
# TensorFlow provides the tf.GradientTape API for automatic differentiation;
# that is, computing the gradient of a computation with respect to some inputs,
# usually tf.Variables. TensorFlow "records" relevant operations executed inside
# the context of a tf.GradientTape onto a "tape". TensorFlow then uses that tape
# to compute the gradients of a "recorded" computation using reverse mode differentiation.
# (https://en.wikipedia.org/wiki/Automatic_differentiation)

# Function to test loss functions and metrics using autodiff


def loss_function_testing():
    """ Uses tensorflow autodifferientiation functionality
        to confirm differientable nature and feasibility
        of custom loss functions.
        Note: code verbatim from tensorflow guide.
        Merely for illustration purposes
    """
    layer = tf.keras.layers.Dense(32, activation='relu')
    x = tf.constant([[1., 2., 3.]])
    # Sets loss functions

    # Set Metrics
    with tf.GradientTape() as tape:
        # Forward pass
        y = layer(x)

        loss = tf.reduce_mean(y**2)
    # Calculate gradients with respect to every trainable variable
    try:
        grad = tape.gradient(loss, layer.trainable_variables)
    except:
        print('Gradient Function Failed')
    # Print the outcomes of the simple model analysis
    for var, g in zip(layer.trainable_variables, grad):
        print(f'{var.name}, shape: {g.shape}')
    return

# Function for implementing autodiff


def autodiff_guide(example):
    """ Execute autodiff examples from Tensorflow resources.
        Used to help gain an understanding of different
        functionalities (Demonstration Purposes Only)

    Args:
        example (int): Example to implement
                     : 1 - 'simple'
                     : 2 - 'simple_tensor'
                     : 3 - 'simple_model'
                     : 4 - 'control_tape'
                     : 5 - 'control_tensor_tape'
                     : 6 - 'stop_recording'
                     : 7 - 'watch_multiple_variables'
                     : 8 - 'higher_order_derivatives'
                     : 9 - 'jacobian'
                     : 10- 'hessian_newton'

    """
    # Uses the autodiff functionality to test custom gradients with gradient tape
    # Extracted from
    if example == 1:
        # Simple example
        print('Starting Simple Example')
        x = tf.Variable(3.0)
        with tf.GradientTape() as tape:
            y = x**2
        # dy = 2x * dx
        dy_dx = tape.gradient(y, x)
        print(dy_dx.numpy())
    if example == 2:
        w = tf.Variable(tf.random.normal((3, 2)), name='w')
        b = tf.Variable(tf.zeros(2, dtype=tf.float32), name='b')
        x = [[1., 2., 3.]]
        with tf.GradientTape(persistent=True) as tape:
            y = x @ w + b
        loss = tf.reduce_mean(y**2)
        [dl_dw, dl_db] = tape.gradient(loss, [w, b])
        print(w.shape)
        print(dl_dw.shape)
    if example == 3:
        layer = tf.keras.layers.Dense(2, activation='relu')
        x = tf.constant([[1., 2., 3.]])
        with tf.GradientTape() as tape:
            # Forward pass
            y = layer(x)
            loss = tf.reduce_mean(y**2)
        # Calculate gradients with respect to every trainable variable
        grad = tape.gradient(loss, layer.trainable_variables)
        # Print the outcomes of the simple model analysis
        for var, g in zip(layer.trainable_variables, grad):
            print(f'{var.name}, shape: {g.shape}')
    if example == 4:
        # A trainable variable
        x0 = tf.Variable(3.0, name='x0')
        # Not trainable
        x1 = tf.Variable(3.0, name='x1', trainable=False)
        # Not a Variable: A variable + tensor returns a tensor.
        x2 = tf.Variable(2.0, name='x2') + 1.0
        # Not a variable
        x3 = tf.constant(3.0, name='x3')
        with tf.GradientTape() as tape:
            y = (x0**2) + (x1**2) + (x2**2)
        grad = tape.gradient(y, [x0, x1, x2, x3])
        for g in grad:
            print(g)
        [var.name for var in tape.watched_variables()]
    if example == 5:
        x = tf.constant(3.0)
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = x**2
        # dy = 2x * dx
        dy_dx = tape.gradient(y, x)
        print(dy_dx.numpy())
    if example == 6:
        # Sets the variables
        x = tf.Variable(2.0)
        y = tf.Variable(3.0)
        # Starts the graident tape
        with tf.GradientTape() as t:
            x_sq = x * x
            with t.stop_recording():
                y_sq = y * y
            z = x_sq + y_sq
        # Compute the gradient
        grad = t.gradient(z, {'x': x, 'y': y})
        # Shows tape starting and stopping with the reporting
        print('dz/dx:', grad['x'])  # 2*x => 4
        print('dz/dy:', grad['y'])
    if example == 7:
        # Set the variables
        x0 = tf.constant(0.0)
        x1 = tf.constant(0.0)
        # Establish gradient tape
        with tf.GradientTape() as tape0, tf.GradientTape() as tape1:
            tape0.watch(x0)
            tape1.watch(x1)
            # Establish sin & sigmoid functions
            y0 = tf.math.sin(x0)
            y1 = tf.nn.sigmoid(x1)
            # Create combined function, tracking multiple components
            y = y0 + y1
            ys = tf.reduce_sum(y)
    if example == 8:
        # Higher order derivatives
        x = tf.Variable(1.0)  # Create a Tensorflow variable initialized to 1.0
        with tf.GradientTape() as t2:
            with tf.GradientTape() as t1:
                y = x * x * x
        # Compute the gradient inside the outer `t2` context manager
        # which means the gradient computation is differentiable as well.
            dy_dx = t1.gradient(y, x)
        d2y_dx2 = t2.gradient(dy_dx, x)
        # Prints the result from the gradient outputs
        print('dy_dx:', dy_dx.numpy())  # 3 * x**2 => 3.0
        print('d2y_dx2:', d2y_dx2.numpy())  # 6 * x => 6.0
    if example == 9:
        # Jacobian Matrices
        x = tf.random.normal([7, 5])
        layer = tf.keras.layers.Dense(10, activation=tf.nn.relu)
        # Shape of the gradient tape
        with tf.GradientTape(persistent=True) as tape:
            y = layer(x)
        # Output Layer Shape
        y.shape
        # Shape of the kernal
        layer.kernel.shape
        # The shape of the Jacobian of the output with respect to the kernel
        # is the combination of the two shapes
        j = tape.jacobian(y, layer.kernel)
        j.shape
        # Summing over the targtes dimensions gives you the amount calculated
        # a scaler gradient
        g = tape.gradient(y, layer.kernel)
        print('g.shape:', g.shape)
        j_sum = tf.reduce_sum(j, axis=[0, 1])
        delta = tf.reduce_max(abs(g - j_sum)).numpy()
        assert delta < 1e-3
        print('delta:', delta)
    if example == 10:
        # Construction of Simple Hessian Matrix
        # A Hessian Matrix is a square matrix of 2nd order PDEs of a scaler
        # valued function, or scaler field, describing the local curvature of
        # a multivariate function
        x = tf.random.normal([7, 5])
        layer1 = tf.keras.layers.Dense(8, activation=tf.nn.relu)
        layer2 = tf.keras.layers.Dense(6, activation=tf.nn.relu)
        with tf.GradientTape() as t2:
            with tf.GradientTape() as t1:
                x = layer1(x)
                x = layer2(x)
                loss = tf.reduce_mean(x**2)
            g = t1.gradient(loss, layer1.kernel)
        h = t2.jacobian(g, layer1.kernel)
        print(f'layer.kernel.shape: {layer1.kernel.shape}')
        print(f'h.shape: {h.shape}')
        # Flatten axes into matrix and flatten to gradient vector
        n_params = tf.reduce_prod(layer1.kernel.shape)
        g_vec = tf.reshape(g, [n_params, 1])
        h_mat = tf.reshape(h, [n_params, n_params])
        # Define function to display hessian matrix

        def imshow_zero_center(image, **kwargs):
            lim = tf.reduce_max(abs(image))
            plt.imshow(image, vmin=-lim, vmax=lim, cmap='seismic', **kwargs)
            plt.colorbar()
        # Shows the hessian matrix
        imshow_zero_center(h_mat)
        # Newton's Method Update Step
        eps = 1e-3
        eye_eps = tf.eye(h_mat.shape[0])*eps
        # X(k+1) = X(k) - (?????f(X(k)))^-1 @ ???f(X(k))
        # h_mat = ?????f(X(k))
        # g_vec = ???f(X(k))
        update = tf.linalg.solve(h_mat + eye_eps, g_vec)
        # Reshape the update and apply it to the variable.
        _ = layer1.kernel.assign_sub(tf.reshape(update, layer1.kernel.shape))
    return
#################################################################################
# Analytical/Calculus
#################################################################################
# Writes functions


def analytical_analysis():
    """ Tests symbolic math functionality
    """
    # Test simple functionality
    print(sym.sqrt(8))
    theta, x = sym.symbols('O X')
    return


def ranking_function():
    """ Ranking function to produce charts for demonstration purposes

    Args:
        type ([type]): String for desired ranking functions
    """
    # Creates an ordered, random array of proxy returns (%)
    num = 100
    returns_uniform = np.sort(np.arange(-10, 10, -0.2))
    print('returns', returns_uniform)
    print('returns size', np.size(returns_uniform))
    returns = np.sort(np.random.uniform(low=-10.0, high=10.0, size=(num,)))
    # returns = returns[::-1].sort
    base = np.zeros(num)
    ones = np.ones(num)
    # Creates rank array
    rank = np.linspace(num, 1, num)
    # Sets thresholds
    u = np.zeros((rank.shape))
    u[:] = 20
    v = np.zeros((rank.shape))
    v[:] = 80
    # rank = np.array(list(range(1,len(returns)+ 1)))
    # Create weights
    weights = returns/transpose(ones)
    print('weights', weights)
    print('Sum of weights', np.sum(weights))
    weights = weights*returns
    print('weights', weights)
    print('Sum of weights', np.sum(weights))
    # Plots the functions
    plt.plot(returns, rank, 'r.', base, rank, 'k.',
             returns, u, 'g--', returns, v, 'b--')
    # Invert the y-axis
    plt.gca().invert_yaxis()
    plt.gca().invert_xaxis()
    plt.legend('Returns', 'Baseline')
    plt.xlabel('Excess Return (y(i,t), %)')
    plt.ylabel('Rank (R(y(i,t)))')
    plt.title('Monotonic Ranking Function')
    plt.savefig(
        '/home/connormcdowall/finance-honours/results/plots/monotonic-ranking.png')
    return


#################################################################################
# Variables
#################################################################################
# Integers
batch_size = 256  # Batch size for creating tf dataset
chunk_size = 100000  # chunk size for reading stata files
# Targets
targets_dictionary = {1: 'ret_exc', 2: 'ret_exc_lead1m'}
# Sets the intended target column (test multiple configurations)
target_column = targets_dictionary[2]
# Lists and arrays
# 1: , 2: , 3:
resizing_options = [True, True, True]
categorical_assignment = ['size_grp', 'permno', 'permco',
                          'crsp_shrcd', 'crsp_exchcd', 'adjfct', 'sic', 'ff49']
# Tensorflow configurations (listed for completeness/reference)
# Optimizers
optimizers = ['Adagrad', 'Adadelta', 'Adam',
              'Adamax', 'Ftrl', 'Nadam', 'RMSprop', 'SGD']
# Losses
binary_classification_losses = ['binary_crossentropy']
multiclass_classfication_losses = ['categorical_crossentropy',
                                   'sparse_categorical_crossentropy', 'poisson', 'kl_divergence']
regression_losses = ['cosine_similarity', 'mean_absolute_error', 'mean_absolute_percentage_error',
                     'mean_squared_logarithmic_error', 'mean_squared_error', 'huber_loss']
extra_losses = ['hinge', 'log_cosh', 'loss', 'squared_hinge']
custom_losses = ['custom_mse', 'custom_hp', 'custom_sharpe',
                 'custom_information', 'custom_treynor', 'custom_hp_mse', 'custom_sharpe_mse',
                 'custom_information_mse']  # List names here when created
losses = binary_classification_losses + multiclass_classfication_losses + \
    regression_losses + extra_losses + custom_losses
# Metrics (Functions used to judge model performance,similar to a loss function but results are not used when training a model)
accuracy_metrics = ['accuracy', 'binary_accuracy', 'categorical_accuracy',
                    'top_k_categorical_accuracy', 'sparse_top_k_categorical_accuracy', 'sparse_categorical_accuracy']
probabilistic_metrics = ['binary_crossentropy',
                         'categorical_crossentropy', 'kullback_leibler_divergence']
regression_metrics = ['root_mean_squared_error', 'mean_absolute_percentage_error', 'mean_metric_wrapper', 'sum',
                      'mean_relative_error', 'mean_squared_error', 'mean_squared_logarithmic_error', 'cosine_similarity', 'logcosh', 'mean', 'mean_absolute_error', 'mean_tensor', 'metric']
classification_tf_pn = ['Auc', 'Fn', 'Fp', 'poisson', 'precision', 'precision_at_recall',
                        'recall', 'recall_at_precision', 'sensitivity_at_specificity', 'Tn', 'Tp']
images_segementation_metrics = ['meaniou']
hinge_metrics = ['categorical_hinge', 'squared_hinge', 'hinge']
custom_metrics = ['custom_mse_metric', 'custom_sharpe_metric',
                  'custom_information_metric', 'custom_hp_metric']  # Add when create the metrics
metrics = accuracy_metrics + probabilistic_metrics + regression_metrics + \
    classification_tf_pn + images_segementation_metrics + hinge_metrics + custom_metrics
# Tensorflow congifuration
optimisation_dictionary = {1: 'SGD', 2: 'SGD',
                           3: 'SGD'}
loss_function_dictionary = {
    1: ['mean_squared_error', 'custom_mse', 'custom_sharpe', 'custom_sharpe_mse', 'custom_information', 'custom_hp'], 2: ['custom_hp', 'custom_hp_mse']}
metrics_dictionary = {1: ['mean_squared_error', 'cosine_similarity', 'mean_absolute_error', 'root_mean_squared_error', 'custom_mse_metric', 'custom_sharpe_metric',
                          'custom_information_metric', 'custom_hp_metric'], 2: ['mean_squared_error', 'cosine_similarity', 'mean_absolute_error', 'root_mean_squared_error', 'custom_mse_metric', 'custom_sharpe_metric',
                                                                                'custom_information_metric', 'custom_hp_metric']}
#################################################################################
# Selected Tensorflow Configuration
#################################################################################
tf_option_array = [1, 2]  # 1 = Analysis, 2 = Testing
tf_option = 1  # Change to 1,2,3,4,5,6,7 for configuration
selected_optimizer = optimisation_dictionary[tf_option]
selected_losses = loss_function_dictionary[tf_option]
selected_metrics = metrics_dictionary[tf_option]
# Custom objects dictionary for importing models (Both metrics and losses)
custom_tf_objects = {'custom_mse_metric': custom_mse_metric, 'custom_hp_metric': custom_hp_metric,
                     'custom_sharpe_metric': custom_sharpe_metric, 'custom_information_metric': custom_information_metric, 'custom_mse': custom_mse, 'custom_sharpe': custom_sharpe, 'custom_sharpe_mse': custom_sharpe_mse, 'custom_information': custom_information, 'custom_hp': custom_hp, 'custom_hp_mse': custom_hp_mse}
# Truth dictionary to inform the type of regressions perform
regression_dictionary = {'capm': True, 'ff3': True, 'ff4': True, 'ff5': True}
#################################################################################
# Strings
#################################################################################
# Subsequent directories for creating tensorflow models
model_name = 'cmcd398-finance-honours'
data_source = 'data/combined_predictors_filtered_us.dta'
csv_location = '/Volumes/Seagate/dataframes/'
data_vm_directory = '/home/connormcdowall/local-data/'
data_vm_dta = '/home/connormcdowall/local-data/combined_predictors_filtered_us.dta'
results_tables = '/home/connormcdowall/finance-honours/results/tables'
list_of_columns = '/home/connormcdowall/finance-honours/data/working-columns.txt'
# Subsequent directories for making predictions
train_data = '/home/connormcdowall/finance-honours/data/dataframes/active_train.dta'
test_data = '/home/connormcdowall/finance-honours/data/dataframes/active_test.dta'
val_data = '/home/connormcdowall/finance-honours/data/dataframes/active_validation.dta'
predictions_data = '/home/connormcdowall/finance-honours/data/dataframes/active_test.dta'
model_directory = '/home/connormcdowall/finance-honours/results/models/tensorflow/cmcd398-finance-honours'
# Subsequent directories for making regressions
factor_location = '/home/connormcdowall/finance-honours/data/factors.csv'
predictions_location = '/home/connormcdowall/finance-honours/results/predictions/'
dependant_column = 'predict'
#################################################################################
# Truth Variables (Set to True or False depending on the functions to run)
#################################################################################
# System Checks
sys_check = False
sys_gpu = False
# Data processing
source_data = False
split_vm_data = False
process_vm_data = False
use_sass = False
need_dataframe = False
assign_features = False
extract_test_data = False
test_implementation = False
example_autodiff = False
test_loss_function = False
chronologically_sort_data = False
# Analytical
analytical = False
rank_functions = False
# Model Building
create_models = False
make_predictions = False
perform_regressions = False
# Output
convert_text = False
plot_learning_curves = False
#################################################################################
# Function Testing
#################################################################################
# System Checks
#################################################################################
if sys_check:
    reconfigure_gpu(restrict_tf=False, growth_memory=True)
if sys_gpu:
    set_gpus(manual_GPU_device_placement=False)
#################################################################################
# Data processing
#################################################################################
if source_data:
    partition_data(data_source, csv_location)
if split_vm_data:
    split_vm_dataset(data_vm_directory, create_statistics=False,
                     split_new_data=True, create_validation_set=False)
if process_vm_data:
    process_vm_dataset(data_vm_dta, save_statistics=False, sample=False)
if need_dataframe:
    data = create_dataframes(csv_location, False)
if use_sass:
    sass_access(data)
if chronologically_sort_data:
    sort_data_chronologically(
        data_vm_directory, size_of_chunks=chunk_size, set_top_500=False)
#################################################################################
# Tensorflow
#################################################################################
if assign_features:
    numerical_features, categorical_features = create_feature_lists(
        list_of_columns, categorical_assignment)
if extract_test_data:
    df, train_data, val_data, test_data = download_test_data()
    if test_implementation:
        implement_test_data(df, train_data, val_data,
                            test_data, full_implementation=True)
if example_autodiff:
    autodiff_guide(example=5)
#################################################################################
# Analytical
#################################################################################
if analytical:
    analytical_analysis()
if rank_functions:
    ranking_function()
##################################################################################
# Model Building
##################################################################################
if create_models:
    create_tensorflow_models(data_vm_directory, list_of_columns, categorical_assignment, target_column, chunk_size, resizing_options,
                             batch_size, model_name, selected_optimizer, selected_losses, selected_metrics, split_data=False, trial=False, sample=False)
if make_predictions:
    make_tensorflow_predictions(model_name=model_name, model_directory=model_directory, selected_losses=selected_losses,
                                dataframe_location=predictions_data, custom_objects=custom_tf_objects)
if perform_regressions:
    create_fama_factor_models(model_name=model_name, selected_losses=selected_losses, factor_location=factor_location, prediction_location=predictions_location,
                              dependant_column=dependant_column, regression_dictionary=regression_dictionary, realised_returns=False)
##################################################################################
# Output
##################################################################################
if convert_text:
    execute_conversion_options(model_name, selected_losses,
                               hp_ols=True, pooled_ols=True, true_excess_returns=True)
if plot_learning_curves:
    losses = ['mean_squared_error', 'custom_mse', 'custom_hp']
    for loss in losses:
        create_learning_curves(
            model_name, loss, model_history=None, from_load_file=True)
