# Imports useful python packages
# System
import psutil as ps
import nvidia_smi
# Analytical
from pandas.core.base import NoNewAttributesMixin
import sympy as sym # Symbolic package for calculus
# Essential
import numpy as np
from numpy.core.fromnumeric import transpose # Arithmetic operations
import pandas as pd # Data analysis package
import dask as ds # Data importing for very large software packages.
import matplotlib.pyplot as plt # Simple plotting
import sklearn as skl # Simple statistical models 
from sklearn.model_selection import train_test_split
import tensorflow as tf # Tensorflow (https://www.tensorflow.org/)
from tensorflow.keras import layers
import csv as csv # read and write csvs
import os # change/manipulate operating systems
# Additional
import random as rd # random functionality
import saspy as sas # Use saspy functionality in python
import seaborn as sb # Imports seaborn library for use
# import wrds as wrds# Wharton Research Data Services API
# import pydatastream as pds # Thomas Reuters Datastream API
# import yfinance as yf # Yahoo Finance API
import datetime as dt # Manipulate datetime values
import statsmodels.api as sm # Create Stats functionalities
# import linearmodels as lp # Ability to use PooledOLS
from sklearn.linear_model import LinearRegression
# from stargazer.stargazer import Stargazer #Stargazor package to produce latex tables
# import finance_byu as fin # Python Package for Fama-MacBeth Regressions
from statsmodels.regression.rolling import RollingOLS # Use factor loadings
# from stargazer.stargazer import Stargazer
import sympy as sy # convert latex code
import scipy as sc # Scipy packages
# import tabulate as tb # Create tables in python
import itertools as it

from tensorflow.python.ops.gen_array_ops import split # Find combinations of lists

#################################################################################
# Function Calls
#################################################################################
# Data Processing
#################################################################################
def partition_data(data_location, data_destination):
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
    return
def create_dataframes(csv_location,multi_csv):
    """ Function to create 
    """
    # Creates list of dataframes
    num_csvs = list(range(1,29,1))
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
            np.savetxt(r'/Users/connor/Google Drive/Documents/University/Courses/2020-21/Finance 788/finance-honours/data/dataframe-columns.txt', df.columns, fmt='%s')
        # Save summary statistics to dataframe
        data_stats = df.describe().round(4)
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
    """[summary]

    Args:
        df (dataframe): Pandas Dataframe
        replacement_method (int): Specify replacement methods
                                : 0 - remove rows with nan values
                                : 1 - remove columns with nan values
                                : 2 - fill nan with column mean
                                : 3 - fill nan with column median
    Returns:
        dataframe: Updated pandas dataframe
    """
    nan_total = df.isnull().sum().sum()
    print('Number of nan values before processing: ',nan_total)
    if nan_total > 0:
        # Replace dataframe level nan (rows or columns)
        # Replacement methods (0: remove rows with nan values, medium, remove, none)
        if replacement_method == 0:
            df.dropna(axis = 0, how = 'any',inplace = True)
        # Caution: Change to dataframe-columns.txt and features list required (Do not use)
        if replacement_method == 1:
            df.dropna(axis = 1, how = 'any',inplace = True)
        # Replace column level nan
        for column in df.columns:
            if df[column].isnull().sum() > 0:
                print('Processing nan in: ', column)
                if replacement_method == 2:
                    df[column].fillna(df[column].mean(), inplace = True)
                elif replacement_method == 3:
                    df[column].fillna(df[column].median(), inplace = True)
    nan_total = df.isnull().sum().sum()
    print('Number of nan values after processing: ',nan_total)
    return df

def monitor_memory_usage():
    # Shows CPU information using psutil
    # Shows GPU information using nvidia-ml-py3
    nvidia_smi.nvmlInit()
    deviceCount = nvidia_smi.nvmlDeviceGetCount()
    for i in range(deviceCount):
        # Gets device handle
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
        # Uses handle to get GPU device info
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        # Prints GPU information
        print("Device {}: {}, Memory : ({:.2f}% free): {}(total), {} (free), {} (used)".format(i, nvidia_smi.nvmlDeviceGetName(handle), 100*info.free/info.total, info.total, info.free, info.used))
    nvidia_smi.nvmlShutdown()
    return

def split_vm_dataset(data_vm_directory,create_statistics,split_new_data, create_validation_set):
    """ Creates summmary statistics from unprocessed dataset

    Args:
        data_vm_directory (str): Directory location of data stored on the VM instance.
    """
    # Create Dataframe from the entire dataset
    # total_df = pd.read_stata(data_vm_directory + 'combined_predictors_filtered_us.dta')
    # Create summary statisitics for the entire dataset
    if create_statistics == True:
        # Read data into one dataframe on python
        total_df = pd.read_stata(data_vm_directory + 'combined_predictors_filtered_us.dta')
        data_stats = total_df.describe().round(4)
        data_stats.T.to_latex('results/tables/summary-statistics.txt')
    # Create training and testing dataframes for Tensorflow
    if split_new_data == True:
        train_df = pd.DataFrame()
        test_df = pd.DataFrame()
        total_df = pd.read_stata(data_vm_directory + 'combined_predictors_filtered_us.dta', chunksize =100000)
        for chunk in total_df:
            # train_df = train_df.append(chunk[chunk["train"] == 1])
            test_df = test_df.append(chunk[chunk["test"] == 1])
        # train_df = total_df[total_df["train"] == 1]
        # test_df = total_df[total_df["test"] == 1]
        # Convert training and testing sets to stata files
        if create_validation_set == True:
            train_new_df,val_df = train_test_split(train_df,test_size=0.2)
            print(train_df.info())
            print(val_df.info())
            train_new_df.to_stata(data_vm_directory + 'train.dta')
            val_df.to_stata(data_vm_directory + 'val.dta')
        else:
            train_df.to_stata(data_vm_directory + 'train.dta')
        test_df.to_stata(data_vm_directory + 'test.dta')
    return

def process_vm_dataset(data_vm_dta,categorical_assignment, save_statistics, sample = False):
    """ This script processes the training and testing datasets for Tensorflow
    following the classify structured data with feature columns tutorial
    """
    # Load the test and train datasets into dataframes in chunks
    #df = pd.read_stata(data_vm_dta)
    subset = pd.read_stata(data_vm_dta, chunksize = 100000)
    df_full = pd.DataFrame()
    for df in subset:
        print('Number of instances: ',len(df))
        print('Excess Return')
        print(df['ret_exc'])
        # Find the dtypes of the dataframe and save them to a data column
        if save_statistics==True:
            # Saves dtypes for column dataframe
            np.savetxt(r'/home/connormcdowall/finance-honours/results/statistics/factor-types.txt', df.dtypes, fmt='%s')
            # Saves information on missing values in the dataframe
            np.savetxt(r'/home/connormcdowall/finance-honours/results/statistics/missing-values.txt', df.isna().sum(), fmt='%s')
        # Gets list of dataframe column values
        column_list = list(df.columns.values)
        # Gets list of unique dataframe dtype 
        data_type_list = list(df.dtypes.unique())
        # Gets unique list of size_grp
        size_grp_list = list(df['size_grp'].unique())
        # Removes the mth column/factor from the dataframe given datatime format
        df['mth'] = pd.to_numeric(df['mth'],downcast='float')
        # df.drop(columns=['mth'])
        #for column in column_list:
            #if column not in categorical_assignment:
                # Sets each column value to float type (Change datatype depending on memory)
                # df[column] = df.astype({column:'float64'}).dtypes
                # Impute missing values with medium values (replace with mean command if necessary)
                # df[column].fillna(df[column].median(), inplace = True)
        # Append values to the dataset
        df_full = df_full.append(df)
        if sample:
            df_full = replace_nan(df_full, replacement_method = 3)
            print(df_full.info(verbose=True))
            return df_full
    # Checks Nan in dataframe
    df_full = replace_nan(df_full, replacement_method = 3)
    # Pribts row information
    print(df_full.info(verbose=True))
    print(df_full.head())
    return df_full

#################################################################################
# Machine Learning
#################################################################################
# Utility method to use pandas dataframe to create a tf.data dataset
# Adapted from https://www.tensorflow.org/tutorials/structured_data/feature_columns#use_pandas_to_create_a_dataframe
# Adapted from https://www.tensorflow.org/tutorials/structured_data/preprocessing_layers
def download_test_data():
    dataset_url = 'http://storage.googleapis.com/download.tensorflow.org/data/petfinder-mini.zip'
    csv_file = 'datasets/petfinder-mini/petfinder-mini.csv'
    tf.keras.utils.get_file('petfinder_mini.zip', dataset_url,
                        extract=True, cache_dir='.')
    dataframe = pd.read_csv(csv_file)

    # Creates the target variable for the assignment
    dataframe['target'] = np.where(dataframe['AdoptionSpeed']==4, 0, 1)
    # Drop unused features.
    dataframe = dataframe.drop(columns=['AdoptionSpeed', 'Description'])
    # Split the dataset into training, validation and testing sets
    train, val, test = np.split(dataframe.sample(frac=1), [int(0.8*len(dataframe)), int(0.9*len(dataframe))])
    # Returns the dataframe and the three subsets
    return dataframe, train, val, test

def create_feature_lists(list_of_columns, categorical_assignment):
    # Assignn variables
    categorical_features = []
    numerical_features = []
    file = open(list_of_columns,'r')
    lines = file.readlines()
    for line in lines:
        line = line.rstrip('\n')
        if line in categorical_assignment:
            categorical_features.append(line)
        else:
            numerical_features.append(line)
    # Returns numerical and categorical features
    return numerical_features, categorical_features

def create_tf_dataset(dataframe, target_column, shuffle=True, batch_size=32):
    """[summary]

    Args:
        df (dataframe): dataframe
        target_column (str): Column used to predict for labels
        shuffle (bool, optional): [description]. Defaults to True.
        batch_size (int, optional): Sets batch size. Defaults to 32.

    Returns:
        [type]: [description]
    """
    df = dataframe.copy()
    print(df[target_column].head())
    labels = df.pop(target_column)
    df = {key: value[:,tf.newaxis] for key, value in dataframe.items()}
    ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(batch_size)
    print('Create Dataset: Successful')
    return ds

def get_normalization_layer(name, dataset):
  # Create a Normalization layer for the feature.
  normalizer = layers.Normalization(axis=None)
  # Prepare a Dataset that only yields the feature.
  feature_ds = dataset.map(lambda x, y: x[name])
  # Learn the statistics of the data.
  normalizer.adapt(feature_ds)
  return normalizer

def get_category_encoding_layer(name, dataset, dtype, max_tokens=None):
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

def encode_tensor_flow_features(train_df, val_df, test_df,target_column, numerical_features, categorical_features,categorical_dictionary, size_of_batch=256):
    """ size of batch may vary, defaults to 256
    """
    print('Train missing values')
    print(train_df.isna().sum())
    print('Validation missing values')
    print(val_df.isna().sum())
    print('Testing missing values')
    print(test_df.isna().sum())
    # Creates the dataset
    train_dataset = create_tf_dataset(train_df,target_column, shuffle=True,batch_size = size_of_batch)
    val_dataset = create_tf_dataset(val_df,target_column,shuffle=False,batch_size = size_of_batch)
    test_dataset = create_tf_dataset(test_df,target_column,shuffle=False,batch_size = size_of_batch)

    # Display a set of batches
    [(train_features, label_batch)] = train_dataset.take(1)
    print('Every feature:', list(train_features.keys()))
    print('A batch of ages:', train_features['size_grp'])
    print('A batch of targets:', label_batch)

    # Initilise input and encoded featture arrays
    all_inputs = []
    encoded_features = []
    numerical_count = 0
    categorical_count = 0
    
    # Encode the remaicategorical features
    for header in categorical_features:
        try:
            print('Start: ', header)
            categorical_col = tf.keras.Input(shape=(1,), name=header, dtype=categorical_dictionary[header])
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
            print('Number of Categorical Features Encoded: ',categorical_count)
        except RuntimeError as e:
            print(e) 

    # Normalise the numerical features
    for header in numerical_features:
        try:
            print('Start: ',header)
            numeric_col = tf.keras.Input(shape=(1,), name=header)
            print('Processing: Input Numeric Column')
            normalization_layer = get_normalization_layer(header, train_dataset)
            print('Processing: Sourced Normalization Layer')
            encoded_numeric_col = normalization_layer(numeric_col)
            print('Processing: Encoded Numerical Column')
            all_inputs.append(numeric_col)
            encoded_features.append(encoded_numeric_col)
            print('Passed: ',header)
            numerical_count = numerical_count + 1
            print('Number of Numerical Features Encoded: ',numerical_count)
        except RuntimeError as e:
            print(e) 

    # Concatenate all encoded layers
    all_features = tf.keras.layers.concatenate(encoded_features)
    print('Encoding: Successful')
    return all_features, all_inputs, train_dataset, val_dataset, test_dataset
    # Create, compile and train the model
def build_tensor_flow_model(train_dataset, val_dataset, test_dataset, model_name, all_features, all_inputs,selected_optimizer, selected_loss,selected_metrics, finance_configuration = True):
    # Information pertaining to the tf.keras.layers.dense function
    if finance_configuration:
        # https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense
        # Configure the neural network layers
        x = tf.keras.layers.Dense(
        units = 32, activation="relu", use_bias=True,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros', kernel_regularizer=None,
        bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
        bias_constraint=None)(all_features)
        # List of activation functions
        # 'relu' = Rectified linear unit activation
        # 'sigmond' = Sigmoid activation function, sigmoid(x) = 1 / (1 + exp(-x)).
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
        x = tf.keras.layers.Dropout(rate=0.5, noise_shape = None, seed = None)(x)
        output = tf.keras.layers.Dense(1)(x)
        # Configure the model (https://www.tensorflow.org/api_docs/python/tf/keras/Model)
        model = tf.keras.Model(all_inputs, output)
        # Initilises optimizer variables
        lr = 0.001
        eps =1e-07
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
        ams= False 
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
            opt = tf.keras.optimizers.SGD(learning_rate=lr, momentum=mom, nesterov=nes, name='SGD')
        #################################################################################
        # Losses
        #################################################################################
        # Loss variables
        red = tf.keras.losses.Reduction
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
            lf = tf.keras.losses.Huber(delta=dta, reduction=red, name='huber_loss')
        if selected_loss == 'kl_divergence': # loss = y_true * log(y_true / y_pred)
            lf = tf.keras.losses.KLDivergence(reduction=red, name='kl_divergence')
        if selected_loss == 'log_cosh': #logcosh = log((exp(x) + exp(-x))/2), where x is the error y_pred - y_true.
            lf = tf.keras.losses.LogCosh(reduction=red, name='log_cosh')
        if selected_loss == 'loss':
            lf = tf.keras.losses.Loss(reduction=red, name=None)
        if selected_loss == 'mean_absolute_error': # loss = abs(y_true - y_pred)
            lf = tf.keras.losses.MeanAbsoluteError(reduction=red, name='mean_absolute_error')
        if selected_loss == 'mean_absolute_percentage_error': # loss = 100 * abs(y_true - y_pred) / y_true
            lf = tf.keras.losses.MeanAbsolutePercentageError(reduction=red, name='mean_absolute_percentage_error')
        if selected_loss == 'mean_squared_error': # loss = square(y_true - y_pred)
            lf = tf.keras.losses.MeanSquaredError(reduction=red, name='mean_squared_error')
        if selected_loss == 'mean_squared_logarithmic_error': # loss = square(log(y_true + 1.) - log(y_pred + 1.))
            lf = tf.keras.losses.MeanSquaredLogarithmicError(reduction=red, name='mean_squared_logarithmic_error')
        if selected_loss == 'poisson': # loss = y_pred - y_true * log(y_pred)
            lf = tf.keras.losses.Poisson(reduction=red, name='poisson')
        if selected_loss == 'sparse_categorical_crossentropy':
            lf = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=flt, reduction=red, name='sparse_categorical_crossentropy')
        if selected_loss == 'squared_hinge': # loss = square(maximum(1 - y_true * y_pred, 0))
            lf = tf.keras.losses.SquaredHinge(reduction=red, name='squared_hinge')
        #################################################################################
        # Metrics
        #################################################################################
        # Metric variables
        metrics_list = []
        meaniou_num_classes = 2
        def mean_metric_wrapper_function(y_true, y_pred):
            return tf.cast(tf.math.equal(y_true, y_pred), tf.float32)
        mean_relative_error_normalizer = [1,2,3,4] # Must be the same size as predictions 
        recall = 0.5 # A scalar value in range [0, 1]
        precision = 0.5 # A scalar value in range [0, 1]
        specificity = 0.5 # A scalar value in range [0, 1]
        sensitivity = 0.5 # A scalar value in range [0, 1]
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
        if 'binary_accuracy'in selected_metrics:
            metrics_list.append(tf.keras.metrics.BinaryAccuracy(
            name='binary_accuracy', dtype=None, threshold=0.5))
        if 'binary_crossentropy'in selected_metrics:
            metrics_list.append(tf.keras.metrics.BinaryCrossentropy(
            name='binary_crossentropy', dtype=None, from_logits=False,
            label_smoothing=0))
        if  'categorical_accuracy' in selected_metrics:
            metrics_list.append(tf.keras.metrics.CategoricalAccuracy(
            name='categorical_accuracy', dtype=None))
        if 'categorical_crossentropy'in selected_metrics:
            metrics_list.append(tf.keras.metrics.CategoricalCrossentropy(
            name='categorical_crossentropy', dtype=None, from_logits=False,
            label_smoothing=0))
        if 'categorical_hinge'in selected_metrics:
            metrics_list.append(tf.keras.metrics.CategoricalHinge(
            name='categorical_hinge', dtype=None))
        if 'cosine_similarity'in selected_metrics:
            metrics_list.append(tf.keras.metrics.CosineSimilarity(
            name='cosine_similarity', dtype=None, axis=-1))
        if 'Fn'in selected_metrics:
            metrics_list.append(tf.keras.metrics.FalseNegatives(
                thresholds=None, name=None, dtype=None))
        if 'Fp'in selected_metrics:
            metrics_list.append(tf.keras.metrics.FalsePositives(
            thresholds=None, name=None, dtype=None))
        if 'hinge'in selected_metrics:
            metrics_list.append(tf.keras.metrics.Hinge(
            name='hinge', dtype=None))
        if 'kullback_leibler_divergence'in selected_metrics:
            metrics_list.append(tf.keras.metrics.KLDivergence(
            name='kullback_leibler_divergence', dtype=None))
        if 'logcosh'in selected_metrics:
            metrics_list.append(tf.keras.metrics.LogCoshError(
            name='logcosh', dtype=None))
        if 'mean'in selected_metrics:
            metrics_list.append(tf.keras.metrics.Mean(
            name='mean', dtype=None))
        if 'mean_absolute_error'in selected_metrics:
            metrics_list.append(tf.keras.metrics.MeanAbsoluteError(
            name='mean_absolute_error', dtype=None))
        if 'mean_absolute_percentage_error'in selected_metrics:
            metrics_list.append(tf.keras.metrics.MeanAbsolutePercentageError(
            name='mean_absolute_percentage_error', dtype=None))
        if 'meaniou'in selected_metrics:
            metrics_list.append(tf.keras.metrics.MeanIoU(
            num_classes=meaniou_num_classes, name=None, dtype=None))
        if 'mean_metric_wrapper'in selected_metrics:
           metrics_list.append(tf.keras.metrics.MeanMetricWrapper(
            fn=mean_metric_wrapper_function, name=None, dtype=None))
        if 'mean_relative_error'in selected_metrics:
            metrics_list.append(tf.keras.metrics.MeanRelativeError(
            normalizer = mean_relative_error_normalizer, name=None, dtype=None))
        if 'mean_squared_error'in selected_metrics:
            metrics_list.append(tf.keras.metrics.MeanSquaredError(
            name='mean_squared_error', dtype=None))
        if 'mean_squared_logarithmic_error'in selected_metrics:
            metrics_list.append(tf.keras.metrics.MeanSquaredLogarithmicError(
            name='mean_squared_logarithmic_error', dtype=None))
        if 'mean_tensor'in selected_metrics:
            metrics_list.append(tf.keras.metrics.MeanTensor(
            name='mean_tensor', dtype=None, shape=None))
        if 'metric'in selected_metrics:
            metrics_list.append(tf.keras.metrics.Metric(
            name=None, dtype=None))
        if 'poisson'in selected_metrics:
            metrics_list.append(tf.keras.metrics.Poisson(
            name='poisson', dtype=None))
        if 'precision'in selected_metrics:
            metrics_list.append(tf.keras.metrics.Precision(
            thresholds=None, top_k=None, class_id=None, name=None, dtype=None))
        if 'precision_at_recall'in selected_metrics:
            metrics_list.append(tf.keras.metrics.PrecisionAtRecall(
            recall, num_thresholds=200, class_id=None, name=None, dtype=None))
        if 'recall'in selected_metrics:
            metrics_list.append(tf.keras.metrics.Recall(
            thresholds=None, top_k=None, class_id=None, name=None, dtype=None))
        if 'recall_at_precision'in selected_metrics:
            metrics_list.append(tf.keras.metrics.RecallAtPrecision(
            precision, num_thresholds=200, class_id=None, name=None, dtype=None))
        if 'root_mean_squared_error'in selected_metrics:
            metrics_list.append(tf.keras.metrics.RootMeanSquaredError(
            name='root_mean_squared_error', dtype=None))
        if 'sensitivity_at_specificity'in selected_metrics:
            metrics_list.append(tf.keras.metrics.SensitivityAtSpecificity(
            specificity, num_thresholds=200, class_id=None, name=None, dtype=None))
        if 'sparse_categorical_accuracy'in selected_metrics:
            metrics_list.append(tf.keras.metrics.SparseCategoricalAccuracy(
            name='sparse_categorical_accuracy', dtype=None))
        if 'sparse_top_k_categorical_accuracy'in selected_metrics:
            metrics_list.append(tf.keras.metrics.SparseTopKCategoricalAccuracy(
            k=5, name='sparse_top_k_categorical_accuracy', dtype=None))
        if 'specificty_at_sensitivity'in selected_metrics:
            metrics_list.append(tf.keras.metrics.SpecificityAtSensitivity(
            sensitivity, num_thresholds=200, class_id=None, name=None, dtype=None))
        if 'squared_hinge'in selected_metrics:
            metrics_list.append(tf.keras.metrics.SquaredHinge(
            name='squared_hinge', dtype=None))
        if 'sum'in selected_metrics:
            metrics_list.append(tf.keras.metrics.Sum(
            name='sum', dtype=None))
        if 'top_k_categorical_accuracy'in selected_metrics:
            metrics_list.append(tf.keras.metrics.TopKCategoricalAccuracy(
            k=5, name='top_k_categorical_accuracy', dtype=None))
        if 'Tn'in selected_metrics:
            metrics_list.append(tf.keras.metrics.TrueNegatives(
            thresholds=None, name=None, dtype=None))
        if 'Tp'in selected_metrics:
            metrics_list.append(tf.keras.metrics.TruePositives(
            thresholds=None, name=None, dtype=None))
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
        # Compiler
        #################################################################################
        # Compiler variables
        # Establishes the compiler
        model.compile(
            optimizer=opt, loss=lf, metrics=metrics_list, loss_weights=lw,
            weighted_metrics=wm, run_eagerly=regly, steps_per_execution=spe)
        #################################################################################
        # Visualise model (https://www.tensorflow.org/api_docs/python/tf/keras/utils/plot_model)
        #################################################################################
        # Visualisation variables
        to_file = 'results/plots/tensorflow-visualisations/'+ model_name +'.png'
        show_shapes = True
        show_dtype = False
        show_layer_names = True
        rankdir = 'LR' # TB (Top Bottom), LR (Left Right)
        expand_nested = False
        dpi = 96
        layer_range = None
        show_layer_activations = False
        tf.keras.utils.plot_model(model, to_file, show_shapes, show_dtype,
        show_layer_names, rankdir, expand_nested, dpi,layer_range, show_layer_activations)
        #################################################################################
        # Model.fir (https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit)
        #################################################################################
        # Fit variables
        x = train_dataset
        y = None
        batch_size = None
        epochs=10
        verbose = 'auto'
        callbacks=None
        validation_split=0.0
        validation_data=val_dataset
        shuffle=True,
        class_weight=None
        sample_weight=None
        initial_epoch=0
        steps_per_epoch=None
        validation_steps=None
        validation_batch_size=None
        validation_freq=1,
        max_queue_size=10
        workers=1
        use_multiprocessing=False
        # Fit the model
        model.fit(x, y, batch_size, epochs, verbose,
        callbacks, validation_split, validation_data, shuffle,
        class_weight, sample_weight, initial_epoch, steps_per_epoch,
        validation_steps, validation_batch_size, validation_freq,
        max_queue_size, workers, use_multiprocessing)
        #################################################################################
        # Model.evaluate (https://www.tensorflow.org/api_docs/python/tf/keras/Model#evaluate)
        #################################################################################
        # Evaluation variables
        x = test_dataset
        y = None#Only use if target variables not specified in dataset, must align with x.
        batch_size = None
        sample_weight = None
        steps = None
        verbose = 1 # 0 or 1. Verbosity mode. 0 = silent, 1 = progress bar.
        sample_weight = None
        steps = None
        callbacks = None
        max_queue_size = 10
        workers = 1
        use_multiprocessing = False
        return_dict = False
        # Model evaluation
        model.evaluate(x, y, batch_size, verbose, sample_weight, steps,
        callbacks, max_queue_size, workers, use_multiprocessing,
        return_dict)
        #################################################################################
        loss, metrics = model.evaluate(test_dataset)
        print("Loss: ", loss)
        print("Metric Descriptions: ", model.metrics_names)
        print("Metric Values: ", metrics)
        # Save the model
        model.save('results/plots/tensorflow-models/'+model_name+'.pb')
        # Return the model, loss and accuracy
        return model,loss, metrics
    else:
        # Set up neural net layers
        x = tf.keras.layers.Dense(32, activation="relu")(all_features)
        x = tf.keras.layers.Dropout(rate=0.5, noise_shape = None, seed = None)(x)
        output = tf.keras.layers.Dense(1)(x)
        # Configure the model
        model = tf.keras.Model(all_inputs, output)
        model.compile(optimizer='adam',
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
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
        return model,loss, accuracy

def perform_tensorflow_model_inference(model_name, sample):
    """ Perform evaluations from model (must be configured)

    Args:
        model_name ([type]): [description]
        sample ([type]): [description]

    Returns:
        [type]: [description]
    """
    reloaded_model = tf.keras.models.load_model(model_name)
    input_dict = {name: tf.convert_to_tensor([value]) for name, value in sample.items()}
    predictions = reloaded_model.predict(input_dict)
    prob = tf.nn.sigmoid(predictions[0])
    return prob

def implement_test_data(dataframe, train, val, test,full_implementation = False):
    # Sets the batch size
    target_column = 'target'
    batch_size = 5
    train_ds = create_tf_dataset(train, target_column, shuffle=True, batch_size=batch_size)
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
        categorical_features = ['Age','Type', 'Color1', 'Color2', 'Gender', 'MaturitySize',
                    'FurLength', 'Vaccinated', 'Sterilized', 'Health', 'Breed1']
        # Create categorical type dictionary
        categorical_dictionary = dict.fromkeys(categorical_features,'string')
        categorical_dictionary["Age"] = 'int64'
        model_name = 'pets_test'
        selected_optimizer = 'adam'
        selected_loss = 'binary_crossentropy'
        selected_metrics = ['accuracy']
        all_features, all_inputs, train_dataset, val_dataset, test_dataset = encode_tensor_flow_features(train, val, test, target_column, numerical_features, categorical_features,categorical_dictionary, size_of_batch=256)
        model, loss, metrics = build_tensor_flow_model(train_dataset, val_dataset, test_dataset, model_name, all_features, all_inputs, selected_optimizer,selected_loss, selected_metrics, finance_configuration = False)
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
        prob = perform_tensorflow_model_inference('results/plots/tensorflow-models/'+model_name+'.pb', sample)
    else:
        print('Test functions complete')
    return

def autodiff_guide(example):
    # Uses the autodiff functionality to test custom gradients with gradient tape
    # Extracted from
    if example == 'simple':
        # Simple example
        print('Starting Simple Example')
        x = tf.Variable(3.0)
        with tf.GradientTape() as tape:
            y = x**2
        # dy = 2x * dx
        dy_dx = tape.gradient(y,x)
        print(dy_dx.numpy())
    if example == 'simple_tensor':
        w = tf.Variable(tf.random.normal((3, 2)), name='w')
        b = tf.Variable(tf.zeros(2, dtype=tf.float32), name='b')
        x = [[1., 2., 3.]]
        with tf.GradientTape(persistent=True) as tape:
            y =x @ w + b
        loss = tf.reduce_mean(y**2)
        [dl_dw, dl_db] = tape.gradient(loss, [w, b])
        print(w.shape)
        print(dl_dw.shape)
    if example == 'simple_model':
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
    if example == 'control_tape':
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
    if example == 'control_tensor_tape':
        x = tf.constant(3.0)
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = x**2
        # dy = 2x * dx
        dy_dx = tape.gradient(y, x)
        print(dy_dx.numpy())
    
    return
class PortfolioReturnsEquallyWeighted(tf.keras.losses.Loss):
    def __init__(self):
        # Initialise the function
        super().__init__()
        # Define the call of the function
    def call(self,y_true,y_pred):
        return 1
def autodiff_implementation():
    """ Implments all the project functionality

    """
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
    # Develop this function to test autodiff functionality

    return
def project_analysis(data_vm_directory,list_of_columns,categorical_assignment,target_column,model_name, selected_optimizer, selected_loss, selected_metrics, split_data = False, trial = False, sample = True):
    # Split the initial vm dataset
    if split_data:
        split_vm_dataset(data_vm_directory,create_statistics=False,split_new_data=True, create_validation_set=True)
    # Creates the training, validation and testing dataframes
    test_df = process_vm_dataset(data_vm_directory + 'test.dta',categorical_assignment,save_statistics=False, sample = True)
    train_df = process_vm_dataset(data_vm_directory + 'train.dta',categorical_assignment,save_statistics=False, sample = True)
    val_df = process_vm_dataset(data_vm_directory + 'val.dta',categorical_assignment,save_statistics=False, sample = True)
    # Use trial to test the dataframe when functions not as large
    if trial:
        test_df,test_discard_df = train_test_split(test_df,test_size=0.95)
        train_df, train_discard_df = train_test_split(train_df,test_size=0.95)
        val_df, val_discard_df = train_test_split(val_df,test_size=0.95)
    print(test_df.info())
    print(train_df.info())
    print(val_df.info())
    print('Excess Return')
    print(train_df['ret_exc'])
    # Creates inputs for the create feature lists function
    # Create feature lists for deep learning
    numerical_features, categorical_features = create_feature_lists(list_of_columns, categorical_assignment)
    # Creates the categorical dictonary (must specify the variables types of each)
    categorical_dictionary = dict.fromkeys(categorical_features,'string')
    category_dtypes = {'size_grp':'string','permno':'int32','permco': 'int32','crsp_shrcd':'int8','crsp_exchcd':'int8','adjfct':'float64','sic':'float64','ff49':'float64'}
    for key in category_dtypes:
        categorical_dictionary[key] = category_dtypes[key]
    # categorical_dictionary["size_grp"] = 'float64'
    # Encodes the tensorflow matrix
    all_features, all_inputs, train_dataset, val_dataset, test_dataset = encode_tensor_flow_features(train_df,val_df,test_df,target_column,numerical_features,categorical_features,categorical_dictionary,size_of_batch=1)
    # Buids tensorflow model
    # model,loss, metrics = build_tensor_flow_model(train_dataset, val_dataset, test_dataset, model_name, all_features, all_inputs,selected_optimizer, selected_loss,selected_metrics, finance_configuration = True)
    return
#################################################################################
# Analytical/Calculus
#################################################################################
# Writes functions
def analytical_analysis():
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
#################################################################################
# Variables
#################################################################################
# Targets
targets_dictionary = {1:'ret_exc',2:'ret_exc_lead1m'}
target_column= targets_dictionary[1] # Sets the intended target column (test multiple configurations)
# Lists and arrays
categorical_assignment = ['size_grp','permno','permco','crsp_shrcd','crsp_exchcd','adjfct','sic','ff49']
# Tensorflow configurations
optimizers = ['Adagrad','Adadelta','Adam','Adamax','Ftrl','Nadam','RMSprop','SGD']
losses = ['binary_crossentropy','categorical_crossentropy','cosine_similarity',
        'hinge','huber_loss','kl_divergence','log_cosh','loss','mean_absolute_error','mean_absolute_percentage_error',
        'mean_squared_error','mean_squared_logarithmic_error','poisson','sparse_categorical_crossentropy',
        'squared_hinge']
metrics = ['Auc','accuracy','binary_accuracy','binary_crossentropy', 'categorical_accuracy',
        'categorical_crossentropy','categorical_hinge','cosine_similarity','Fn','Fp','hinge',
        'kullback_leibler_divergence','logcosh','mean','mean_absolute_error',
        'mean_absolute_percentage_error','meaniou', 'mean_metric_wrapper',
        'mean_relative_error','mean_squared_error', 'mean_squared_logarithmic_error',
         'mean_tensor','metric','poisson','precision','precision_at_recall',
         'recall','recall_at_precision','root_mean_squared_error','sensitivity_at_specificity',
        'sparse_categorical_accuracy','sparse_top_k_categorical_accuracy','squared_hinge',
        'sum','top_k_categorical_accuracy','Tn','Tp']
# Tensorflow selections
model_name = 'finance-honours-test'
selected_optimizer = 'Adam'
selected_loss = 'mean_squared_error'
selected_metrics = ['mean_relative_error','mean_squared_error','mean_absolute_error']
# File paths
data_source = 'data/combined_predictors_filtered_us.dta'
csv_location = '/Volumes/Seagate/dataframes/'
data_vm_directory = '/home/connormcdowall/local-data/'
data_vm_dta = '/home/connormcdowall/local-data/combined_predictors_filtered_us.dta'
results_tables = '/home/connormcdowall/finance-honours/results/tables'
list_of_columns = '/home/connormcdowall/finance-honours/data/dataframe-columns.txt'
# Binary (Set to True or False depending on the functions to run)
# System Checks
sys_check = False
# Data processing
source_data = False
split_vm_data = False
process_vm_data = False
use_sass = False
need_dataframe = False
# Tensorflow
assign_features = False
extract_test_data = False
test_implementation = False
enable_autodiff = False
# Analytical
analytical = False
rank_functions = False
# Research Proposal Analysis
begin_analysis = True
#################################################################################
# Function Calls - Testing
#################################################################################
# System Checks
#################################################################################
if sys_check:
    restrict_tf = True
    growth_memory = True
    # Check the number of GPUs avaiable to Tensorflow and in use
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    # Limit tf to a specfic set of GO devices
    gpus = tf.config.list_physical_devices('GPU')
    # Restrict TensorFlow to only use the first GPU
    if gpus and restrict_tf:
        try:
            tf.config.set_visible_devices(gpus[0], 'GPU')
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
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
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
            print(e)
#################################################################################
# Data processing
#################################################################################
# Source data from local drive
if source_data:
    partition_data(data_source,csv_location)
# Source data from VM Instance
if split_vm_data:
    split_vm_dataset(data_vm_directory,create_statistics=False, split_new_data= False,create_validation_set= False)
# Process vm data for Tensorflow
if process_vm_data:
    process_vm_dataset(data_vm_dta,save_statistics=False, sample= False)
if need_dataframe:
    data = create_dataframes(csv_location,False)
    print(data.info())
    print(data.head())
if use_sass:
    sass_access(data)
#################################################################################
# Tensorflow
#################################################################################
if assign_features:
    numerical_features, categorical_features = create_feature_lists(list_of_columns, categorical_assignment)
if extract_test_data:
    df, train_data, val_data, test_data = download_test_data()
    if test_implementation:
        implement_test_data(df, train_data, val_data, test_data,full_implementation = True)
if enable_autodiff:
    autodiff_guide(example='simple_model')
#################################################################################
# Analytical
#################################################################################
# Analytical function
# Do analytical function
if analytical:
    analytical_analysis()
# Creates monotonic ranking function plots
if rank_functions:
    ranking_function()
##################################################################################
# Function Call - Analysis
##################################################################################
if begin_analysis:
    project_analysis(data_vm_directory,list_of_columns,categorical_assignment,target_column, model_name, selected_optimizer, selected_loss, selected_metrics, split_data = False, trial = True, sample = True)
    

