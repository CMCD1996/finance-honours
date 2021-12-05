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
import itertools as it # Find combinations of lists

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

def split_vm_dataset(data_vm_directory,create_statistics,split_new_data, create_validation_set):
    """ Creates summmary statistics from unprocessed dataset

    Args:
        data_vm_directory (str): Directory location of data stored on the VM instance.
    """
    
    # Create summary statisitics for the entire dataset
    if create_statistics == True:
        # Read data into one dataframe on python
        total_df = pd.read_stata(data_vm_directory + 'combined_predictors_filtered_us.dta')
        data_stats = total_df.describe().round(4)
        data_stats.T.to_latex('results/tables/summary-statistics.txt')
    # Create training and testing dataframes for Tensorflow
    if split_new_data == True:
        train_df = total_df[total_df["train"] == 1]
        test_df = total_df[total_df["test"] == 1]
        # Convert training and testing sets to stata files
        if create_validation_set == True:
            train_new_df,val_df = train_test_split(train_df,test_size=0.2)
            train_new_df.to_stata(data_vm_directory + 'train.dta')
            val_df.to_stata(data_vm_directory + 'val.dta')
        else:
            train_df.to_stata(data_vm_directory + 'train.dta')
        test_df.to_stata(data_vm_directory + 'test.dta')
    return

def process_vm_dataset(data_vm_dta, save_statistics):
    """ This script processes the training and testing datasets for Tensorflow
    following the classify structured data with feature columns tutorial
    """
    # Load the test and train datasets into dataframes
    df = pd.read_stata(data_vm_dta)
    print('Number of instances: ',len(df))
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
    for column in column_list:
        if column != 'size_grp':
            # Sets each column value to float type
            df.astype({column:'float64'}).dtypes
            # Impute missing values with medium values (replace with mean command if necessary)
            df[column].fillna(df[column].median(), inplace = True)
    # Get list of unique values for 
    print(df.info(verbose=True))
    return df

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
    labels = df.pop(target_column)
    df = {key: value[:,tf.newaxis] for key, value in dataframe.items()}
    ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(batch_size)
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

def encode_tensor_flow_features(train_df, val_df, test_df,target_column, numerical_columns, categorical_colummns,categorical_dictionary, size_of_batch=256):
    """ size of batch may vary, defaults to 256
    """
    # Creates the dataset
    train_dataset = create_tf_dataset(train_df,target_column, shuffle=True,batch_size = size_of_batch)
    val_dataset = create_tf_dataset(val_df,target_column,shuffle=False,batch_size = size_of_batch)
    test_dataset = create_tf_dataset(test_df,target_column,shuffle=False,batch_size = size_of_batch)

    # Display a set of batches
    [(train_features, label_batch)] = train_dataset.take(1)
    print('Every feature:', list(train_features.keys()))
    print('A batch of ages:', train_features['Age'])
    print('A batch of targets:', label_batch )

    # Initilise input and encoded featture arrays
    all_inputs = []
    encoded_features = []

    # Normalise the numerical features
    for header in numerical_columns:
        numeric_col = tf.keras.Input(shape=(1,), name=header)
        normalization_layer = get_normalization_layer(header, train_dataset)
        encoded_numeric_col = normalization_layer(numeric_col)
        all_inputs.append(numeric_col)
        encoded_features.append(encoded_numeric_col)

    # Encode the remaicategorical features
    for header in categorical_colummns:
        print('Start: ', header)
        categorical_col = tf.keras.Input(shape=(1,), name=header, dtype=categorical_dictionary[header])
        encoding_layer = get_category_encoding_layer(name=header,
                                                    dataset=train_dataset,
                                                    dtype=categorical_dictionary[header],
                                                    max_tokens=5)
        encoded_categorical_col = encoding_layer(categorical_col)
        all_inputs.append(categorical_col)
        encoded_features.append(encoded_categorical_col)
        print('Passed: ', header)

    # Concatenate all encoded layers
    all_features = tf.keras.layers.concatenate(encoded_features)
    return all_features, all_inputs, train_dataset, val_dataset, test_dataset
    # Create, compile and train the model
def build_tensor_flow_model(train_dataset, val_dataset, test_dataset, model_name, all_features, all_inputs):

    x = tf.keras.layers.Dense(32, activation="relu")(all_features)
    x = tf.keras.layers.Dropout(0.5)(x)
    output = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(all_inputs, output)

    # Configure the model
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
    model.save(model_name)

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
        print("Continues with full implmentation")
        numerical_features = ['PhotoAmt', 'Fee']
        categorical_features = ['Age','Type', 'Color1', 'Color2', 'Gender', 'MaturitySize',
                    'FurLength', 'Vaccinated', 'Sterilized', 'Health', 'Breed1']
        # Create categorical type dictionary
        categorical_dictionary = dict.fromkeys(categorical_features,'string')
        categorical_dictionary["Age"] = 'int64'
        model_name = 'pets_test'
        all_features, all_inputs, train_dataset, val_dataset, test_dataset = encode_tensor_flow_features(train, val, test, target_column, numerical_features, categorical_features,categorical_dictionary, size_of_batch=256)
        model, loss, accuracy = build_tensor_flow_model(train_dataset, val_dataset, test_dataset, model_name, all_features, all_inputs)
    else:
        print('Test functions complete')
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
# Strings
target_variable = 'ret'
# Lists and arrays
categorical_assignment = ['size_grp']
# File paths
data_source = 'data/combined_predictors_filtered_us.dta'
csv_location = '/Volumes/Seagate/dataframes/'
data_vm_directory = '/home/connormcdowall/local-data/'
data_vm_dta = '/home/connormcdowall/local-data/train.dta'
results_tables = '/home/connormcdowall/finance-honours/results/tables'
list_of_columns = 'data/dataframe-columns.txt'
# Binary (Set to True or False depending on the functions to run)
# Data processing
source_data = False
split_vm_data = False
process_vm_data = False
use_sass = False
need_dataframe = False
# Tensorflow
assign_features = False
extract_test_data = True
test_implementation = True
# Analytical
analytical = False
rank_functions = False

#################################################################################
# Function Calls
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
    process_vm_dataset(data_vm_dta,save_statistics=False)

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

