import numpy as np
import pandas as pd
from numpy import genfromtxt

import re
import random

def get_conditions(data, pattern):
    conditions = np.unique(data[1:, 6:26])
    conditions = list(filter(None, conditions))
    return list(filter(pattern.match, conditions))

def parse():
    # train_data1 = pd.read_csv('2005_data.csv', delimiter=',', dtype=str)
    # row = np.unique(train_data1.iloc[0:,7:21])
    data_2006 = genfromtxt('2006_data.csv', max_rows=50000, delimiter=',', skip_header=1, dtype=str)
    data_2005 = genfromtxt('2005_data.csv', max_rows=50000, delimiter=',', dtype=str)
    print('reading data...')
    # train_data1 = genfromtxt('2005_data.csv', max_rows=50000, delimiter=',', dtype=str)
    # data_2007 = genfromtxt('2007_data.csv', delimiter=',', dtype=str)
    train_data1 = np.concatenate((data_2005, data_2006), axis=0)

    print('building dataframes...')
    processed_data = np.concatenate([train_data1[:,0:5], train_data1[:,26:]], axis=1)
    processed_dataframe = pd.DataFrame(data=processed_data[1:], columns=processed_data[0,:])
    original_dataframe = pd.DataFrame(data=train_data1[1:], columns=train_data1[0,:])

    # pattern = re.compile("^(\d\.\d{2}E\+\d?)")
    print('getting columns')
    filter_pattern = re.compile("^(?!\d\.\d{2}E\+\d?).*")
    conditions = get_conditions(train_data1, filter_pattern)

    print('adding columns')
    processed_dataframe = pd.concat(
        [
            processed_dataframe,
            pd.DataFrame(
                [[0]*len(conditions)],
                index=processed_dataframe.index,
                columns=conditions
            )
        ], axis=1
    )

    print('filling up columns')
    for index, row in original_dataframe.iterrows():
        for i in range(20):
            column = "entity_condition_" + str(i+1)
            c = row[column]
            if c:
                if not (str(c) == 'nan') and filter_pattern.match(c):
                    processed_dataframe.at[index, c] = 1

        print(".", end="")
    processed_dataframe.replace('', np.nan, inplace=True)
    processed_dataframe.dropna(axis=0, how='any', inplace=True)
    processed_dataframe.to_csv('data_processed.csv', index=False);

parse()
