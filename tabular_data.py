import pandas as pd
import numpy as np
import plotly.express as px
import missingno as msno
import matplotlib.pyplot as plt
def remove_rows_with_missing_ratings(table):
    # table = pd.read_csv(address)
    # print(table.columns)

    #Check duplicated data
    # print("duplicated data count: ",table.duplicated().sum())
    # print(table.describe())
    # print(table.info())

    #Check zeros data
    # questionable_columns = table.columns
    # zero_counts = {col: 0 for col in questionable_columns}
    # for col in questionable_columns:
    #     zero_counts[col] = table[col][table[col] == 'Nan'].count()
    # print(zero_counts)

    # fig = px.histogram(table, "bathrooms")
    # fig.show()
    # print(table.describe())
    # for col in questionable_columns:
    #     table[col][table[col] == 0] = np.nan
    # print(table.isnull().sum()/ table.count() * 100)
    # fig = msno.matrix(table)
    # plt.show()
    table.dropna(subset=['Value_rating'],inplace=True)
    # fig = msno.matrix(table)
    # plt.show()
    # table.dropna(subset=['bathrooms'],inplace=True)
    print(table[table['guests']=='Somerford Keynes England United Kingdom'])
    table = table.drop([586])
    
    return table

def clean_string(data):
    data = data[21:]
    return data
def combine_description_strings(table):
    table.dropna(subset=['Description'],inplace=True)
    
    ### Task 1
    # The "Description" column contains lists of strings. You'll need to define a function called combine_description_strings which combines the list items into the same string. Unfortunately, pandas doesn't recognise the values as lists, but as strings whose contents are valid Python lists. You should look up how to do this (don't implement a from-scratch solution to parse the string into a list). The lists contain many empty quotes which should be removed. If you don't remove them before joining the list elements with a whitespace, they might cause the result to contain multiple whitespaces in places. The function should take in the dataset as a pandas dataframe and return the same type. It should remove any records with a missing description, and also remove the "About this space" prefix which every description starts with.
    table.apply(clean_string)

    # fig = msno.matrix(table)
    # plt.show()
    return table

def set_default_feature_values(table):
    # fig = msno.matrix(table)
    # plt.show()
    table = table.fillna(1)
    # fig = msno.matrix(table)
    # plt.show()
    return table
def clean_tabular_data(table):
    table = remove_rows_with_missing_ratings(table)
    table = combine_description_strings(table)
    table = set_default_feature_values(table)
    return table
def load_airbnb(column = 'Price_Night'):
    # print(type(table))
    # print(table.columns())
    table = pd.read_csv('./clean_tabular_data.csv')
    labels = table[column]
    table.drop(columns=[column])
    return table,labels

if __name__ == '__main__':
    table = pd.read_csv('./airbnb-property-listings/tabular_data/listing.csv')
    table = clean_tabular_data(table)
    table.to_csv('./clean_tabular_data.csv')
    table,labels = load_airbnb()
    print(table.columns)
