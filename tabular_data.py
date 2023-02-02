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
    return table

# def clean_string(data):
#     data = data[21:]
#     data = data.split(" ''")
#     return "".join(data.split())
def combine_description_strings(table):
    table.dropna(subset=['Description'],inplace=True)
    
    ### Task 1
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
if __name__ == '__main__':
    table = pd.read_csv('./airbnb-property-listings/tabular_data/listing.csv')
    table = clean_tabular_data(table)
    table.to_csv('./clean_tabular_data.csv')
