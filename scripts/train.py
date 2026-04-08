import numpy as np
import pandas as pd
import sqlite3
from matplotlib import pyplot as plt


def create_database(path):
    #read data from csv
    #pd = pd.read_csv(path)

    #save data to sqlite
    conn = sqlite3.connect("./data/data.db")
    #pd.to_sql("data", conn, if_exists="replace", index=False)
    return conn



def load_data(query, conn):
    df = pd.read_sql_query(query, conn)
    return df

def main():
    #------step 1: load data------
    conn = create_database("./data/data.csv")
   

    #------step 2: clean data------
    df = load_data("SELECT * FROM data", conn)

    # first we look at the shape of the data
    print("="*50)
    print("Shape of the data:")
    print(df.shape)
    print("="*50)

    # then we look at the data types of the columns
    print("="*50)
    print("Data types of the columns:")
    print(df.dtypes)
    print("="*50)

    # replace empty strings with NaN values
    df = df.replace(r'^\s*$', np.nan, regex=True)

    # now, we look at the unique values, null values and missing values in the data
    print("="*50)
    print("Unique values:")
    print(df.nunique())
    print("="*50)

    print("="*50)
    print("Null values:")
    print(df.isnull().sum())
    print("="*50)

    print("="*50)
    print("Missing values:")
    print(df.isna().sum())
    print("="*50)

    #we can see that there are some unique values that are not useful for our model, we will drop them

    df = df.drop(columns=["customerID"])
    df = df.dropna()

    #------step 3: visualize data------
    #we can visualize the distribution of the target variable
    #but first, we need to convert the target variable with only 2 unique values to a integer[0,1] variable
    # we can use the map function to do this
    # we will map "Yes" to 1 and "No" to 0
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    #df[""] 

if __name__ == "__main__":
    main()