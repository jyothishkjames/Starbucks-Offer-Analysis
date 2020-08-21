import argparse

import pandas as pd
import numpy as np
import math
import json

from sqlalchemy import create_engine

from utils import *


def load_data():
    """
    Function to read the data

    OUTPUT:
    portfolio - portfolio dataframe
    profile - profile dataframe
    transcript - transcript dataframe

    """
    # read in the json files
    portfolio = pd.read_json('../data/dataset/portfolio.json', orient='records', lines=True)
    profile = pd.read_json('../data/dataset/profile.json', orient='records', lines=True)
    transcript = pd.read_json('../data/dataset/transcript.json', orient='records', lines=True)

    return portfolio, profile, transcript


def clean_data(profile, portfolio, transcript, offer, amount):
    """
    Function to clean the data

    INPUT:
    profile - (pandas dataframe) profile as defined at the top of the notebook
    portfolio - (pandas dataframe) portfolio as defined at the top of the notebook
    transcript - (pandas dataframe) transcript as defined at the top of the notebook

    OUTPUT:
    offer_type_df - merged dataframe containing columns offer id, offer type, age,
                    became_member_on, gender, person, income
    amount_df - merged dataframe containing columns event, amount, age,
                became_member_on, gender, person, income

    """
    # Converting None to NAN
    transcript['offer id'] = transcript['value'].apply(offer)
    transcript['amount'] = transcript['value'].apply(amount)

    # Rename column 'id' to 'person'
    profile = profile.rename(columns={'id': 'person'})

    # Rename column 'id' to 'offer id'
    portfolio = portfolio.rename(columns={'id': 'offer id'})

    # Merge dataframes proflie and transcript
    merged_df = profile.merge(transcript, how='right', on='person')

    # Drop Nan values in column 'Gender', 'Income'
    merged_df = merged_df.dropna(subset=['income'])

    # Drop column 'value'
    merged_df.drop(columns=['value'], inplace=True)

    # Create offer dataframe - offer_df
    offer_df = merged_df.dropna(subset=['offer id'])

    # Drop column 'amount' from offer_df dataframe
    offer_df.drop(columns=['amount'], inplace=True)

    # Merge dataframes portfolio and offer_df, map columns 'offer id' to 'offer type'
    offer_type_df = portfolio.merge(offer_df, how='right', on='offer id')

    # Create amount dataframe - amount_df
    amount_df = merged_df.dropna(subset=['amount'])

    # Drop column 'offer id'
    amount_df.drop(columns=['offer id'], inplace=True)

    return offer_type_df, amount_df


def purchase_without_offer(offer_type_df, amount_df):
    """
    Function to find the demographic groups that will make purchases even if they don't receive an offer

    INPUT:
    offer_type_df - (pandas dataframe) offer_type_df returned by function clean_data
    amount_df - (pandas dataframe) amount_df returned by function clean_data

    OUTPUT:
    match_df - (pandas dataframe) dataframe which contains demographic groups that will
                make purchases even if they don't receive an offer

    """

    persons_completed_offer = list(offer_type_df['person'][offer_type_df['event'] == 'offer completed'].unique())

    match_df = amount_df[amount_df['age'] == 144]

    for person in persons_completed_offer:
        match_df = pd.concat([match_df, amount_df[amount_df['person'].isin([person])]])

    return match_df


def generate_features(portfolio, transcript, amount_df):
    """
    Function to generate features for training the model

    INPUT:
    portfolio - (pandas dataframe) portfolio as defined at the top of the notebook
    transcript - (pandas dataframe) transcript as defined at the top of the notebook
    amount_df - (pandas dataframe) amount_df returned by function clean_data

    OUTPUT:
    df_offer_type_amount - (pandas dataframe) dataframe which contains the features for training the model

    """

    # Find duplicated rows based on duplicted time
    duplicated_df = transcript[transcript.duplicated(subset=['time'])]

    # Get rows with event as 'offer completed'
    df_offer_completed = duplicated_df[duplicated_df['event'] == 'offer completed']

    # Drop column amount
    df_offer_completed.drop(columns=['amount', 'event'], inplace=True)

    # Merge dataframes amount_df and df_offer_completed, map columns 'amount' to 'offer id'
    df_offer_amount = amount_df.merge(df_offer_completed, how='right', on=['person', 'time'])

    # Rename column 'id' to 'offer id'
    portfolio = portfolio.rename(columns={'id': 'offer id'})

    # Merge dataframes portfolio and df_offer_amount, map columns 'offer id' to 'offer type'
    df_offer_type_amount = portfolio.merge(df_offer_amount, how='right', on='offer id')

    # Drop unnecessary columns
    df_offer_type_amount.drop(columns=['offer id', 'channels', 'person', 'event', 'time', 'value'], inplace=True)

    # Generate year and month from column became_member_on
    df_became_member_on = pd.to_datetime(df_offer_type_amount['became_member_on'], format='%Y%m%d',
                                         errors='ignore').to_frame()
    df_offer_type_amount['year'] = pd.DatetimeIndex(df_became_member_on['became_member_on']).year
    df_offer_type_amount['month'] = pd.DatetimeIndex(df_became_member_on['became_member_on']).month

    # Drop column became_member_on
    df_offer_type_amount.drop(columns=['became_member_on'], inplace=True)

    return df_offer_type_amount


def create_dummy_df(num_df, cat_df, dummy_na):
    """
    Function to create dummy variables for categorical data

    INPUT:
    num_df - pandas dataframe with numerical variables
    cat_df - pandas dataframe with categorical variables
    dummy_na - Bool holding whether you want to dummy NA vals of categorical columns or not


    OUTPUT:
    num_df - a new dataframe that has the following characteristics:
            1. dummy columns for each of the categorical columns in cat_df
            2. if dummy_na is True - it also contains dummy columns for the NaN values
            3. Use a prefix of the column name with an underscore (_) for separating
    """

    cat_df = pd.get_dummies(cat_df, dummy_na=dummy_na)

    num_df = pd.concat([num_df, cat_df], axis=1)

    return num_df


def save_data(df, database_filepath):
    """
    Function to save the dataframe to a database

    INPUT:
    df - dataframe to save
    database_filepath - path where the database has to saved

    OUTPUT:
    X - features
    y - labels
    """

    engine = create_engine('sqlite:///' + database_filepath + 'starbucks_database.db')
    df.to_sql('Data_Table', engine, index=False)


def main():
    # Read the command line arguments and store them
    parser = argparse.ArgumentParser()
    parser.add_argument('--file-path-database', action='store', dest='file_path_database', help='filepath of the '
                                                                                                'database',
                        default=False, required=True)

    results = parser.parse_args()

    print('Loading data...')
    portfolio, profile, transcript = load_data()

    print('Cleaning data...')
    offer_type_df, amount_df = clean_data(profile, portfolio, transcript, offer, amount)

    print('Creating features...')
    df_offer_type_amount = generate_features(portfolio, transcript, amount_df)

    print('Creating dummy data for categorical features...')

    # numeric cols- difficulty, duration, reward, age, income, amount, year, month
    df_offer_type_amount_numeric = df_offer_type_amount[['reward', 'age', 'income', 'amount', 'year']]

    # categoric cols- , offer_type, channel_1, channel_2, channel_3, channel_4, gender
    df_offer_type_amount_categoric = df_offer_type_amount[['offer_type', 'gender']]

    df = create_dummy_df(df_offer_type_amount_numeric, df_offer_type_amount_categoric, dummy_na=False)

    print('Saving data...\n    DATABASE: {}'.format(results.file_path_database))
    save_data(df, results.file_path_database)

    print('Cleaned data saved to database!')


if __name__ == '__main__':
    main()
