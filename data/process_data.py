import pandas as pd
import numpy as np
import math
import json


def read_data():
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


def match(offer_type_df, amount_df):
    """
    Function to get the demographic groups that will make purchases even if they don't receive an offer

    INPUT:
    offer_type_df - (pandas dataframe) offer_type_df returned by function clean_data
    amount_df - (pandas dataframe) amount_df returned by function clean_data

    OUTPUT:
    df_match - (pandas dataframe) dataframe which contains demographic groups that will
                make purchases even if they don't receive an offer

    """

    persons_completed_offer = list(offer_type_df['person'][offer_type_df['event'] == 'offer completed'].unique())

    df_match = amount_df[amount_df['age'] == 144]

    for person in persons_completed_offer:
        df_match = pd.concat([df_match, amount_df[amount_df['person'].isin([person])]])

    return df_match


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


