import pandas as pd
import numpy as np
import math
import json


def read_data():
    # read in the json files
    portfolio = pd.read_json('../data/dataset/portfolio.json', orient='records', lines=True)
    profile = pd.read_json('../data/dataset/profile.json', orient='records', lines=True)
    transcript = pd.read_json('../data/dataset/transcript.json', orient='records', lines=True)

    return portfolio, profile, transcript


def clean_data(profile, portfolio, transcript, offer, amount):
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
