import argparse
import pandas as pd

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

    # Filter by offer viewed
    offer_type_viewed_df = offer_type_df[offer_type_df['event'] == 'offer viewed']

    # Filter by offer completed
    offer_type_completed_df = offer_type_df[offer_type_df['event'] == 'offer completed']

    # Select obly columns 'person' and 'time' from the dataframe offer_type_completed_df
    offer_type_completed_df = offer_type_completed_df[['person', 'time']]

    # Merge dataframes offer_type_viewed_df and offer_type_completed_df
    merged_offer = offer_type_viewed_df.merge(offer_type_completed_df, how='inner', on=['person', 'time'])

    # List of persons
    person_list = list(merged_offer['person'])

    # List of time
    time_list = list(merged_offer['time'])

    # Empty dataframe of type amount_df
    match_df = amount_df[amount_df['age'] == 144]

    for person, time in zip(person_list, time_list):
        match_df = pd.concat([match_df, amount_df[amount_df['person'].isin([person]) & amount_df['time'].isin([time])]])

    return match_df


def generate_features_classification(merged_offer, offer_type_df):
    """
    Function to generate features for training the model

    INPUT:
    merged_offer - (pandas dataframe) merged_offer returned by function offer_viewed_completed
    offer_type_df - (pandas dataframe) offer_type_df returned by function clean_data

    OUTPUT:
    df_concatinated_sorted - (pandas dataframe) df_concatinated_sorted which contains the features
                             for training the model for classification

    """

    # Generate year and month from column became_member_on
    df_merged_became_member_on = pd.to_datetime(merged_offer['became_member_on'], format='%Y%m%d',
                                                errors='ignore').to_frame()
    df_offer_type_became_member_on = pd.to_datetime(offer_type_df['became_member_on'], format='%Y%m%d',
                                                    errors='ignore').to_frame()

    merged_offer['year'] = pd.DatetimeIndex(df_merged_became_member_on['became_member_on']).year
    merged_offer['month'] = pd.DatetimeIndex(df_merged_became_member_on['became_member_on']).month

    offer_type_df['year'] = pd.DatetimeIndex(df_offer_type_became_member_on['became_member_on']).year
    offer_type_df['month'] = pd.DatetimeIndex(df_offer_type_became_member_on['became_member_on']).month

    # Drop column 'event'
    merged_offer.drop(columns=['event'], inplace=True)

    # Find difference of dataframes
    diff_df2 = offer_type_df[~offer_type_df.apply(tuple, 1).isin(merged_offer.apply(tuple, 1))]

    # Filter by 'offer completed'
    df_event_offer_completed = diff_df2[diff_df2['event'] == 'offer completed']

    # Drop column 'event', 'time'
    df_event_offer_completed.drop(columns=['event', 'time'], inplace=True)

    # Drop column 'time'
    merged_offer.drop(columns=['time'], inplace=True)

    # Find difference of dataframes
    diff_df3 = df_event_offer_completed[~df_event_offer_completed.apply(tuple, 1).isin(merged_offer.apply(tuple, 1))]

    # Filter by 'offer completed'
    diff_df2 = diff_df2[diff_df2['event'] != 'offer completed']

    # Dataframes to be concatenated
    merged_offer['respond'] = 'yes'
    diff_df2['respond'] = 'no'
    diff_df3['respond'] = 'no'

    # Drop column 'event', 'time'
    diff_df2.drop(columns=['event', 'time'], inplace=True)

    # Concatenate dataframes
    df_concatenate = pd.concat([merged_offer, diff_df2, diff_df3], ignore_index=True)

    # Remove duplicate rows
    df_concatenate_sorted = df_concatenate.drop_duplicates(
        subset=['reward', 'age', 'became_member_on', 'income', 'offer_type', 'gender', 'person', 'offer id'])

    # Sort dataframes by 'income'
    df_concatenate_sorted = df_concatenate_sorted.sort_values(by=['income'], ascending=False)

    return df_concatenate_sorted


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
