import argparse
import pandas as pd
import numpy as np
import pickle

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sqlalchemy import create_engine


def load_data(database_filepath):
    """
    Function to load the data from database

    INPUT:
    database_filepath - file path to the database

    OUTPUT:
    X - features
    y - labels
    """
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table(table_name='starbucks_database', con=engine)
    X = df.drop('amount', axis=1)
    y = df['amount']
    return X, y


def build_model():
    """
    Function to build the machine learning pipeline

    OUTPUT:
    model - model that is build
    """
    # build machine learning pipeline
    pipeline = Pipeline([
        ('scaler', MinMaxScaler()),
        ('clf', RandomForestRegressor())
    ])

    parameters = {
        'scaler__feature_range': [(0, 1)],
        'clf__n_estimators': [10, 50, 100],
        'clf__random_state': [50],
        'clf__max_depth': [10, 20],
        'clf__criterion': ['mse', 'mae']
    }

    model = GridSearchCV(pipeline, param_grid=parameters)

    return model
