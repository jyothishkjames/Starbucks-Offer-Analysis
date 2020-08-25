import argparse
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
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
    df = pd.read_sql_table(table_name='Data_Table', con=engine)
    X = df.drop('respond', axis=1)
    y = df['respond']
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
        ('clf', RandomForestClassifier())
    ])

    parameters = {
        'scaler__feature_range': [(0, 1)],
        'clf__n_estimators': [10, 50, 100],
        'clf__random_state': [50],
        'clf__class_weight': ['balanced'],
        'clf__max_features': ['auto', 'log2', 'sqrt']
    }

    model = GridSearchCV(pipeline, param_grid=parameters)

    return model


def evaluate_model(model, X_test, Y_test):
    """
    Function to evaluate the model

    INPUT:
    model - model to evaluate
    X_test - test feature
    Y_test - test label
    """
    # predict on test data
    Y_pred = model.predict(X_test)
    # display classification report
    print(classification_report(Y_pred, Y_test))
    # display classification accuracy
    accuracy_test = accuracy_score(Y_test, Y_pred)
    print("The model accuracy on test data is: ", accuracy_test)


def main():
    # Read the command line arguments and store them
    parser = argparse.ArgumentParser()

    parser.add_argument('--file-path-database', action='store', dest='file_path_database', help='filepath of the '
                                                                                                'database',
                        default=False, required=True)

    results = parser.parse_args()

    print('Loading data...')
    X, Y = load_data(results.file_path_database)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    print('Building model...')
    model = build_model()

    print('Training model...')
    model.fit(X_train, Y_train)

    print('Evaluating model...')
    evaluate_model(model, X_test, Y_test)


if __name__ == '__main__':
    main()
