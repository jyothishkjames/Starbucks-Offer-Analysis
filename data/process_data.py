import pandas as pd
import numpy as np
import math
import json


def read_data():
    # read in the json files
    portfolio = pd.read_json('../data/dataset/portfolio.json', orient='records', lines=True)
    profile = pd.read_json('../data/dataset/profile.json', orient='records', lines=True)
    transcript = pd.read_json('../data/dataset/transcript.json', orient='records', lines=True)


def clean_data():
    pass
