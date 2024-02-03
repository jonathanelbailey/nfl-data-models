import pandas as pd
import nfl_data_py as nfl


def load_pbp_data(seasons):
    return nfl.import_pbp_data(seasons, thread_requests=True)


def clean_nfl_data(df):
    return nfl.clean_nfl_data(df)


def save_data(data, file_path):
    data.to_csv(file_path, index=False)