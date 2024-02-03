import nfl_data_py as nfl
import pandas as pd
from multiprocessing import Pool
import os
from clearml import Dataset


def load_pbp_data(seasons):
    pbp_data = nfl.import_pbp_data(seasons, thread_requests=True)
    return pbp_data


def clean_nfl_data(df):
    cleaned_data = nfl.clean_nfl_data(df)
    return cleaned_data


def save_to_csv(df_tuple):
    season, file_prefix, df = df_tuple
    filename = f'{file_prefix}_{season}.csv'
    file_folder_path = os.path.join(file_prefix, filename)
    df.to_csv(file_folder_path)
    print(f'Saved {file_folder_path}')


def process_dataframe(df, file_prefix):
    grouped = df.groupby('season')
    df_tuples = [(season, file_prefix, group) for season, group in grouped]
    with Pool() as pool:
        pool.map(save_to_csv, df_tuples)


def read_from_csv(file_path):
    print(f'Reading {file_path}')
    return pd.read_csv(file_path, low_memory=False, index_col=0)


def read_csvs_in_parallel(file_list):
    with Pool() as pool:
        dataframes = pool.map(read_from_csv, file_list)
    result = pd.concat(dataframes, ignore_index=True)


def get_dataset(name):
    dataset = Dataset.get(
        dataset_name=name
    )
    dataset_files = dataset.list_files()
    dataset_files_path = dataset.get_local_copy()
    dataset_files_list = [os.path.join(dataset_files_path, csv) for csv in dataset_files]
    print(f"Gathered the following Data:\n{dataset_files_list}")

    return dataset_files_list


def update_dataset(dataset_name, dataset_project, file_prefix):
    dataset = Dataset.get(
        dataset_name=dataset_name,
        dataset_project=dataset_project,
        writable_copy=True,
        auto_create=True)
    wildcard = f"{file_prefix}_*.csv"
    dataset.add_files(path=file_prefix, wildcard=wildcard)
    dataset.upload()
    dataset.finalize()
