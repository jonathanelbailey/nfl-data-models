import numpy as np
from src.preprocessing.data_utils import load_pbp_data, clean_nfl_data, save_data


class WinProbabilityCalibrationData:
    def __init__(self, start, end):
        self.seasons = list(range(start, end + 1))
        self.output_path = 'calibration_data/wp_model_calibration_data.csv'
        self.selected_columns = [
            'label', 'game_id', 'home_team', 'away_team', 'season', 'half_seconds_remaining',
            'game_seconds_remaining', 'score_differential', 'down', 'ydstogo', 'yardline_100',
            'posteam_timeouts_remaining', 'defteam_timeouts_remaining', 'home', 'receive_2h_ko',
            'spread_time', 'diff_time_ratio'
        ]
        self.drop_columns = ['season', 'game_id', 'label', 'home_team', 'away_team']

    @staticmethod
    def add_home_column(data):
        data['home'] = data.apply(lambda row: 1 if row['posteam'] == row['home_team'] else 0, axis=1)
        return data

    @staticmethod
    def add_label_column(data):
        data['label'] = data.apply(lambda row: 1 if (row['result'] > 0 and row['posteam'] == row['home_team']) or (row['result'] < 0 and row['posteam'] == row['away_team']) else 0, axis=1)
        return data

    @staticmethod
    def add_receive_2h_ko_column(data):
        data = data.groupby('game_id', group_keys=False).apply(
            lambda x: x.assign(
                receive_2h_ko=np.where((x['qtr'] <= 2) & (x['posteam'] == x['defteam'].dropna().iloc[0]), 1, 0)
            )
        )
        return data

    def preprocess_data(self, data):
        data = self.add_features(data)
        data = self.select_relevant_columns(data)
        return data

    def add_features(self, data):
        data = self.add_home_column(data)
        data = self.add_label_column(data)
        data = self.add_receive_2h_ko_column(data)
        data = data.assign(
            posteam_spread=np.where(data['home'] == 1, data['spread_line'], -1 * data['spread_line']),
            elapsed_share=(3600 - data['game_seconds_remaining']) / 3600,
        )
        data = data.assign(
            spread_time=data['posteam_spread'] * np.exp(-4 * data['elapsed_share']),
            diff_time_ratio=data['score_differential'] / (np.exp(-4 * data['elapsed_share']))
        )
        return data

    def select_relevant_columns(self, data):
        return data.filter(items=self.selected_columns)

    def drop_irrelevant_columns(self, data):
        return data.drop(columns=self.drop_columns)

    def generate_calibration_data(self):
        data = load_pbp_data(self.seasons)
        data = clean_nfl_data(data)  # Utilizing data_utils.py for data cleaning
        data = self.preprocess_data(data)
        data = self.drop_irrelevant_columns(data)
        save_data(data, self.output_path)  # Utilizing data_utils.py for saving data
