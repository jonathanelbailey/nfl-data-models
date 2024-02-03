import unittest
import pandas as pd
from unittest.mock import patch
from src.preprocessing.preprocessing import WinProbabilityCalibrationData


class TestWinProbabilityCalibrationData(unittest.TestCase):

    def setUp(self):
        self.calibrator = WinProbabilityCalibrationData(2019, 2020)
        self.test_data = pd.DataFrame({
            'posteam': ['team1', 'team2'],
            'home_team': ['team1', 'team2'],
            'qtr': [1, 3],
            'game_seconds_remaining': [3600, 1800],
            'score_differential': [0, 7],
            'spread_line': [-3, 3],
            'game_id': ['TEAM1_TEAM2_2019', 'TEAM2_TEAM1_2019'],
            'season': [2019, 2019],
            'label': [1, 0],
            'away_team': ['team2', 'team1'],
            'half_seconds_remaining': [1800, 3600],
            'down': [1, 2],
            'ydstogo': [10, 5],
            'yardline_100': [50, 50],
            'posteam_timeouts_remaining': [3, 3],
            'defteam_timeouts_remaining': [3, 3],
            'spread_time': [-3, 3],
            'diff_time_ratio': [0, 7],
            'receive_2h_ko': [0, 0],
            'elapsed_share': [0.5, 0.5],
            'posteam_spread': [-3, 3],
            'home': [1, 0],
            'defteam': [0, 0]
        })

    def test_add_home_column(self):
        processed_data = self.calibrator.add_home_column(self.test_data.copy())
        self.assertIn('home', processed_data.columns)
        self.assertTrue((processed_data['home'] == (processed_data['posteam'] == processed_data['home_team'])).all())

    def test_add_receive_2h_ko_column(self):
        processed_data = self.calibrator.add_receive_2h_ko_column(self.test_data.copy())
        self.assertIn('receive_2h_ko', processed_data.columns)

    def test_preprocess_data(self):
        processed_data = self.calibrator.preprocess_data(self.test_data.copy())
        for col in self.calibrator.selected_columns:
            self.assertIn(col, processed_data.columns)

    def test_add_features(self):
        processed_data = self.calibrator.add_features(self.test_data.copy())
        self.assertIn('posteam_spread', processed_data.columns)
        self.assertIn('elapsed_share', processed_data.columns)
        self.assertIn('spread_time', processed_data.columns)
        self.assertIn('diff_time_ratio', processed_data.columns)

    def test_select_relevant_columns(self):
        processed_data = self.calibrator.select_relevant_columns(self.test_data.copy())
        self.assertEqual(set(processed_data.columns), set(self.calibrator.selected_columns))

    @patch('src.preprocessing.preprocessing.load_pbp_data')
    @patch('src.preprocessing.preprocessing.save_data')
    def test_generate_calibration_data(self, mock_save_data, mock_load_pbp_data):
        mock_load_pbp_data.return_value = self.test_data.copy()
        self.calibrator.generate_calibration_data()
        mock_load_pbp_data.assert_called_with(self.calibrator.seasons)
        mock_save_data.assert_called()


if __name__ == '__main__':
    unittest.main()