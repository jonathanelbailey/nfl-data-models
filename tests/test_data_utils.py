import unittest
import pandas as pd
from unittest.mock import patch, mock_open
from src.preprocessing.data_utils import load_pbp_data, clean_nfl_data, save_data

class TestDataUtils(unittest.TestCase):

    def setUp(self):
        self.test_data = pd.DataFrame({
            'column1': [1, 2, 3],
            'column2': ['a', 'b', 'c']
        })
        self.seasons = [2019, 2020]

    @patch('src.preprocessing.data_utils.nfl.import_pbp_data')
    def test_load_pbp_data(self, mock_import_pbp_data):
        mock_import_pbp_data.return_value = self.test_data
        result = load_pbp_data(self.seasons)
        mock_import_pbp_data.assert_called_with(self.seasons, thread_requests=True)
        pd.testing.assert_frame_equal(result, self.test_data)

    @patch('src.preprocessing.data_utils.nfl.clean_nfl_data')
    def test_clean_nfl_data(self, mock_clean_nfl_data):
        mock_clean_nfl_data.return_value = self.test_data
        result = clean_nfl_data(self.test_data)
        mock_clean_nfl_data.assert_called_with(self.test_data)
        pd.testing.assert_frame_equal(result, self.test_data)

    @patch("builtins.open", new_callable=mock_open)
    def test_save_data(self, mock_file_open):
        save_data(self.test_data, 'dummy_path.csv')
        mock_file_open.assert_called_with('dummy_path.csv', 'w', encoding='utf-8', errors='strict', newline='')


if __name__ == '__main__':
    unittest.main()
