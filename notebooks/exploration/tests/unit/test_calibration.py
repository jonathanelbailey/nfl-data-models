import pandas as pd
import numpy as np
import pytest
from notebooks.exploration.utils import calibration as cal


@pytest.fixture
def data():
    return pd.DataFrame({
        'game_id': [1, 2, 3, 4],
        'season': [2000, 2003, 2007, 2015],
        'week': [1, 2, 3, 4],
        'drive': [1, 2, 3, 4],
        'ep': [1, 2, 3, 4],
        'play_type': ['type1', 'type2', 'type3', 'type4'],
        'roof': ['open', 'closed', 'dome', 'retractable'],
        'ydstogo': [10, 20, 30, 40],
        'home_score': [10, 20, 30, 40],
        'away_score': [5, 25, 35, 45],
        'home_team': ['team1', 'team2', 'team3', 'team4'],
        'away_team': ['team5', 'team6', 'team7', 'team8'],
        'down': [1, 2, 3, 4],
        'half_seconds_remaining': [100, 200, 300, 400],
        'game_seconds_remaining': [100, 200, 300, 400],
        'yardline_100': [10, 20, 30, 40],
        'score_differential': [5, -5, -5, -5],
        'result': [5, -5, -5, -5],
        'posteam': ['team1', 'team2', 'team3', 'team4'],
        'defteam': ['team5', 'team6', 'team7', 'team8'],
        'posteam_timeouts_remaining': [1, 2, 3, 4],
        'defteam_timeouts_remaining': [1, 2, 3, 4],
        'desc': ['desc1', 'desc2', 'desc3', 'desc4'],
        'spread_line': [1, 2, 3, 4],
        'total_line': [1, 2, 3, 4],
        'qtr': [1, 2, 3, 4]
    })


def test_create_wp_calibration_data(data):
    result = cal.create_wp_calibration_data(data)
    assert result['Winner'].tolist() == ['team1', 'team6', 'team7', 'team8']


def test_make_model_mutations(data):
    data['season'] = [2000, 2003, 2007, 2015]
    result = cal.make_model_mutations(data)
    assert result['era'].tolist() == [0, 1, 2, 3]
    assert result['model_roof'].tolist() == ['retractable', 'retractable', 'dome', 'retractable']


def test_prepare_wp_data(data):
    data['home'] = [1, 0, 1, 0]
    result = cal.prepare_wp_data(data)
    assert result['receive_2h_ko'].tolist() == [0, 0, 0, 0]


def test_add_label_column(data):
    data['Winner'] = ['team1', 'team6', 'team7', 'team8']
    result = cal.add_label_column(data)
    assert result['label'].tolist() == [1, 0, 0, 0]


def test_drop_rows(data):
    data['ep'] = [np.nan, 1, 2, 3]
    data['play_type'] = ['type1', 'type2', np.nan, 'type4']
    data['label'] = [1, 0, 1, 0]
    data['defteam_timeouts_remaining'] = [1, 2, 3, np.nan]
    data['posteam_timeouts_remaining'] = [1, 2, np.nan, 4]
    data['yardline_100'] = [10, 20, np.nan, 40]
    result = cal.drop_rows(data)
    assert len(result) == 1