import numpy as np
import pandas as pd
def create_wp_calibration_data(df):
    data = df.copy()
    # Fix bug in pit-ari game and create Winner column
    data['result'] = data['home_score'] - data['away_score']
    data['Winner'] = np.where(data['home_score'] > data['away_score'], data['home_team'],
                              np.where(data['home_score'] < data['away_score'], data['away_team'], 'TIE'))

    # Filters
    pbp_data = data.loc[
        data[['down', 'game_seconds_remaining', 'yardline_100', 'score_differential', 'result', 'posteam']].notna().all(
            axis=1) &
        (data['qtr'] <= 4)
        ]

    # Select specific columns
    pbp_data = pbp_data[[
        'game_id', 'play_type', 'game_seconds_remaining', 'half_seconds_remaining', 'yardline_100', 'roof',
        'posteam', 'defteam', 'home_team', 'ydstogo', 'season', 'qtr', 'down', 'week', 'drive', 'ep',
        'score_differential', 'posteam_timeouts_remaining', 'defteam_timeouts_remaining', 'desc', 'Winner',
        'spread_line', 'total_line'
    ]]

    return pbp_data


def make_model_mutations(pbp):
    pbp['era0'] = np.where(pbp['season'] <= 2001, 1, 0)
    pbp['era1'] = np.where((pbp['season'] > 2001) & (pbp['season'] <= 2005), 1, 0)
    pbp['era2'] = np.where((pbp['season'] > 2005) & (pbp['season'] <= 2013), 1, 0)
    pbp['era3'] = np.where((pbp['season'] > 2013) & (pbp['season'] <= 2017), 1, 0)
    pbp['era4'] = np.where(pbp['season'] > 2017, 1, 0)

    pbp['era'] = np.select([
        pbp['era0'] == 1,
        pbp['era1'] == 1,
        pbp['era2'] == 1,
        (pbp['era3'] == 1) | (pbp['era4'] == 1)
    ], [
        0,
        1,
        2,
        3
    ])

    pbp['era'] = pd.Categorical(pbp['era'])

    pbp['down1'] = np.where(pbp['down'] == 1, 1, 0)
    pbp['down2'] = np.where(pbp['down'] == 2, 1, 0)
    pbp['down3'] = np.where(pbp['down'] == 3, 1, 0)
    pbp['down4'] = np.where(pbp['down'] == 4, 1, 0)

    pbp['home'] = np.where(pbp['posteam'] == pbp['home_team'], 1, 0)

    pbp['model_roof'] = np.where((pbp['roof'].isna()) | (pbp['roof'] == 'open') | (pbp['roof'] == 'closed'),
                                 'retractable', pbp['roof'])
    pbp['model_roof'] = pd.Categorical(pbp['model_roof'])

    pbp['retractable'] = np.where(pbp['model_roof'] == 'retractable', 1, 0)
    pbp['dome'] = np.where(pbp['model_roof'] == 'dome', 1, 0)
    pbp['outdoors'] = np.where(pbp['model_roof'] == 'outdoors', 1, 0)

    return pbp


def prepare_wp_data(pbp):
    # Group by game_id and create receive_2h_ko
    pbp['receive_2h_ko'] = pbp.groupby('game_id').apply(
        lambda x: np.where((x['qtr'] <= 2) & (x['posteam'] == x['defteam'].first_valid_index()), 1, 0)
    ).reset_index(level=0, drop=True)

    # Ungroup and create additional columns
    pbp['posteam_spread'] = np.where(pbp['home'] == 1, pbp['spread_line'], -1 * pbp['spread_line'])
    pbp['elapsed_share'] = (3600 - pbp['game_seconds_remaining']) / 3600
    pbp['spread_time'] = pbp['posteam_spread'] * np.exp(-4 * pbp['elapsed_share'])
    pbp['Diff_Time_Ratio'] = pbp['score_differential'] / (np.exp(-4 * pbp['elapsed_share']))

    return pbp