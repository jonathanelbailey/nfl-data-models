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


def prepare_wp_data(df):
    # Group by game_id and create receive_2h_ko
    grouped = df.groupby('game_id', group_keys=False)
    pbp = grouped.apply(
        lambda row: row.assign(
            receive_2h_ko=np.where((row['qtr'] <= 2) & (row['posteam'] == row['defteam'].dropna().iloc[0]), 1, 0)
        )
    )

    # Ungroup and create additional columns
    pbp['posteam_spread'] = np.where(pbp['home'] == 1, pbp['spread_line'], -1 * pbp['spread_line'])
    pbp['elapsed_share'] = (3600 - pbp['game_seconds_remaining']) / 3600
    pbp['spread_time'] = pbp['posteam_spread'] * np.exp(-4 * pbp['elapsed_share'])
    pbp['Diff_Time_Ratio'] = pbp['score_differential'] / (np.exp(-4 * pbp['elapsed_share']))

    return pbp


def add_label_column(df):
    df['label'] = np.where(df['posteam'] == df['Winner'], 1, 0)
    return df


def drop_rows(df):
    df.dropna(subset=['ep', 'score_differential', 'play_type', 'label', 'defteam_timeouts_remaining', 'posteam_timeouts_remaining', 'yardline_100'], inplace=True)
    return df.loc[df['qtr'] <= 4]


def add_home_column(df):
    print("Adding Home Column")
    df['home'] = df.apply(lambda row: 1 if row['posteam'] == row['home_team'] else 0, axis=1)
    return df


# def add_label_column(df):
#     print("Adding Label Column")
#     df['label'] = df.apply(lambda row: 1 if (row['result'] > 0 and row['posteam'] == row['home_team']) or (row['result'] < 0 and row['posteam'] == row['away_team']) else 0, axis=1)
#     return df


def add_receive_2h_ko_column(df):
    print("Adding Receive 2H KO Column")
    new_df = df.groupby('game_id', group_keys=False).apply(
        lambda x: x.assign(
            receive_2h_ko=np.where((x['qtr'] <= 2) & (x['posteam'] == x['defteam'].dropna().iloc[0]), 1, 0)
        )
    )
    return new_df


def add_posteam_spread_elasped_share_columns(df):
    print("Adding Posessing Team Spread and Elapsed Share Columns")
    new_df = df.assign(
        posteam_spread=np.where(df['home'] == 1, df['spread_line'], -1 * df['spread_line']),
        elapsed_share=(3600 - df['game_seconds_remaining']) / 3600,
    )
    return new_df


def add_spread_time_diff_time_ration_columns(df):
    print("Adding Spread Time and Diff Time Ratio Columns")
    new_df = df.assign(
        spread_time=df['posteam_spread'] * np.exp(-4 * df['elapsed_share']),
        diff_time_ratio=df['score_differential'] / (np.exp(-4 * df['elapsed_share']))
    )
    return new_df


def select_relevant_columns(df, col):
    print("Selecting Relevant Columns")
    return df.filter(items=col)


def drop_irrelevant_columns(df, col):
    print("Dropping Irrelevant Columns")
    return df.drop(columns=col)


def add_features(df):
    new_df = add_home_column(df)
    new_df = add_label_column(new_df)
    new_df = add_receive_2h_ko_column(new_df)
    new_df = add_posteam_spread_elasped_share_columns(new_df)
    new_df = add_spread_time_diff_time_ration_columns(new_df)

    return new_df