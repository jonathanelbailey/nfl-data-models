import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve, CalibrationDisplay
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import xgboost as xgb


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


def plot_feature_importance(clf):
    return xgb.plot_importance(clf)


def plot_calibration_curve(clf, X_test, y_test):
    prob_y = clf.predict_proba(X_test.to_numpy(), validate_features=True)[:, 1]
    prob_true, prob_pred = calibration_curve(y_test, prob_y, n_bins=50)

    return CalibrationDisplay(prob_pred, prob_true, prob_pred).plot()


def plot_confusion_matrix(clf, X_test, y_test):
    preds = clf.predict(X_test.to_numpy(), validate_features=True)
    cm = confusion_matrix(y_test, preds, labels=clf.classes_)

    return ConfusionMatrixDisplay(cm, display_labels=clf.classes_).plot()