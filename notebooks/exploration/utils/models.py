import pandas as pd
import xgboost as xgb


class WPModel:
    def __init__(self):
        self.wp_spread_model = None
        self.wp_spread_model_path = 'models/wp_spread_model.json'
        self.n_rounds = 15000
        self.wp_spread_monotone_constraints = {
            'receive_2h_ko': 0,
            'spread_time': 1,
            'home': 0,
            'half_seconds_remaining': 0,
            'game_seconds_remaining': 0,
            'Diff_Time_Ratio': 1,
            'score_differential': 1,
            'down': -1,
            'ydstogo': -1,
            'yardline_100': -1,
            'posteam_timeouts_remaining': 1,
            'defteam_timeouts_remaining': -1
        }
        self.wp_model_parameters = {
            'n_estimators': self.n_rounds,
            'booster': 'gbtree',
            'device': 'cuda',
            'objective': 'binary:logistic',
            'tree_method': 'approx',
            'grow_policy': 'lossguide',
            'sampling_method': 'gradient_based',
            'eval_metric': ['logloss', 'auc', 'error'],
            'early_stopping_rounds': 200,
            'learning_rate': 0.05,
            'gamma': 0.79012017,
            'subsample': 0.9224245,
            'colsample_bytree': 5 / 12,
            'max_depth': 5,
            'min_child_weight': 7,
            'monotone_constraints': self.wp_spread_monotone_constraints
        }
        self.drop_columns = ['season', 'game_id', 'label']

    def train(self, X, y):
        clf = xgb.XGBClassifier(**self.wp_model_parameters)
        clf.fit(X, y,  eval_set=[(X, y)], verbose=50)
        return clf