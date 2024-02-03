import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import RocCurveDisplay
from sklearn.inspection import PartialDependenceDisplay
from sklearn.calibration import CalibrationDisplay
import matplotlib.pyplot as plt
import pickle

class WPModel:
    def __init__(self, hyperparameters=None):
        self.hyperparameters = hyperparameters
        self.clf = None
        self.drop_columns = ['season', 'game_id', 'label', 'home_team', 'away_team']
        self.groups = None
        self.Xy = None
        self.Xy_train = None
        self.Xy_test = None
        self.y_hat = None
        self.y_pred = None

    def preprocess_data(self, df):
        data = df.copy()
        self.Xy = self.split_xy(data)
        self.groups = data['game_id']
        self.Xy_train, self.Xy_test = self.create_folds()

        return self.Xy, self.groups, self.Xy_train, self.Xy_test

    def split_xy(self, df):
        data = df.copy()
        X = data.loc[:, ~data.columns.isin(self.drop_columns)]
        y = data['label']

        return X, y

    def create_folds(self, n_splits=5):
        group_k_fold = GroupKFold(n_splits=n_splits)
        X, y = self.Xy

        for train_index, test_index in group_k_fold.split(X, y, self.groups):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        Xy_train = (X_train, y_train)
        Xy_test = (X_test, y_test)

        return Xy_train, Xy_test

    # TODO: add cross validation
    # TODO: add feature selection
    # TODO: add non-vegas wp model
    # TODO: retrain previous models
    # TODO: database utilities
    # TODO: nfl simulation
    # TODO: create interception likelihood model
    # TODO: create penalty likelihood model
    # TODO: add gpu numpy/pandas

if __name__ == '__main__':
    import argparse
    from textwrap import dedent
    from clearml import Task


    def main(args):
        params_task = Task.init(
            project_name="NFL WP Model",
            task_name="Prepare parameters for XGBClassifier",
            tags=["NFL", "WP", "Model", "Calibration", "Vegas", "Prepare", "Parameters", "XGBClassifier"]
        )

        hyperparams = args.__dict__.pop('values')
        cal_data_path = args.__dict__.pop('calibration_data')
        hyperparameters = args if hyperparams is None else hyperparams
        monotone_constraints = args.monotone_constraints.split(',')
        args.monotone_constraints = {pair.split('=')[0]: int(pair.split(':')[1]) for pair in monotone_constraints}
        params_task.mark_completed()

        preprocess_task = Task.init(
            project_name="NFL WP Model",
            task_name="Preprocessing",
            tags=["NFL", "WP", "Model", "Calibration", "Vegas", "Preprocessing"]
        )
        wp_model = WPModel(hyperparameters)
        cal_df = pd.read_csv(cal_data_path)
        cal_data = cal_df.copy()
        (X, y), groups, (X_train, y_train), (X_test, y_test) = wp_model.preprocess_data(cal_data)
        preprocess_task.mark_completed()

        train_task = Task.init(
            project_name="NFL WP Model",
            task_name="Train XGBClassifier",
            tags=["NFL", "WP", "Model", "Calibration", "Vegas", "Train", "XGBClassifier"]
        )
        clf = xgb.XGBClassifier(**wp_model.hyperparameters)
        clf.fit(X_train, y_train,  eval_set=[(X_train, y_train), (X_test, y_test)], verbose=50)

        scores = clf.evals_result(X_test)
        y_val = clf.predict(X_test)
        y_pred = clf.predict_proba(X_test, validate_features=True)

        CalibrationDisplay.from_estimator(clf, X_test, y_test)
        RocCurveDisplay.from_estimator(clf, X_test, y_test)
        fig, ax = plt.subplots(figsize=(12, 12))
        fig.tight_layout(pad=3.0)
        PartialDependenceDisplay.from_estimator(clf, X_test, features=X.columns, ax=ax)

        pickle.dump(open('model.pkl', 'wb'))
        train_task.setup_aws_upload(bucket='test', host='minio-test-s3.internal.magiccityit.com:443', key='IJOxeQlvKPgSA84Eq5zr', secret='21fmw0PZuTd35fXqE3YgwSssSqEJMcbBwDgNNWzy', region='us-east-1', verify=True, secure=True)
        train_task.mark_completed()



    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Process XGBClassifier hyperparameters.',
        epilog=dedent('''\
            FURTHER READING
            ---------------
            
            Subject                             Link
            -------                             ----
            XGBClassifier hyperparameters:      https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBClassifier
            GENERAL parameters:                 https://xgboost.readthedocs.io/en/stable/parameter.html#general-parameters.
            TREE_BOOSTER parameters:            https://xgboost.readthedocs.io/en/stable/parameter.html#parameters-for-tree-booster.
            LEARNING_TASK parameters:           https://xgboost.readthedocs.io/en/stable/parameter.html#learning-task-parameters.
        ''')
    )
    general_param = 'GENERAL'
    treeboost_param = 'TREE_BOOSTER'
    learning_task_param = 'LEARNING_TASK'

    # Add arguments for all hyperparameters
    parser.add_argument('-f', '--values', type=str, metavar='PATH', required=False, default='hyperparameters.yaml', help='Path to hyperparameters file. Default: hyperparameters.yaml')
    parser.add_argument('--calibration-data', type=str, metavar='PATH', required=False, default='cal_data.csv', help='Path to calibration data. Default: cal_data.csv')
    parser.add_argument('--n_estimators', type=int, metavar='N', required=False, default=15000, help='Number of estimators. Default: 15000')
    parser.add_argument('--early_stopping_rounds', type=int, metavar='N', required=False, default=200,
                        help='Early stopping rounds. Default: 200')
    parser.add_argument('--booster', type=str, metavar=general_param, required=False, default='gbtree', help='Booster type. Default: gbtree.')
    parser.add_argument('--device', type=str, metavar=general_param, required=False, default='cuda', help='Device type. Default: cuda.')
    parser.add_argument('--sampling_method', type=str, metavar=general_param, required=False, default='gradient_based', help='Sampling method. Default: gradient_based.')
    parser.add_argument('--objective', type=str, metavar=learning_task_param, required=False, default='binary:logistic', help='Objective function. Default: binary:logistic.')
    parser.add_argument('--eval_metric', type=str, metavar=learning_task_param, nargs='+', required=False, default='logloss,auc,error',
                        help='Evaluation metrics. Default: logloss,auc,error.')
    parser.add_argument('--tree_method', type=str, metavar=treeboost_param, required=False, default='approx', help='Tree method. Default: approx.')
    parser.add_argument('--grow_policy', type=str, metavar=treeboost_param, required=False, default='lossguide', help='Grow policy. Default: lossguide.')
    parser.add_argument('--learning_rate', type=float, metavar=treeboost_param, required=False, default=0.05, help='Learning rate. Default: 0.05.')
    parser.add_argument('--gamma', type=float, metavar=treeboost_param, required=False, default=0.79012017, help='Gamma. Default: 0.79012017.')
    parser.add_argument('--subsample', type=float, metavar=treeboost_param, required=False, default=0.9224245, help='Subsample ratio. Default: 0.9224245.')
    parser.add_argument('--colsample_bytree', type=float, metavar=treeboost_param, required=False, default=0.4166666666666667,
                        help='Colsample bytree ratio. Default: 0.4166666666666667.')
    parser.add_argument('--max_depth', type=int, metavar=treeboost_param, required=False, default=5, help='Maximum depth of the trees. Default: 5.')
    parser.add_argument('--min_child_weight', type=int, metavar=treeboost_param, required=False, default=7, help='Minimum child weight. Default: 7.')
    parser.add_argument('--monotone_constraints', type=str, metavar=treeboost_param, nargs='+', required=False, default='receive_2h_ko=0,spread_time=1,home=0,half_seconds_remaining=0,game_seconds_remaining=0,diff_time_ratio=1,score_differential=1,down=-1,ydstogo=-1,yardline_100=-1,posteam_timeouts_remaining=1,defteam_timeouts_remaining=-1',
                        help='Monotone constraints. Default: receive_2h_ko=0,spread_time=1,home=0,half_seconds_remaining=0,game_seconds_remaining=0,diff_time_ratio=1,score_differential=1,down=-1,ydstogo=-1,yardline_100=-1,posteam_timeouts_remaining=1,defteam_timeouts_remaining=-1')

    args = parser.parse_args()
    main(args)
