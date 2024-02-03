from clearml import PipelineController, TaskTypes


def get_data(name: str):
    from pipeline import get_dataset, read_from_csv
    import pandas as pd
    from multiprocessing import Pool
    print("Getting EPA calibration dataset...")
    epa_cal_data_csv_paths = get_dataset(name)
    with Pool() as pool:
        dataframes = pool.map(read_from_csv, epa_cal_data_csv_paths)

    epa_cal_data = pd.concat(dataframes, ignore_index=True)

    print(f"Total DataFrames Processed: {len(epa_cal_data)}")
    print(f"Calibration Data:\n{epa_cal_data.head()}")

    return epa_cal_data


# def preprocessing(**kwargs):
#     from sklearn.model_selection import GroupKFold
#     import pandas as pd
#
#     epa_cal_data = kwargs['epa_cal_data']
#
#     print("Preprocessing Data...")
#     X = epa_cal_data.loc[:, ~epa_cal_data.columns.isin(['season', 'game_id', 'label', 'home_team', 'away_team'])]
#     y = epa_cal_data['label']
#     groups = epa_cal_data['game_id']
#
#     print("Creating Folds...")
#     group_fold = GroupKFold(n_splits=5)
#     for train_index, test_index in group_fold.split(X, y, groups):
#         X_train, X_test = X.iloc[train_index], X.iloc[test_index]
#         y_train, y_test = y.iloc[train_index], y.iloc[test_index]
#
#     return X_train, y_train, X_test, y_test


# def generate_classifier(**kwargs):
#     import xgboost as xgb
#     kwargs['eval_metric'] = kwargs['eval_metric'].split(',')
#     kwargs['n_estimators'] = int(kwargs['n_estimators'])
#     kwargs['early_stopping_rounds'] = int(kwargs['early_stopping_rounds'])
#     kwargs['max_depth'] = int(kwargs['max_depth'])
#     kwargs['min_child_weight'] = int(kwargs['min_child_weight'])
#     kwargs['learning_rate'] = float(kwargs['learning_rate'])
#     kwargs['gamma'] = float(kwargs['gamma'])
#     kwargs['subsample'] = float(kwargs['subsample'])
#     kwargs['colsample_bytree'] = float(kwargs['colsample_bytree'])
#     print("Creating Classifier...")
#     model = xgb.XGBClassifier(**kwargs)
#
#     return model


def train(**kwargs):
    import xgboost as xgb
    import matplotlib.pyplot as plt

    params = {
        'booster': kwargs['booster'],
        'device': kwargs['device'],
        'sampling_method': kwargs['sampling_method'],
        'objective': kwargs['objective'],
        'eval_metric': kwargs['eval_metric'].split(','),
        'num_class': int(kwargs['num_class']),
        'tree_method': kwargs['tree_method'],
        'grow_policy': kwargs['grow_policy'],
        'eta': float(kwargs['learning_rate']),
        'gamma': float(kwargs['gamma']),
        'subsample': float(kwargs['subsample']),
        'colsample_bytree': float(kwargs['colsample_bytree']),
        'max_depth': int(kwargs['max_depth']),
        'min_child_weight': int(kwargs['min_child_weight'])
    }

    model_data = kwargs["epa_cal_data"]
    model_data = model_data.dropna(subset=['defteam_timeouts_remaining', 'posteam_timeouts_remaining', 'yardline_100'])
    model_data = model_data[
        ['label', 'half_seconds_remaining', 'yardline_100', 'home', 'retractable', 'dome', 'outdoors',
         'ydstogo',
         'era0', 'era1', 'era2', 'era3', 'era4', 'down1', 'down2', 'down3', 'down4', 'posteam_timeouts_remaining',
         'defteam_timeouts_remaining', 'Total_W_Scaled']]

    full_train = xgb.DMatrix(data=model_data.drop(columns=['label', 'Total_W_Scaled']), label=model_data['label'],
                             weight=model_data['Total_W_Scaled'])
    model = xgb.train(params=params,
                      dtrain=full_train,
                      num_boost_round=int(kwargs['n_estimators']),
                      evals=[(full_train, 'train')],
                      verbose_eval=50)
    model.save_model('epa_model.json')
    xgb.plot_importance(model)
    plt.show()




if __name__ == '__main__':

    # create the pipeline controller
    pipe = PipelineController(name="EPA Model Training Pipeline", project="NFL WP Model")

    # set the default execution queue to be used (per step we can override the execution)
    pipe.set_default_execution_queue('default')

    # add pipeline components
    pipe.add_parameter(
        name='dataset',
        description='Dataset Name',
        default='EPA Calibration Data',
        param_type='str'
    )

    pipe.add_parameter(
        name='n_estimators',
        description='Number of estimators',
        default=525,
        param_type='int'
    )

    pipe.add_parameter(
        name='booster',
        description='Booster type',
        default='gbtree',
        param_type='str'
    )

    pipe.add_parameter(
        name='device',
        description='Device type',
        default='cuda',
        param_type='str'
    )

    pipe.add_parameter(
        name='sampling_method',
        description='Sampling method',
        default='gradient_based',
        param_type='str'
    )

    pipe.add_parameter(
        name='objective',
        description='Objective function',
        default='multi:softprob',
        param_type='str'
    )

    pipe.add_parameter(
        name='eval_metric',
        description='Evaluation metrics',
        default='auc,merror,mlogloss',
        param_type='list(str)',
    )

    pipe.add_parameter(
        name='num_class',
        description='Number of Classes',
        default=7,
        param_type='int',
    )

    pipe.add_parameter(
        name='early_stopping_rounds',
        description='Early stopping rounds',
        default=200,
        param_type='int'
    )

    pipe.add_parameter(
        name='tree_method',
        description='Tree method',
        default='approx',
        param_type='str'
    )

    pipe.add_parameter(
        name='grow_policy',
        description='Grow policy',
        default='lossguide',
        param_type='str'
    )

    pipe.add_parameter(
        name='learning_rate',
        description='Learning rate',
        default=0.025,
        param_type='float'
    )

    pipe.add_parameter(
        name='gamma',
        description='Gamma',
        default=1,
        param_type='float'
    )

    pipe.add_parameter(
        name='subsample',
        description='Subsample ratio',
        default=0.8,
        param_type='float'
    )

    pipe.add_parameter(
        name='colsample_bytree',
        description='Colsample bytree ratio',
        default=0.8,
        param_type='float'
    )

    pipe.add_parameter(
        name='max_depth',
        description='Max depth',
        default=5,
        param_type='int'
    )

    pipe.add_parameter(
        name='min_child_weight',
        description='Minimum child weight',
        default=1,
        param_type='int'
    )

    pipe.add_function_step(
        name='get_data',
        function=get_data,
        function_kwargs=dict(name='${pipeline.dataset}'),
        function_return=['epa_cal_data'],
        cache_executed_step=True,
        task_type=TaskTypes.data_processing
    )
    # pipe.add_function_step(
    #     name='preprocessing',
    #     function=preprocessing,
    #     parents=['get_data'],
    #     function_kwargs=dict(epa_cal_data='${get_data.epa_cal_data}'),
    #     function_return=["X_train", "y_train", "X_test", "y_test"],
    #     cache_executed_step=True,
    #     task_type=TaskTypes.data_processing
    # )
    # pipe.add_function_step(
    #     name='generate_classifier',
    #     function=generate_classifier,
    #     function_kwargs=dict(n_estimators='${pipeline.n_estimators}',
    #                          booster='${pipeline.booster}',
    #                          device='${pipeline.device}',
    #                          sampling_method='${pipeline.sampling_method}',
    #                          objective='${pipeline.objective}',
    #                          eval_metric='${pipeline.eval_metric}',
    #                          early_stopping_rounds='${pipeline.early_stopping_rounds}',
    #                          tree_method='${pipeline.tree_method}',
    #                          grow_policy='${pipeline.grow_policy}',
    #                          learning_rate='${pipeline.learning_rate}',
    #                          gamma='${pipeline.gamma}',
    #                          subsample='${pipeline.subsample}',
    #                          colsample_bytree='${pipeline.colsample_bytree}',
    #                          max_depth='${pipeline.max_depth}',
    #                          min_child_weight='${pipeline.min_child_weight}',
    #                          monotone_constraints='${pipeline.monotone_constraints}'),
    #     function_return=['model'],
    #     monitor_models=['model'],
    #     cache_executed_step=True,
    #     task_type=TaskTypes.training
    # )
    pipe.add_function_step(
        name='train',
        function=train,
        parents=['get_data'],
        function_kwargs=dict(epa_cal_data='${get_data.epa_cal_data}',
                             n_estimators='${pipeline.n_estimators}',
                             booster='${pipeline.booster}',
                             device='${pipeline.device}',
                             sampling_method='${pipeline.sampling_method}',
                             objective='${pipeline.objective}',
                             eval_metric='${pipeline.eval_metric}',
                             num_class='${pipeline.num_class}',
                             early_stopping_rounds='${pipeline.early_stopping_rounds}',
                             tree_method='${pipeline.tree_method}',
                             grow_policy='${pipeline.grow_policy}',
                             learning_rate='${pipeline.learning_rate}',
                             gamma='${pipeline.gamma}',
                             subsample='${pipeline.subsample}',
                             colsample_bytree='${pipeline.colsample_bytree}',
                             max_depth='${pipeline.max_depth}',
                             min_child_weight='${pipeline.min_child_weight}'),
        function_return=['model'],
        monitor_models=['model'],
        cache_executed_step=True,
        task_type=TaskTypes.training
    )

    # For debugging purposes run on the pipeline on current machine
    # Use run_pipeline_steps_locally=True to further execute the pipeline component Tasks as subprocesses.
    pipe.start_locally(run_pipeline_steps_locally=True)

    # Start the pipeline on the services queue (remote machine, default on the clearml-server)
    # pipe.start()

    print('pipeline completed')