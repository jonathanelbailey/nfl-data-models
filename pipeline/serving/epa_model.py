from typing import Any, Optional, Callable
import pandas as pd
import numpy as np
import xgboost as xgb


# Notice Preprocess class Must be named "Preprocess"
class Preprocess(object):
    def __init__(self):
        # set internal state, this will be called only once. (i.e. not per request)
        pass

    def preprocess(self, body: dict, state: dict, collect_custom_statistics_fn=None) -> Any:
        EPA_MODEL_SELECTED_COLUMNS = ['half_seconds_remaining', 'yardline_100', 'home',
                                      'retractable', 'dome', 'outdoors', 'ydstogo', 'era0', 'era1',
                                      'era2', 'era3', 'era4', 'down1', 'down2', 'down3', 'down4',
                                      'posteam_timeouts_remaining', 'defteam_timeouts_remaining']

        df = pd.DataFrame.from_dict(body)

        downs = pd.get_dummies(df['down'])
        df = df.join(downs)


        df['era0'] = np.where(df['season'] <= 2001, 1, 0)
        df['era1'] = np.where((df['season'] > 2001) & (df['season'] <= 2005), 1, 0)
        df['era2'] = np.where((df['season'] > 2005) & (df['season'] <= 2013), 1, 0)
        df['era3'] = np.where((df['season'] > 2013) & (df['season'] <= 2017), 1, 0)
        df['era4'] = np.where(df['season'] > 2017, 1, 0)

        df['era'] = np.select([
            df['era0'] == 1,
            df['era1'] == 1,
            df['era2'] == 1,
            (df['era3'] == 1) | (df['era4'] == 1)
        ], [
            0,
            1,
            2,
            3
        ])

        df['era'] = pd.Categorical(df['era'])

        df['down1'] = np.where(df['down'] == 1, 1, 0)
        df['down2'] = np.where(df['down'] == 2, 1, 0)
        df['down3'] = np.where(df['down'] == 3, 1, 0)
        df['down4'] = np.where(df['down'] == 4, 1, 0)

        df['home'] = np.where(df['posteam'] == df['home_team'], 1, 0)

        df['model_roof'] = np.where((df['roof'].isna()) | (df['roof'] == 'open') | (df['roof'] == 'closed'),
                                     'retractable', df['roof'])
        df['model_roof'] = pd.Categorical(df['model_roof'])

        df['retractable'] = np.where(df['model_roof'] == 'retractable', 1, 0)
        df['dome'] = np.where(df['model_roof'] == 'dome', 1, 0)
        df['outdoors'] = np.where(df['model_roof'] == 'outdoors', 1, 0)

        df = df.filter(items=EPA_MODEL_SELECTED_COLUMNS)

        return xgb.DMatrix(df)

    def postprocess(self, data: Any, state: dict, collect_custom_statistics_fn=None) -> dict:
        # post process the data returned from the model inference engine
        # data is the return value from model.predict we will put is inside a return value as Y
        preds = pd.DataFrame(data)
        preds.columns = ["Touchdown", "Opp_Touchdown", "Field_Goal", "Opp_Field_Goal", "Safety", "Opp_Safety",
                         "No_Score"]
        return dict(y=preds.to_dict(orient='records'))
