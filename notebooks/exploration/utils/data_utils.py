import pyreadr
import nfl_data_py as nfl

REFERENCE_CAL_DATA_URL = "https://raw.githubusercontent.com/guga31bb/metrics/master/wp_tuning/cal_data.rds"
REFERENCE_CAL_DATA_RDS_PATH = 'tests/validation/calibration_data/cal_data.rds'
REFERENCE_CAL_DATA_CSV_PATH = 'tests/validation/calibration_data/reference_cal_data.csv'


def import_rds(url, rds_path):
    pyreadr.download_file(url, rds_path)
    result = pyreadr.read_r(rds_path)
    df = result[None]
    df.reset_index(drop=True, inplace=True)
    return df


def save_to_csv(df, path):
    df.to_csv(path, index=False)


def generate_seasons(start, end):
    return list(range(start, end + 1))


def import_pbp_data(start, end):
    seasons = generate_seasons(start, end)
    df = nfl.clean_nfl_data(nfl.import_pbp_data(seasons, thread_requests=True))

    return df