from pipeline.preprocessing.data_utils import get_dataset, update_dataset, read_from_csv, save_to_csv
from pipeline.preprocessing.calibration import (
    get_regular_season_games,
    process_game_epa
)
import pandas as pd
from multiprocessing import Pool
import os


def preprocessing(file_list):
    with Pool(16) as pool:
        print("Reading CSVs...")
        pbp_df_by_season = pool.map(read_from_csv, file_list)
        print("Generating EPA data...")
        processed_pbp_dataframes = pool.map(process_game_epa, pbp_df_by_season)
        # combine all the dataframes into one
        print(len(processed_pbp_dataframes))
        print(processed_pbp_dataframes[1].head())
        epa_cal_data = pd.concat(processed_pbp_dataframes, ignore_index=True)
        # group by season
        grouped = epa_cal_data.groupby('season')
        # create a list of tuples with season, file_prefix, and the dataframe
        df_tuples = [(season, 'epa_cal_data', group) for season, group in grouped]
        pool.map(save_to_csv, df_tuples)


def main(args):
    years = list(range(args.start_year, args.end_year + 1, 1))
    file_prefix = 'epa_cal_data'
    os.makedirs(file_prefix, exist_ok=True)
    print(f"Importing Play-by-Play Data for the following years:\n{years}")
    pbp_csv_paths = get_dataset(args.pbp_dataset)
    print("Preprocessing Play-by-Play Data...")
    preprocessing(pbp_csv_paths)

    print("Updating EPA Calibration dataset...")
    update_dataset(
        dataset_name=args.dataset,
        dataset_project=args.project,
        file_prefix=file_prefix
    )


if __name__ == '__main__':
    import argparse
    from textwrap import dedent

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Generate Vegas Win Probability Calibration Data',
        epilog=dedent('''\
            FURTHER READING
            ---------------

            Subject                             Link
            -------                             ----
            Field Descriptions:                 https://www.nflfastr.com/articles/field_descriptions.html
        ''')
    )

    meta_var = 'YYYY'

    # Add arguments for all hyperparameters
    parser.add_argument('-s', '--start-year', type=int, metavar=meta_var, required=False, default=1999,
                        help='Starting Season.  Default: 1999')
    parser.add_argument('-e', '--end-year', type=int, metavar=meta_var, required=False, default=2023,
                        help='Ending Season.  Default: 2023')
    parser.add_argument('-d', '--dataset', type=str, metavar='DATASET_NAME', required=False, default="EPA Calibration Data",
                        help='Dataset Name.  Default: EPA Calibration Data')
    parser.add_argument('-i', '--pbp-dataset', type=str, metavar='PBP_DATA', required=False, default="NFL Play-by-Play Data", help='Project Name.  Default: NFL Play-by-Play Data')
    parser.add_argument('-p', '--project', type=str, metavar='PROJECT_NAME', required=False, default="NFL WP Model", help='Project Name.  Default: NFL WP Model')

    args = parser.parse_args()
    main(args)