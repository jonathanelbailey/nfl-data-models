from pipeline.preprocessing.calibration import generate_vegas_wp_calibration_data
from pipeline.preprocessing.data_utils import read_from_csv, save_to_csv, get_dataset, update_dataset
import pandas as pd
from multiprocessing import Pool
import os


def preprocessing(file_list):
    with Pool() as pool:
        dataframes = pool.map(read_from_csv, file_list)
        print(dataframes[0].head())
        print("generating calibration data")
        preprocessing_dfs = pool.map(generate_vegas_wp_calibration_data, dataframes)
        cal_data = pd.concat(preprocessing_dfs, ignore_index=True)
        grouped = cal_data.groupby('season')
        df_tuples = [(season, 'cal_data', group) for season, group in grouped]
        pool.map(save_to_csv, df_tuples)


def main(args):
    years = list(range(args.start_year, args.end_year + 1, 1))
    file_prefix = 'cal_data'
    os.makedirs(file_prefix, exist_ok=True)
    print(f"Importing Play-by-Play Data for the following years:\n{years}")
    pbp_csv_paths = get_dataset(args.pbp_dataset)
    print("Preprocessing Play-by-Play Data...")
    preprocessing(pbp_csv_paths)

    print("Getting calibration dataset...")
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
    parser.add_argument('-d', '--dataset', type=str, metavar='DATASET_NAME', required=False, default="Vegas WP Calibration Data",
                        help='Dataset Name.  Default: Vegas WP Calibration Data')
    parser.add_argument('-i', '--pbp-dataset', type=str, metavar='PBP_DATA', required=False, default="NFL Play-by-Play Data", help='PBP Dataset Name.  Default: NFL Play-by-Play Data')
    parser.add_argument('-p', '--project', type=str, metavar='PROJECT_NAME', required=False, default="NFL WP Model", help='Project Name.  Default: NFL WP Model')

    args = parser.parse_args()
    main(args)