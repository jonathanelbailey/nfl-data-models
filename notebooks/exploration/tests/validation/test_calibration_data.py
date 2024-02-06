import unittest
import pandas as pd


REFERENCE_CAL_DATA_CSV_PATH = 'tests/validation/calibration_data/reference_cal_data.csv'
CAL_DATA_PATH = 'tests/validation/calibration_data/cal_data.csv'


class TestWPCalDataValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_df = pd.read_csv(CAL_DATA_PATH)
        cls.reference_df = pd.read_csv(REFERENCE_CAL_DATA_CSV_PATH)

    def test_compare_shape_to_reference(self):
        self.assertEqual(self.test_df.shape, self.reference_df.shape, "Shape does not match")

    def test_compare_columns_to_reference(self):
        self.assertEqual(self.test_df.columns.tolist(), self.reference_df.columns.tolist(), "Columns do not match")

    def test_compare_column_types_to_reference(self):
        self.assertDictEqual(self.test_df.dtypes.to_dict(), self.reference_df.dtypes.to_dict())

    def test_compare_missing_values_in_each_column_to_reference(self):
        for col in self.test_df.columns:
            self.assertLessEqual(self.test_df[col].isnull().sum(), self.reference_df[col].isnull().sum(), f"Column {col} has different number of missing values")
