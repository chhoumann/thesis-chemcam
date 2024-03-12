import unittest
import pandas as pd

from lib.full_flow_dataloader import load_and_scale_data


class TestNorm1Scaler(unittest.TestCase):
    def test_equals1(self):
        train, test = load_and_scale_data(1)

        self._sumEquals(train, 1)
        self._sumEquals(test, 1)

    def test_equals3(self):
        train, test = load_and_scale_data(3)

        self._sumEquals(train, 3)
        self._sumEquals(test, 3)

    def _sumEquals(self, df: pd.DataFrame, n: float):
        columns_numeric = pd.to_numeric(df.columns, errors="coerce")
        numeric_col_names = df.columns[~columns_numeric.isna()]

        # Select columns in the DataFrame that have numeric names
        df = df[numeric_col_names]

        self.assertEqual(df.sum().sum(), n)


if __name__ == "__main__":
    unittest.main()
