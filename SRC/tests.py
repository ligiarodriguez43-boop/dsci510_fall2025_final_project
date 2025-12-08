import unittest
import os
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from config import (
    BRFSS_FEATURES, BRFSS_TARGET,
    PREVALENCE_FEATURES, PREVALENCE_TARGET,
    FEATURES_USHR, TARGET_USHR,
    SAMPLE_SIZE, RANDOM_STATE
)
from data_loader import load_brfss, load_prevalence_data, load_ushr_data  # assume you have these

class TestDataLoaders(unittest.TestCase):
    def test_brfss_loader(self):
        """Test that BRFSS dataset loads correctly."""
        kaggle_user = os.getenv("KAGGLE_USERNAME")
        kaggle_key = os.getenv("KAGGLE_KEY")
        self.assertIsNotNone(kaggle_user, "KAGGLE_USERNAME not set")
        self.assertIsNotNone(kaggle_key, "KAGGLE_KEY not set")

        df = load_brfss(kaggle_user, kaggle_key)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue(all(col in df.columns for col in BRFSS_FEATURES + [BRFSS_TARGET]))

    def test_prevalence_loader(self):
        """Test that BRFSS prevalence dataset loads correctly."""
        df = load_prevalence_data()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue(all(col in df.columns for col in PREVALENCE_FEATURES + [PREVALENCE_TARGET]))

    def test_ushr_loader(self):
        """Test that US Health Rankings dataset loads correctly."""
        df = load_ushr_data()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue(all(col in df.columns for col in FEATURES_USHR + [TARGET_USHR]))

class TestMLPipelines(unittest.TestCase):
    def test_brfss_pipeline(self):
        """Test that a Random Forest pipeline can fit BRFSS data."""
        df = pd.DataFrame({
            'physhlth': [1, 2, 3],
            'menthlth': [1, 2, 3],
            'poorhlth': [1, 2, 3],
            'smoke100': [0, 1, 0],
            'alcday4': [0, 1, 1],
            'exerany2': [1, 0, 1],
            '_bmi5': [25, 30, 28],
            'diabete4': [0, 1, 0],
            'cvdinfr4': [0, 1, 0],
            'asthma3': [0, 1, 0],
            '_age_g': [1, 2, 3],
            'educa': [1, 2, 3],
            'income3': [1, 2, 3],
            '_sex': [1, 0, 1],
            'colncncr': [0, 1, 0],
            'chccopd3': [0, 1, 0],
            'genhlth': [1, 2, 3]
        })
        X = df[BRFSS_FEATURES]
        y = df[BRFSS_TARGET]

        model = RandomForestClassifier(n_estimators=10, random_state=RANDOM_STATE)
        model.fit(X, y)
        preds = model.predict(X)
        self.assertEqual(len(preds), len(y))

    def test_prevalence_pipeline(self):
        """Test Random Forest Regressor on prevalence data."""
        df = pd.DataFrame({
            'State': [0,1,2],
            'BRFSS_Survey_Questions': [0,1,2],
            'Response': [0,1,2],
            'Year': [2016,2017,2018],
            'Data_value': [10,20,30]
        })
        X = df[PREVALENCE_FEATURES]
        y = df[PREVALENCE_TARGET]

        model = RandomForestRegressor(n_estimators=10, random_state=RANDOM_STATE)
        model.fit(X, y)
        preds = model.predict(X)
        self.assertEqual(len(preds), len(y))

if __name__ == '__main__':
    unittest.main()
