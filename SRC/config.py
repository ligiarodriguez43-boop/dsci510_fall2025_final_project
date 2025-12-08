from pathlib import Path
from dotenv import load_dotenv
import os


# Load .env file
env_path = Path(__file__).resolve().parent.parent / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    print(f"Warning: .env file not found at {env_path}. You must create one with required credentials!")


# Helper function
def get_env(key: str, default=None):
    value = os.getenv(key, default)
    if value is None:
        raise ValueError(f"Missing required environment variable: {key}")
    return value


# Directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / get_env("DATA_DIR", "data")
RESULTS_DIR = BASE_DIR / get_env("RESULTS_DIR", "results")
LOGS_DIR = BASE_DIR / get_env("LOGS_DIR", "logs")

for d in [DATA_DIR, RESULTS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Kaggle Configuration
KAGGLE_USERNAME = get_env("KAGGLE_USERNAME")
KAGGLE_KEY = get_env("KAGGLE_KEY")
KAGGLE_DATASET = get_env("KAGGLE_DATASET")
KAGGLE_FILE = get_env("KAGGLE_FILE")


# API Keys
AHR_API_KEY = os.getenv("AHR_API_KEY", None)
AHR_API_GRAPHQL_ENDPOINT = 'https://api.americashealthrankings.org/graphql'

# Data sources URLs
BRFSS_KAGGLE_URL = "https://www.kaggle.com/datasets/isuruprabath/brfss-2023-csv-dataset?select=BRFSS2023.csv"
HEALTHDATA_GOV_BRFSS = "https://healthdata.gov/CDC/Behavioral-Risk-Factor-Surveillance-System-BRFSS-P/khk6-anfn/about_data"
AHR_GRAPHQL_ENDPOINT = "https://api.americashealthrankings.org/graphql"

# ML Settings
SAMPLE_SIZE = int(os.getenv("SAMPLE_SIZE", 25000))
RANDOM_STATE = int(os.getenv("RANDOM_STATE", 42))

# BRFSS General Health Model
BRFSS_FEATURES = [
    'physhlth', 'menthlth', 'poorhlth',
    'smoke100', 'alcday4', 'exerany2',
    '_bmi5', 'diabete4', 'cvdinfr4', 'asthma3',
    '_age_g', 'educa', 'income3', '_sex',
    'colncncr', 'chccopd3'
]
BRFSS_TARGET = 'genhlth'

NUMERIC_LABELS = {
    'physhlth': 'Physical Health',
    'menthlth': 'Mental Health',
    'poorhlth': 'Poor Health',
    '_bmi5': 'BMI'
}

SIMPLE_LABELS = {
    'diabete4': 'Diabetes',
    'cvdinfr4': 'Heart Attack',
    'asthma3': 'Asthma',
    'colncncr': 'Colon Cancer',
    'chccopd3': 'COPD',
    'smoke100': 'Smoking',
    'alcday4': 'Alcohol',
    'exerany2': 'Exercise',
    '_age_g': 'Age',
    'educa': 'Education',
    'income3': 'Income',
    '_sex': 'Gender'
}

# BRFSS Prevalence Model
PREVALENCE_FEATURES = ["State", "BRFSS_Survey_Questions", "Response", "Year"]
PREVALENCE_TARGET = "Data_value"
START_YEAR = int(os.getenv("START_YEAR", 2016))
END_YEAR = int(os.getenv("END_YEAR", 2023))
PREVALENCE_SAMPLE_SIZE = int(os.getenv("PREVALENCE_SAMPLE_SIZE", 25000))


# US Health Rankings Model
FEATURES_USHR = ["m1", "m2", "m3"]
TARGET_USHR = "m4"
FEATURE_NAMES_MAP_USHR = {
    "m1": "Risk Behaviors (Annual)",
    "m2": "Behaviors",
    "m3": "Behavioral Health"
}
METRIC_IDS_USHR = ["16465", "16535", "17679", "16540"]
