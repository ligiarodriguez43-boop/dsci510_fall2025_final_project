import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, f1_score, mean_absolute_error, \
    mean_squared_error, median_absolute_error, r2_score, explained_variance_score
import requests

# Config
from config import *


# Helper functions
def load_csv(file_env_var, fallback_name):
    file_name = get_env(file_env_var, fallback_name)
    full_path = DATA_DIR / file_name
    if not full_path.exists():
        raise FileNotFoundError(f"File not found: {full_path}")
    return pd.read_csv(full_path)


# BRFSS General Health RandomForestClassifier ML Model
def run_brfss_general_health():
    df = load_csv("KAGGLE_FILE", "BRFSS2023.csv")

    if len(df) > SAMPLE_SIZE:
        df = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE)

    df = df[df[BRFSS_TARGET].isin([1, 2, 3, 4, 5])]
    y = df[BRFSS_TARGET]
    X = df[BRFSS_FEATURES]

    numeric = [f for f in BRFSS_FEATURES if f in NUMERIC_LABELS]
    categorical = [f for f in BRFSS_FEATURES if f in SIMPLE_LABELS]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numeric),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore'))
            ]), categorical)
        ]
    )

    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=400, random_state=RANDOM_STATE, n_jobs=-1))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Metrics
    print(classification_report(y_test, y_pred))
    print("Macro F1 Score:", round(f1_score(y_test, y_pred, average='macro'), 3))

    # Confusion Matrix
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap="Greens", normalize="true")
    plt.title("Confusion Matrix - General Health")
    plt.show()

    # Feature Importance
    importances = model.named_steps['classifier'].feature_importances_
    feature_names = numeric + list(
        model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['encoder'].get_feature_names_out(
            categorical))
    fi_df = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values("importance",
                                                                                            ascending=False)

    sns.barplot(x="importance", y="feature", data=fi_df.head(20))
    plt.title("Top 20 Feature Importances - General Health")
    plt.show()


# BRFSS Prevalence RandomForestRegressor ML Model
def run_brfss_prevalence():
    df = load_csv("BRFSS_PREVALENCE_FILE", "BRFSS_Prevalence.csv")

    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["Data_value"] = pd.to_numeric(df["Data_value"], errors="coerce")
    df = df.dropna(subset=["Year", "Data_value", "Sample_Size"])
    df = df[(df["Year"] >= START_YEAR) & (df["Year"] <= END_YEAR)]

    import us
    states = [s.abbr for s in us.states.STATES]
    df = df[df["Locationabbr"].isin(states)]

    if len(df) > PREVALENCE_SAMPLE_SIZE:
        df = df.sample(n=PREVALENCE_SAMPLE_SIZE, random_state=RANDOM_STATE)

    df["State"] = LabelEncoder().fit_transform(df["Locationabbr"])
    df["Response"] = LabelEncoder().fit_transform(df["Response"])
    df["BRFSS_Survey_Questions"] = LabelEncoder().fit_transform(df["Topic"])

    X = df[PREVALENCE_FEATURES]
    y = df[PREVALENCE_TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Metrics
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))
    print("Median AE:", median_absolute_error(y_test, y_pred))
    print("R^2:", r2_score(y_test, y_pred))
    print("Explained Variance Score:", explained_variance_score(y_test, y_pred))

    # Actual vs Predicted
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("BRFSS Prevalence: Actual vs Predicted")
    plt.show()

    # Feature Importance
    fi_df = pd.DataFrame({"feature": X.columns, "importance": model.feature_importances_}).sort_values("importance",
                                                                                                       ascending=False)
    sns.barplot(x="importance", y="feature", data=fi_df.head(20))
    plt.title("Top 20 Feature Importances - BRFSS Prevalence")
    plt.show()


# US Health Rankings RandomForestRegressor ML Model
def run_ushr_model():
    if AHR_API_KEY is None:
        raise ValueError("Missing AHR_API_KEY in .env.example.txt")

    headers = {"Content-Type": "application/json", "X-Api-Key": AHR_API_KEY}
    query = """
    query GetMeasureData {
      m1: measure_A(metricId: 16465) { data { dateLabel rank state value } }
      m2: measure_A(metricId: 16535) { data { dateLabel rank state value } }
      m3: measure_A(metricId: 17679) { data { dateLabel rank state value } }
      m4: measure_A(metricId: 16540) { data { dateLabel rank state value } }
    }
    """

    response = requests.post(AHR_API_GRAPHQL_ENDPOINT, headers=headers, json={"query": query})
    if response.status_code != 200:
        raise RuntimeError(f"AHR API request failed: {response.text}")

    data = response.json()["data"]
    cleaned = {}
    for m in ["m1", "m2", "m3", "m4"]:
        df_m = pd.DataFrame(data[m]["data"])
        cleaned[m] = df_m[(df_m["state"] != "ALL") & df_m["rank"].notnull() & df_m["value"].notnull()]

    df = (
        cleaned["m1"][["state", "value"]].rename(columns={"value": "m1"})
        .merge(cleaned["m2"][["state", "value"]].rename(columns={"value": "m2"}), on="state")
        .merge(cleaned["m3"][["state", "value"]].rename(columns={"value": "m3"}), on="state")
        .merge(cleaned["m4"][["state", "value"]].rename(columns={"value": "m4"}), on="state")
    )

    X = df[FEATURES_USHR]
    y = df[TARGET_USHR]
    model = RandomForestRegressor(random_state=RANDOM_STATE)
    model.fit(X, y)
    df["predict_score"] = model.predict(X)
    df["predict_rank"] = df["predict_score"].rank(ascending=False)
    print(df.sort_values("predict_rank"))

    # Plot predicted scores by state
    sns.barplot(x="state", y="predict_score", data=df.sort_values("predict_score", ascending=False))
    plt.xticks(rotation=90)
    plt.title("Predicted US Health Scores by State")
    plt.show()

    # Feature Importance
    fi_df = pd.DataFrame({"feature": FEATURES_USHR, "importance": model.feature_importances_}).sort_values("importance",
                                                                                                           ascending=False)
    sns.barplot(x="importance", y="feature", data=fi_df)
    plt.title("Feature Importances - US Health Rankings")
    plt.show()

# Main
if __name__ == "__main__":
    run_brfss_general_health()
    run_brfss_prevalence()
    run_ushr_model()
