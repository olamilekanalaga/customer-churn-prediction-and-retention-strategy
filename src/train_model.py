from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "telco_customer_churn.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "telco_customer_churn_clean.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.joblib"
METRICS_PATH = PROJECT_ROOT / "reports" / "model_metrics.json"
SUMMARY_PATH = PROJECT_ROOT / "reports" / "business_summary.json"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"


NUMERIC_FEATURES = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
TARGET = "Churn"


def load_and_clean_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)
    df[TARGET] = df[TARGET].map({"No": 0, "Yes": 1})
    return df


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def build_models() -> dict[str, Pipeline]:
    preprocessor = build_preprocessor()
    return {
        "Logistic Regression": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "Decision Tree": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "model",
                    DecisionTreeClassifier(
                        max_depth=5,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=8,
                        min_samples_leaf=8,
                        class_weight="balanced",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def evaluate_model(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]
    return {
        "accuracy": round(accuracy_score(y_test, predictions), 4),
        "precision": round(precision_score(y_test, predictions), 4),
        "recall": round(recall_score(y_test, predictions), 4),
        "f1": round(f1_score(y_test, predictions), 4),
        "roc_auc": round(roc_auc_score(y_test, probabilities), 4),
    }


def get_feature_importance(model: Pipeline) -> pd.DataFrame:
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()

    if hasattr(classifier, "feature_importances_"):
        values = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        values = classifier.coef_[0]
    else:
        return pd.DataFrame(columns=["feature", "importance"])

    importance = pd.DataFrame({"feature": feature_names, "importance": values})
    importance["abs_importance"] = importance["importance"].abs()
    return importance.sort_values("abs_importance", ascending=False).head(15)


def create_figures(
    df: pd.DataFrame,
    best_model: Pipeline,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    feature_importance: pd.DataFrame,
) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(7, 5))
    churn_rates = df.groupby("Contract")[TARGET].mean().sort_values(ascending=False)
    sns.barplot(x=churn_rates.index, y=churn_rates.values, color="#2563eb")
    plt.title("Churn Rate by Contract Type")
    plt.ylabel("Churn Rate")
    plt.xlabel("Contract Type")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "churn_by_contract.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 5))
    sns.kdeplot(data=df, x="tenure", hue=TARGET, common_norm=False, fill=True, alpha=0.35)
    plt.title("Tenure Distribution by Churn Outcome")
    plt.xlabel("Tenure in Months")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "tenure_by_churn.png", dpi=180)
    plt.close()

    probabilities = best_model.predict_proba(x_test)[:, 1]
    predictions = best_model.predict(x_test)

    plt.figure(figsize=(6, 5))
    ConfusionMatrixDisplay(confusion_matrix(y_test, predictions)).plot(
        cmap="Blues", colorbar=False
    )
    plt.title("Best Model Confusion Matrix")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=180)
    plt.close()

    plt.figure(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_test, probabilities)
    plt.title("Best Model ROC Curve")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "roc_curve.png", dpi=180)
    plt.close()

    if not feature_importance.empty:
        cleaned = feature_importance.copy()
        cleaned["feature"] = cleaned["feature"].str.replace("num__", "", regex=False)
        cleaned["feature"] = cleaned["feature"].str.replace("cat__", "", regex=False)
        cleaned["feature"] = cleaned["feature"].str.replace("_", " ", regex=False)
        plt.figure(figsize=(8, 6))
        sns.barplot(data=cleaned, y="feature", x="abs_importance", color="#059669")
        plt.title("Top Model Drivers")
        plt.xlabel("Absolute Importance")
        plt.ylabel("")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "feature_importance.png", dpi=180)
        plt.close()


def build_business_summary(df: pd.DataFrame, metrics: dict[str, dict[str, float]]) -> dict:
    churn_rate = df[TARGET].mean()
    monthly_revenue_at_risk = df.loc[df[TARGET] == 1, "MonthlyCharges"].sum()
    high_risk_segments = {
        "month_to_month_churn_rate": round(
            df.loc[df["Contract"] == "Month-to-month", TARGET].mean(), 4
        ),
        "electronic_check_churn_rate": round(
            df.loc[df["PaymentMethod"] == "Electronic check", TARGET].mean(), 4
        ),
        "fiber_optic_churn_rate": round(
            df.loc[df["InternetService"] == "Fiber optic", TARGET].mean(), 4
        ),
    }
    return {
        "customers": int(len(df)),
        "overall_churn_rate": round(churn_rate, 4),
        "observed_monthly_revenue_at_risk": round(float(monthly_revenue_at_risk), 2),
        "best_model": max(metrics, key=lambda name: metrics[name]["roc_auc"]),
        "high_risk_segments": high_risk_segments,
    }


def main() -> None:
    df = load_and_clean_data()
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)

    x = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    metrics = {}
    trained_models = {}
    for name, model in build_models().items():
        model.fit(x_train, y_train)
        metrics[name] = evaluate_model(model, x_test, y_test)
        trained_models[name] = model

    best_model_name = max(metrics, key=lambda name: metrics[name]["roc_auc"])
    best_model = trained_models[best_model_name]
    feature_importance = get_feature_importance(best_model)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": best_model,
            "model_name": best_model_name,
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "metrics": metrics[best_model_name],
        },
        MODEL_PATH,
    )

    create_figures(df, best_model, x_test, y_test, feature_importance)

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    SUMMARY_PATH.write_text(
        json.dumps(build_business_summary(df, metrics), indent=2),
        encoding="utf-8",
    )
    feature_importance.to_csv(PROJECT_ROOT / "reports" / "feature_importance.csv", index=False)

    print(f"Best model: {best_model_name}")
    print(json.dumps(metrics[best_model_name], indent=2))


if __name__ == "__main__":
    main()
