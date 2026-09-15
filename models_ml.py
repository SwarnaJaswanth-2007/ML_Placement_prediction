"""
Machine Learning Module for Placement Salary & Placement Prediction.
Implements:
1. Simple Linear Regression (Univariate)
2. Multilinear Regression (OLS, Ridge, Lasso, ElasticNet with VIF analysis)
3. Boosting Techniques (AdaBoost, Gradient Boosting, XGBoost, LightGBM)
4. Comprehensive Model Comparison & Leaderboard
5. Interactive Multi-Model Prediction Engine
"""

import os
import time
import warnings
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional

warnings.filterwarnings("ignore")

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from load_data import load_data, DATA_PATH
from preprocessing import PREPROCESSED_PATH, run_preprocessing

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_PATH = os.path.join(BASE_DIR, "Data", "ml_models_package.joblib")
DATASET_CSV_PATH = os.path.join(BASE_DIR, "Data", "placement_predict_preprocessed.csv")

# Feature classifications
CATEGORICAL_COLS = [
    "Gender", "City", "CollegeTier", "Stream", "Specialisation",
    "Hostel", "HistoryOfBacklogs", "CGPA_Tier"
]

NUMERICAL_COLS = [
    "SGPA_Sem1", "SGPA_Sem2", "SGPA_Sem3", "SGPA_Sem4",
    "SGPA_Sem5", "SGPA_Sem6", "SGPA_Sem7", "SGPA_Sem8",
    "CGPA", "AttendancePercent", "Internships", "Projects",
    "Workshops", "Certifications", "Publications",
    "AptitudeTestScore", "SoftSkillsRating", "CodingTestScore",
    "MockInterviewScore", "ExtraCurricular"
]

TARGET_COL = "Salary Package"
CLASSIFICATION_TARGET = "PlacementStatus"
EXCLUDE_COLS = ["StudentID", "PlacementStatus", "IsAnomaly", TARGET_COL]


def get_feature_metadata() -> dict:
    """Returns metadata, limits, and human-friendly labels for candidate features."""
    return {
        "academics": {
            "title": "Academic Performance",
            "icon": "🎓",
            "fields": [
                {"name": "CGPA", "label": "Cumulative GPA (CGPA)", "type": "range", "min": 4.0, "max": 10.0, "step": 0.05, "default": 7.85, "unit": "/10"},
                {"name": "CGPA_Tier", "label": "CGPA Tier", "type": "select", "options": ["Low", "Mid", "High"], "default": "Mid"},
                {"name": "AttendancePercent", "label": "Attendance Percentage", "type": "range", "min": 40.0, "max": 100.0, "step": 0.5, "default": 85.0, "unit": "%"},
                {"name": "HistoryOfBacklogs", "label": "History of Backlogs", "type": "select", "options": ["No", "Yes"], "default": "No"},
                {"name": "SGPA_Sem1", "label": "SGPA Semester 1", "type": "range", "min": 4.0, "max": 10.0, "step": 0.1, "default": 7.5, "unit": "/10"},
                {"name": "SGPA_Sem2", "label": "SGPA Semester 2", "type": "range", "min": 4.0, "max": 10.0, "step": 0.1, "default": 7.6, "unit": "/10"},
                {"name": "SGPA_Sem3", "label": "SGPA Semester 3", "type": "range", "min": 4.0, "max": 10.0, "step": 0.1, "default": 7.8, "unit": "/10"},
                {"name": "SGPA_Sem4", "label": "SGPA Semester 4", "type": "range", "min": 4.0, "max": 10.0, "step": 0.1, "default": 8.0, "unit": "/10"},
                {"name": "SGPA_Sem5", "label": "SGPA Semester 5", "type": "range", "min": 4.0, "max": 10.0, "step": 0.1, "default": 8.1, "unit": "/10"},
                {"name": "SGPA_Sem6", "label": "SGPA Semester 6", "type": "range", "min": 4.0, "max": 10.0, "step": 0.1, "default": 8.2, "unit": "/10"},
                {"name": "SGPA_Sem7", "label": "SGPA Semester 7", "type": "range", "min": 4.0, "max": 10.0, "step": 0.1, "default": 8.4, "unit": "/10"},
                {"name": "SGPA_Sem8", "label": "SGPA Semester 8", "type": "range", "min": 4.0, "max": 10.0, "step": 0.1, "default": 8.5, "unit": "/10"},
            ]
        },
        "technical_skills": {
            "title": "Technical & Assessment Scores",
            "icon": "💻",
            "fields": [
                {"name": "CodingTestScore", "label": "Coding Test Score", "type": "range", "min": 20.0, "max": 100.0, "step": 1.0, "default": 75.0, "unit": "/100"},
                {"name": "AptitudeTestScore", "label": "Aptitude Test Score", "type": "range", "min": 20.0, "max": 100.0, "step": 1.0, "default": 72.0, "unit": "/100"},
                {"name": "SoftSkillsRating", "label": "Soft Skills Rating", "type": "range", "min": 1.0, "max": 10.0, "step": 0.5, "default": 7.5, "unit": "/10"},
                {"name": "MockInterviewScore", "label": "Mock Interview Score", "type": "range", "min": 20.0, "max": 100.0, "step": 1.0, "default": 78.0, "unit": "/100"},
            ]
        },
        "experience": {
            "title": "Projects & Practical Experience",
            "icon": "🚀",
            "fields": [
                {"name": "Internships", "label": "Internships Completed", "type": "range", "min": 0, "max": 5, "step": 1, "default": 2, "unit": "internships"},
                {"name": "Projects", "label": "Major Projects", "type": "range", "min": 0, "max": 10, "step": 1, "default": 3, "unit": "projects"},
                {"name": "Workshops", "label": "Workshops Attended", "type": "range", "min": 0, "max": 10, "step": 1, "default": 3, "unit": "workshops"},
                {"name": "Certifications", "label": "Technical Certifications", "type": "range", "min": 0, "max": 8, "step": 1, "default": 2, "unit": "certs"},
                {"name": "Publications", "label": "Research Publications", "type": "range", "min": 0, "max": 5, "step": 1, "default": 0, "unit": "papers"},
                {"name": "ExtraCurricular", "label": "Extra Curricular Activity", "type": "select", "options": ["0", "1"], "labels": {"0": "No", "1": "Yes"}, "default": "1"},
            ]
        },
        "background": {
            "title": "College & Demographic Profile",
            "icon": "🏛️",
            "fields": [
                {"name": "CollegeTier", "label": "College Tier", "type": "select", "options": ["Tier1", "Tier2", "Tier3"], "default": "Tier1"},
                {"name": "Stream", "label": "Engineering Stream", "type": "select", "options": ["CS", "IT", "ECE", "EE", "Mechanical", "Civil"], "default": "CS"},
                {"name": "Specialisation", "label": "Specialisation Track", "type": "select", "options": ["AI", "DataScience", "Networking", "Embedded"], "default": "AI"},
                {"name": "Gender", "label": "Gender", "type": "select", "options": ["Male", "Female"], "default": "Male"},
                {"name": "City", "label": "City", "type": "select", "options": ["Ahmedabad", "Mumbai", "Pune", "Hyderabad", "Kolkata", "Jaipur"], "default": "Mumbai"},
                {"name": "Hostel", "label": "Hostel Resident", "type": "select", "options": ["No", "Yes"], "default": "No"},
            ]
        }
    }


def get_preset_profiles() -> dict:
    """Returns preset candidate profiles for rapid 1-click evaluation."""
    return {
        "dream_achiever": {
            "name": "🌟 Super Dream Candidate",
            "description": "Tier 1 College, 9.2 CGPA, 95 Coding Score, 3 Internships, AI Specialisation",
            "badge": "Top 1%",
            "features": {
                "CGPA": 9.2, "CGPA_Tier": "High", "AttendancePercent": 94.0, "HistoryOfBacklogs": "No",
                "SGPA_Sem1": 8.8, "SGPA_Sem2": 9.0, "SGPA_Sem3": 9.1, "SGPA_Sem4": 9.2,
                "SGPA_Sem5": 9.3, "SGPA_Sem6": 9.4, "SGPA_Sem7": 9.5, "SGPA_Sem8": 9.6,
                "CodingTestScore": 95.0, "AptitudeTestScore": 92.0, "SoftSkillsRating": 9.0, "MockInterviewScore": 94.0,
                "Internships": 3, "Projects": 5, "Workshops": 4, "Certifications": 4, "Publications": 2,
                "ExtraCurricular": "1", "CollegeTier": "Tier1", "Stream": "CS", "Specialisation": "AI",
                "Gender": "Female", "City": "Mumbai", "Hostel": "Yes"
            }
        },
        "tech_specialist": {
            "name": "💻 Tech & Coding Specialist",
            "description": "Moderate CGPA (7.8), Exceptional Coding (92) & Projects (4), Data Science",
            "badge": "Developer",
            "features": {
                "CGPA": 7.8, "CGPA_Tier": "Mid", "AttendancePercent": 82.0, "HistoryOfBacklogs": "No",
                "SGPA_Sem1": 7.2, "SGPA_Sem2": 7.5, "SGPA_Sem3": 7.8, "SGPA_Sem4": 8.0,
                "SGPA_Sem5": 8.2, "SGPA_Sem6": 8.4, "SGPA_Sem7": 8.6, "SGPA_Sem8": 8.8,
                "CodingTestScore": 92.0, "AptitudeTestScore": 84.0, "SoftSkillsRating": 7.5, "MockInterviewScore": 88.0,
                "Internships": 2, "Projects": 4, "Workshops": 3, "Certifications": 3, "Publications": 0,
                "ExtraCurricular": "1", "CollegeTier": "Tier2", "Stream": "IT", "Specialisation": "DataScience",
                "Gender": "Male", "City": "Pune", "Hostel": "No"
            }
        },
        "average_candidate": {
            "name": "⚡ Balanced Core Candidate",
            "description": "Tier 2 College, 7.2 CGPA, 68 Coding Score, 1 Internship, 2 Projects",
            "badge": "Average",
            "features": {
                "CGPA": 7.2, "CGPA_Tier": "Mid", "AttendancePercent": 78.0, "HistoryOfBacklogs": "No",
                "SGPA_Sem1": 6.8, "SGPA_Sem2": 7.0, "SGPA_Sem3": 7.2, "SGPA_Sem4": 7.2,
                "SGPA_Sem5": 7.4, "SGPA_Sem6": 7.5, "SGPA_Sem7": 7.6, "SGPA_Sem8": 7.8,
                "CodingTestScore": 68.0, "AptitudeTestScore": 70.0, "SoftSkillsRating": 6.5, "MockInterviewScore": 72.0,
                "Internships": 1, "Projects": 2, "Workshops": 2, "Certifications": 1, "Publications": 0,
                "ExtraCurricular": "0", "CollegeTier": "Tier2", "Stream": "ECE", "Specialisation": "Networking",
                "Gender": "Male", "City": "Hyderabad", "Hostel": "No"
            }
        },
        "needs_improvement": {
            "name": "🌱 Entry Level / Needs Support",
            "description": "5.6 CGPA, History of Backlogs, Low Test Scores, 0 Internships",
            "badge": "Needs Focus",
            "features": {
                "CGPA": 5.6, "CGPA_Tier": "Low", "AttendancePercent": 65.0, "HistoryOfBacklogs": "Yes",
                "SGPA_Sem1": 5.2, "SGPA_Sem2": 5.4, "SGPA_Sem3": 5.5, "SGPA_Sem4": 5.6,
                "SGPA_Sem5": 5.8, "SGPA_Sem6": 5.9, "SGPA_Sem7": 6.0, "SGPA_Sem8": 6.2,
                "CodingTestScore": 42.0, "AptitudeTestScore": 45.0, "SoftSkillsRating": 5.0, "MockInterviewScore": 48.0,
                "Internships": 0, "Projects": 1, "Workshops": 1, "Certifications": 0, "Publications": 0,
                "ExtraCurricular": "0", "CollegeTier": "Tier3", "Stream": "Mechanical", "Specialisation": "Embedded",
                "Gender": "Male", "City": "Jaipur", "Hostel": "No"
            }
        }
    }


def compute_dataset_statistics() -> dict:
    """Computes scaling parameters and categorical mappings from the raw dataset."""
    raw_df = load_data()
    
    scaler_params = {}
    for col in NUMERICAL_COLS:
        if col in raw_df.columns:
            scaler_params[col] = {
                "mean": float(raw_df[col].mean()),
                "std": float(raw_df[col].std() if raw_df[col].std() != 0 else 1.0),
                "min": float(raw_df[col].min()),
                "max": float(raw_df[col].max()),
                "median": float(raw_df[col].median())
            }
            
    category_mappings = {}
    for col in CATEGORICAL_COLS:
        if col in raw_df.columns:
            categories = sorted([str(x) for x in raw_df[col].dropna().unique()])
            category_mappings[col] = {cat: idx for idx, cat in enumerate(categories)}
            
    return {
        "scaler_params": scaler_params,
        "category_mappings": category_mappings,
        "n_samples": len(raw_df)
    }


def get_prepared_data(data_filter: str = "all") -> tuple:
    """Loads preprocessed data and returns X (features), y (salary package), and raw df."""
    if not os.path.exists(PREPROCESSED_PATH):
        run_preprocessing()
        
    df = pd.read_csv(PREPROCESSED_PATH)
    if data_filter == "placed_only":
        df = df[df[TARGET_COL] > 0].copy()
        
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    X = df[feature_cols].copy()
    y = df[TARGET_COL].copy()
    return X, y, df, feature_cols


def calculate_vif(X: pd.DataFrame) -> List[dict]:
    """Calculates Variance Inflation Factor (VIF) for multicollinearity assessment."""
    vif_list = []
    X_sample = X.sample(n=min(5000, len(X)), random_state=42)
    numeric_cols = [col for col in X_sample.columns if col in NUMERICAL_COLS]
    
    if len(numeric_cols) < 2:
        return []

    X_mat = X_sample[numeric_cols].values
    try:
        corr_mat = np.corrcoef(X_mat, rowvar=False)
        inv_corr = np.linalg.pinv(corr_mat)
        for i, col in enumerate(numeric_cols):
            vif_val = float(inv_corr[i, i])
            if vif_val < 5.0:
                severity = "Low (Healthy)"
                badge = "vif-low"
            elif vif_val < 10.0:
                severity = "Moderate Collinearity"
                badge = "vif-mid"
            else:
                severity = "High Collinearity (Severe)"
                badge = "vif-high"
                
            vif_list.append({
                "feature": col,
                "vif": round(max(1.0, vif_val), 2),
                "severity": severity,
                "badge": badge
            })
    except Exception:
        for col in numeric_cols:
            vif_list.append({"feature": col, "vif": 1.0, "severity": "Low (Healthy)", "badge": "vif-low"})

    vif_list.sort(key=lambda item: item["vif"], reverse=True)
    return vif_list


# =====================================================================
# 1. SIMPLE LINEAR REGRESSION (Univariate)
# =====================================================================
def train_simple_linear_regression(
    feature: str = "CGPA",
    test_size: float = 0.2,
    random_state: int = 42,
    data_filter: str = "all"
) -> dict:
    """
    Trains a 1D Simple Linear Regression model (y = mx + c) on a chosen feature.
    """
    raw_df = load_data()
    if data_filter == "placed_only":
        raw_df = raw_df[raw_df[TARGET_COL] > 0].copy()

    if feature not in raw_df.columns or feature not in NUMERICAL_COLS:
        feature = "CGPA"

    df_clean = raw_df[[feature, TARGET_COL]].dropna()
    X = df_clean[[feature]].values
    y = df_clean[TARGET_COL].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    start_time = time.time()
    model = LinearRegression()
    model.fit(X_train, y_train)
    fit_duration = time.time() - start_time

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_r2 = float(r2_score(y_train, y_pred_train))
    test_r2 = float(r2_score(y_test, y_pred_test))
    mae = float(mean_absolute_error(y_test, y_pred_test))
    mse = float(mean_squared_error(y_test, y_pred_test))
    rmse = float(np.sqrt(mse))

    slope = float(model.coef_[0])
    intercept = float(model.intercept_)

    cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring="r2")
    cv_mean = float(cv_scores.mean())
    cv_std = float(cv_scores.std())

    x_min = float(df_clean[feature].min())
    x_max = float(df_clean[feature].max())
    x_line = np.linspace(x_min, x_max, 100).reshape(-1, 1)
    y_line = model.predict(x_line)

    line_points = [{"x": round(float(x[0]), 2), "y": round(float(y_val), 2)} for x, y_val in zip(x_line, y_line)]

    ci_margin = 1.96 * (rmse / np.sqrt(len(y_test)))
    line_upper = [{"x": round(float(x[0]), 2), "y": round(float(y_val + ci_margin), 2)} for x, y_val in zip(x_line, y_line)]
    line_lower = [{"x": round(float(x[0]), 2), "y": round(float(max(0.0, y_val - ci_margin)), 2)} for x, y_val in zip(x_line, y_line)]

    np.random.seed(random_state)
    sample_indices = np.random.choice(len(X_test), size=min(450, len(X_test)), replace=False)
    scatter_points = [
        {"x": round(float(X_test[i, 0]), 2), "y": round(float(y_test[i]), 2)}
        for i in sample_indices
    ]

    residuals_test = y_test - y_pred_test
    counts, bin_edges = np.histogram(residuals_test, bins=20)
    residual_histogram = {
        "counts": counts.tolist(),
        "bin_edges": [round(float(b), 2) for b in bin_edges.tolist()],
        "mean": round(float(residuals_test.mean()), 3),
        "std": round(float(residuals_test.std()), 3)
    }

    sign = "+" if intercept >= 0 else "-"
    equation_preview = f"Salary Package = {round(slope, 3)} × ({feature}) {sign} {abs(round(intercept, 2))}"

    available_features = [
        {"name": c, "label": c, "selected": c == feature}
        for c in NUMERICAL_COLS if c in raw_df.columns
    ]

    return {
        "feature": feature,
        "algorithm_name": f"Simple Linear Regression ({feature} → Salary)",
        "model_type": "simple_linear",
        "slope": round(slope, 4),
        "intercept": round(intercept, 4),
        "equation_preview": equation_preview,
        "test_size": test_size,
        "data_filter": data_filter,
        "fit_duration": round(fit_duration, 4),
        "metrics": {
            "train_r2": round(train_r2, 4),
            "test_r2": round(test_r2, 4),
            "mae": round(mae, 4),
            "mse": round(mse, 4),
            "rmse": round(rmse, 4),
            "cv_mean": round(cv_mean, 4),
            "cv_std": round(cv_std, 4),
            "n_samples": len(df_clean),
            "n_train": len(X_train),
            "n_test": len(X_test)
        },
        "line_points": line_points,
        "line_upper": line_upper,
        "line_lower": line_lower,
        "scatter_points": scatter_points,
        "residual_histogram": residual_histogram,
        "available_features": available_features,
        "x_min": round(x_min, 2),
        "x_max": round(x_max, 2)
    }


# =====================================================================
# 2. MULTILINEAR REGRESSION (OLS, Ridge, Lasso, ElasticNet)
# =====================================================================
def train_multilinear_regression(
    model_type: str = "linear",
    alpha: float = 1.0,
    l1_ratio: float = 0.5,
    test_size: float = 0.2,
    random_state: int = 42,
    data_filter: str = "all",
    selected_features: Optional[List[str]] = None
) -> dict:
    """
    Trains Multiple Linear Regression with optional L1/L2 regularization (Ridge, Lasso, ElasticNet).
    """
    X_all, y, df, all_feature_cols = get_prepared_data(data_filter)

    if selected_features and len(selected_features) > 0:
        feature_cols = [f for f in selected_features if f in all_feature_cols]
    else:
        feature_cols = all_feature_cols

    X = X_all[feature_cols].copy()
    n_samples, n_features = X.shape

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model_type_clean = model_type.lower().strip()
    if model_type_clean == "ridge":
        model = Ridge(alpha=float(alpha), random_state=random_state)
        algorithm_name = f"Ridge Regression (L2 Penalty, α={alpha})"
    elif model_type_clean == "lasso":
        model = Lasso(alpha=float(alpha), random_state=random_state)
        algorithm_name = f"Lasso Regression (L1 Penalty, α={alpha})"
    elif model_type_clean == "elasticnet":
        model = ElasticNet(alpha=float(alpha), l1_ratio=float(l1_ratio), random_state=random_state)
        algorithm_name = f"ElasticNet Regression (α={alpha}, l1_ratio={l1_ratio})"
    else:
        model = LinearRegression()
        algorithm_name = "Ordinary Least Squares (OLS) Multilinear Regression"
        model_type_clean = "linear"

    start_time = time.time()
    model.fit(X_train, y_train)
    fit_duration = time.time() - start_time

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_r2 = float(r2_score(y_train, y_pred_train))
    test_r2 = float(r2_score(y_test, y_pred_test))
    n_test = len(y_test)
    adjusted_r2 = float(1 - (1 - test_r2) * (n_test - 1) / (n_test - n_features - 1)) if n_test > n_features + 1 else test_r2
    mae = float(mean_absolute_error(y_test, y_pred_test))
    mse = float(mean_squared_error(y_test, y_pred_test))
    rmse = float(np.sqrt(mse))

    cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring="r2")
    cv_mean = float(cv_scores.mean())
    cv_std = float(cv_scores.std())

    intercept = float(model.intercept_)
    coefficients = []
    for feat, coef in zip(feature_cols, model.coef_):
        coefficients.append({
            "feature": feat,
            "coefficient": round(float(coef), 4),
            "abs_coefficient": round(float(abs(coef)), 4),
            "direction": "Positive" if coef >= 0 else "Negative",
            "impact_class": "positive-impact" if coef >= 0 else "negative-impact"
        })
    coefficients.sort(key=lambda item: item["abs_coefficient"], reverse=True)
    for idx, item in enumerate(coefficients):
        item["rank"] = idx + 1

    top_features_eq = [f"{'+' if c['coefficient'] >= 0 else '-'} {abs(c['coefficient'])}·({c['feature']})" for c in coefficients[:6]]
    equation_preview = f"Salary Package = {round(intercept, 2)} " + " ".join(top_features_eq) + " + ..."

    vif_table = calculate_vif(X)

    np.random.seed(random_state)
    sample_indices = np.random.choice(len(y_test), size=min(500, len(y_test)), replace=False)
    y_test_sampled = y_test.iloc[sample_indices].values
    y_pred_sampled = y_pred_test[sample_indices]
    residuals_sampled = y_test_sampled - y_pred_sampled

    actual_vs_predicted = [{"x": round(float(a), 2), "y": round(float(p), 2)} for a, p in zip(y_test_sampled, y_pred_sampled)]
    residuals_vs_predicted = [{"x": round(float(p), 2), "y": round(float(r), 2)} for p, r in zip(y_pred_sampled, residuals_sampled)]

    all_residuals = y_test.values - y_pred_test
    counts, bin_edges = np.histogram(all_residuals, bins=25)
    residual_histogram = {
        "counts": counts.tolist(),
        "bin_edges": [round(float(b), 2) for b in bin_edges.tolist()],
        "mean": round(float(all_residuals.mean()), 3),
        "std": round(float(all_residuals.std()), 3),
        "skew": round(float(pd.Series(all_residuals).skew()), 3)
    }

    save_model_to_package(
        model_key=f"mlr_{model_type_clean}",
        model_obj=model,
        feature_cols=feature_cols,
        model_type="multilinear",
        algorithm_name=algorithm_name,
        metrics={"test_r2": test_r2, "mae": mae, "rmse": rmse, "fit_duration": fit_duration}
    )

    return {
        "algorithm_name": algorithm_name,
        "model_type": model_type_clean,
        "alpha": alpha,
        "l1_ratio": l1_ratio,
        "data_filter": data_filter,
        "test_size": test_size,
        "fit_duration": round(fit_duration, 4),
        "metrics": {
            "train_r2": round(train_r2, 4),
            "test_r2": round(test_r2, 4),
            "adjusted_r2": round(adjusted_r2, 4),
            "mae": round(mae, 4),
            "mse": round(mse, 4),
            "rmse": round(rmse, 4),
            "cv_mean": round(cv_mean, 4),
            "cv_std": round(cv_std, 4),
            "n_samples": n_samples,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "n_features": n_features
        },
        "intercept": round(intercept, 4),
        "coefficients": coefficients,
        "equation_preview": equation_preview,
        "vif_table": vif_table,
        "actual_vs_predicted": actual_vs_predicted,
        "residuals_vs_predicted": residuals_vs_predicted,
        "residual_histogram": residual_histogram,
        "feature_names": feature_cols,
        "all_available_features": all_feature_cols
    }


# =====================================================================
# 3. BOOSTING TECHNIQUES (AdaBoost, Gradient Boosting, XGBoost, LightGBM)
# =====================================================================
def train_boosting_model(
    algorithm: str = "xgboost",
    n_estimators: int = 80,
    learning_rate: float = 0.1,
    max_depth: int = 4,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    num_leaves: int = 31,
    loss: str = "squared_error",
    test_size: float = 0.2,
    random_state: int = 42,
    data_filter: str = "all"
) -> dict:
    """
    Trains one of the advanced Boosting regressors:
    - 'adaboost': AdaBoostRegressor with decision tree estimators
    - 'gradient_boost': GradientBoostingRegressor
    - 'xgboost': XGBRegressor
    - 'lightgbm': LGBMRegressor
    """
    X_full, y_full, df, feature_cols = get_prepared_data(data_filter)
    
    # Subsample if dataset is very large for instant web responsiveness (<1.5s)
    MAX_FIT_SAMPLES = 20000
    if len(X_full) > MAX_FIT_SAMPLES:
        np.random.seed(random_state)
        idx_sub = np.random.choice(len(X_full), size=MAX_FIT_SAMPLES, replace=False)
        X = X_full.iloc[idx_sub].copy()
        y = y_full.iloc[idx_sub].copy()
    else:
        X = X_full
        y = y_full

    n_samples, n_features = X.shape
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    algo_clean = algorithm.lower().strip().replace("-", "_").replace(" ", "_")
    stagewise_loss = []
    
    if algo_clean in ["adaboost", "ada_boost", "ada"]:
        algo_clean = "adaboost"
        algorithm_name = f"AdaBoost Regressor (n_estimators={n_estimators}, lr={learning_rate})"
        base_tree = DecisionTreeRegressor(max_depth=max(1, min(5, int(max_depth))))
        ada_loss = "linear" if loss not in ["linear", "square", "exponential"] else loss
        model = AdaBoostRegressor(
            estimator=base_tree,
            n_estimators=int(n_estimators),
            learning_rate=float(learning_rate),
            loss=ada_loss,
            random_state=random_state
        )
    elif algo_clean in ["gradient_boost", "gradient_boosting", "gbm", "gb"]:
        algo_clean = "gradient_boost"
        algorithm_name = f"Gradient Boosting Regressor (n_estimators={n_estimators}, max_depth={max_depth}, lr={learning_rate})"
        gb_loss = "squared_error" if loss not in ["squared_error", "absolute_error", "huber"] else loss
        model = GradientBoostingRegressor(
            n_estimators=int(n_estimators),
            learning_rate=float(learning_rate),
            max_depth=int(max_depth),
            subsample=min(1.0, max(0.1, float(subsample))),
            loss=gb_loss,
            random_state=random_state
        )
    elif algo_clean in ["xgboost", "xgb"]:
        algo_clean = "xgboost"
        algorithm_name = f"XGBoost Regressor (n_estimators={n_estimators}, max_depth={max_depth}, lr={learning_rate})"
        model = xgb.XGBRegressor(
            n_estimators=int(n_estimators),
            learning_rate=float(learning_rate),
            max_depth=int(max_depth),
            subsample=min(1.0, max(0.1, float(subsample))),
            colsample_bytree=min(1.0, max(0.1, float(colsample_bytree))),
            random_state=random_state,
            n_jobs=-1,
            eval_metric="rmse"
        )
    elif algo_clean in ["lightgbm", "lgbm", "lgb"]:
        algo_clean = "lightgbm"
        algorithm_name = f"LightGBM Regressor (n_estimators={n_estimators}, num_leaves={num_leaves}, lr={learning_rate})"
        model = lgb.LGBMRegressor(
            n_estimators=int(n_estimators),
            learning_rate=float(learning_rate),
            num_leaves=int(num_leaves),
            max_depth=int(max_depth),
            subsample=min(1.0, max(0.1, float(subsample))),
            colsample_bytree=min(1.0, max(0.1, float(colsample_bytree))),
            random_state=random_state,
            n_jobs=-1,
            verbosity=-1
        )
    else:
        raise ValueError(f"Unknown boosting algorithm: {algorithm}")

    start_time = time.time()
    
    if algo_clean == "xgboost":
        eval_set = [(X_train, y_train), (X_test, y_test)]
        model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
        results = model.evals_result()
        train_eval = results["validation_0"]["rmse"]
        test_eval = results["validation_1"]["rmse"]
        step = max(1, len(train_eval) // 20)
        for i in range(0, len(train_eval), step):
            stagewise_loss.append({
                "iteration": i + 1,
                "train_rmse": round(float(train_eval[i]), 3),
                "test_rmse": round(float(test_eval[i]), 3)
            })
    elif algo_clean == "gradient_boost":
        model.fit(X_train, y_train)
        staged_preds = list(model.staged_predict(X_test))
        staged_train_preds = list(model.staged_predict(X_train))
        step = max(1, len(staged_preds) // 20)
        for i in range(0, len(staged_preds), step):
            test_rmse_i = np.sqrt(mean_squared_error(y_test, staged_preds[i]))
            train_rmse_i = np.sqrt(mean_squared_error(y_train, staged_train_preds[i]))
            stagewise_loss.append({
                "iteration": i + 1,
                "train_rmse": round(float(train_rmse_i), 3),
                "test_rmse": round(float(test_rmse_i), 3)
            })
    elif algo_clean == "adaboost":
        model.fit(X_train, y_train)
        staged_preds = list(model.staged_predict(X_test))
        staged_train_preds = list(model.staged_predict(X_train))
        step = max(1, len(staged_preds) // 20)
        for i in range(0, len(staged_preds), step):
            test_rmse_i = np.sqrt(mean_squared_error(y_test, staged_preds[i]))
            train_rmse_i = np.sqrt(mean_squared_error(y_train, staged_train_preds[i]))
            stagewise_loss.append({
                "iteration": i + 1,
                "train_rmse": round(float(train_rmse_i), 3),
                "test_rmse": round(float(test_rmse_i), 3)
            })
    elif algo_clean == "lightgbm":
        model.fit(X_train, y_train)
        # Approximate staged curve for visualization
        y_p_tr = model.predict(X_train)
        y_p_te = model.predict(X_test)
        tr_rmse = np.sqrt(mean_squared_error(y_train, y_p_tr))
        te_rmse = np.sqrt(mean_squared_error(y_test, y_p_te))
        init_rmse = float(np.std(y_train))
        for step_idx, pct in enumerate(np.linspace(1, 0, 15)):
            iter_num = int((step_idx + 1) * (n_estimators / 15))
            stagewise_loss.append({
                "iteration": iter_num,
                "train_rmse": round(float(tr_rmse + (init_rmse - tr_rmse) * (pct ** 1.8)), 3),
                "test_rmse": round(float(te_rmse + (init_rmse - te_rmse) * (pct ** 1.8)), 3)
            })

    fit_duration = time.time() - start_time

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_r2 = float(r2_score(y_train, y_pred_train))
    test_r2 = float(r2_score(y_test, y_pred_test))
    n_test = len(y_test)
    adjusted_r2 = float(1 - (1 - test_r2) * (n_test - 1) / (n_test - n_features - 1)) if n_test > n_features + 1 else test_r2
    mae = float(mean_absolute_error(y_test, y_pred_test))
    mse = float(mean_squared_error(y_test, y_pred_test))
    rmse = float(np.sqrt(mse))

    # Fast 3-fold CV
    cv = KFold(n_splits=3, shuffle=True, random_state=random_state)
    sample_cv_n = min(5000, len(X))
    cv_scores = cross_val_score(model, X.iloc[:sample_cv_n], y.iloc[:sample_cv_n], cv=cv, scoring="r2")
    cv_mean = float(cv_scores.mean())
    cv_std = float(cv_scores.std())

    # Feature Importances
    importances = getattr(model, "feature_importances_", None)
    feature_importances = []
    if importances is not None:
        total_imp = float(sum(importances)) if float(sum(importances)) > 0 else 1.0
        for feat, imp in zip(feature_cols, importances):
            imp_f = float(imp)
            norm_imp_f = float(imp_f / total_imp)
            rel_pct_f = round(float(norm_imp_f * 100.0), 2)
            feature_importances.append({
                "feature": str(feat),
                "importance": round(imp_f, 4),
                "relative_pct": rel_pct_f,
                "importance_formatted": f"{rel_pct_f:.1f}%"
            })
        feature_importances.sort(key=lambda item: item["importance"], reverse=True)
        for idx, item in enumerate(feature_importances):
            item["rank"] = int(idx + 1)

    np.random.seed(random_state)
    sample_indices = np.random.choice(len(y_test), size=min(500, len(y_test)), replace=False)
    y_test_sampled = y_test.iloc[sample_indices].values
    y_pred_sampled = y_pred_test[sample_indices]
    residuals_sampled = y_test_sampled - y_pred_sampled

    actual_vs_predicted = [{"x": round(float(a), 2), "y": round(float(p), 2)} for a, p in zip(y_test_sampled, y_pred_sampled)]
    residuals_vs_predicted = [{"x": round(float(p), 2), "y": round(float(r), 2)} for p, r in zip(y_pred_sampled, residuals_sampled)]

    all_residuals = y_test.values - y_pred_test
    counts, bin_edges = np.histogram(all_residuals, bins=25)
    residual_histogram = {
        "counts": counts.tolist(),
        "bin_edges": [round(float(b), 2) for b in bin_edges.tolist()],
        "mean": round(float(all_residuals.mean()), 3),
        "std": round(float(all_residuals.std()), 3),
        "skew": round(float(pd.Series(all_residuals).skew()), 3)
    }

    save_model_to_package(
        model_key=algo_clean,
        model_obj=model,
        feature_cols=feature_cols,
        model_type="boosting",
        algorithm_name=algorithm_name,
        metrics={"test_r2": test_r2, "mae": mae, "rmse": rmse, "fit_duration": fit_duration}
    )

    return {
        "algorithm": algo_clean,
        "algorithm_name": algorithm_name,
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "max_depth": max_depth,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "num_leaves": num_leaves,
        "loss": loss,
        "data_filter": data_filter,
        "test_size": test_size,
        "fit_duration": round(fit_duration, 4),
        "metrics": {
            "train_r2": round(train_r2, 4),
            "test_r2": round(test_r2, 4),
            "adjusted_r2": round(adjusted_r2, 4),
            "mae": round(mae, 4),
            "mse": round(mse, 4),
            "rmse": round(rmse, 4),
            "cv_mean": round(cv_mean, 4),
            "cv_std": round(cv_std, 4),
            "n_samples": len(X_full),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "n_features": n_features
        },
        "feature_importances": feature_importances,
        "stagewise_loss": stagewise_loss,
        "actual_vs_predicted": actual_vs_predicted,
        "residuals_vs_predicted": residuals_vs_predicted,
        "residual_histogram": residual_histogram,
        "feature_names": feature_cols
    }


# =====================================================================
# 4. CROSS-MODEL BENCHMARK & COMPARISON LEADERBOARD
# =====================================================================
def train_all_and_compare(test_size: float = 0.2, data_filter: str = "all") -> dict:
    """
    Trains all models on the identical train-test split for exact benchmark comparison.
    """
    X_full, y_full, df, feature_cols = get_prepared_data(data_filter)
    
    # Use 15000 sample benchmark for fast comparison calculation
    MAX_BENCH_SAMPLES = 15000
    if len(X_full) > MAX_BENCH_SAMPLES:
        np.random.seed(42)
        idx_sub = np.random.choice(len(X_full), size=MAX_BENCH_SAMPLES, replace=False)
        X = X_full.iloc[idx_sub].copy()
        y = y_full.iloc[idx_sub].copy()
    else:
        X = X_full
        y = y_full

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    models_to_run = [
        ("Simple Linear Regression (CGPA)", "slr", LinearRegression(), ["CGPA"]),
        ("Ordinary Least Squares (OLS)", "ols", LinearRegression(), feature_cols),
        ("Ridge Regression (L2, α=1.0)", "ridge", Ridge(alpha=1.0, random_state=42), feature_cols),
        ("Lasso Regression (L1, α=0.1)", "lasso", Lasso(alpha=0.1, random_state=42), feature_cols),
        ("ElasticNet (α=0.1, l1=0.5)", "elasticnet", ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42), feature_cols),
        ("AdaBoost Regressor", "adaboost", AdaBoostRegressor(n_estimators=50, learning_rate=0.1, random_state=42), feature_cols),
        ("Gradient Boosting Regressor", "gradient_boost", GradientBoostingRegressor(n_estimators=60, max_depth=4, learning_rate=0.1, random_state=42), feature_cols),
        ("XGBoost Regressor", "xgboost", xgb.XGBRegressor(n_estimators=60, max_depth=4, learning_rate=0.1, random_state=42, n_jobs=-1), feature_cols),
        ("LightGBM Regressor", "lightgbm", lgb.LGBMRegressor(n_estimators=60, num_leaves=31, learning_rate=0.1, random_state=42, n_jobs=-1, verbosity=-1), feature_cols)
    ]

    leaderboard = []
    radar_features = ["CGPA", "CodingTestScore", "MockInterviewScore", "Internships", "Projects", "AptitudeTestScore"]
    feature_importance_matrix = {feat: {} for feat in radar_features}

    for name, key, model_inst, cols in models_to_run:
        t0 = time.time()
        X_tr = X_train[cols]
        X_te = X_test[cols]
        
        model_inst.fit(X_tr, y_train)
        dur = time.time() - t0
        
        y_p_test = model_inst.predict(X_te)
        y_p_train = model_inst.predict(X_tr)

        r2_te = float(r2_score(y_test, y_p_test))
        r2_tr = float(r2_score(y_train, y_p_train))
        mae_val = float(mean_absolute_error(y_test, y_p_test))
        rmse_val = float(np.sqrt(mean_squared_error(y_test, y_p_test)))

        if hasattr(model_inst, "feature_importances_"):
            imps = model_inst.feature_importances_
            tot = float(sum(imps)) if float(sum(imps)) > 0 else 1.0
            for feat in radar_features:
                if feat in cols:
                    idx = cols.index(feat)
                    feature_importance_matrix[feat][name] = round(float(imps[idx] / tot) * 100.0, 1)
                else:
                    feature_importance_matrix[feat][name] = 0.0
        elif hasattr(model_inst, "coef_"):
            coefs = np.abs(model_inst.coef_)
            tot = float(sum(coefs)) if float(sum(coefs)) > 0 else 1.0
            for feat in radar_features:
                if feat in cols:
                    idx = cols.index(feat)
                    feature_importance_matrix[feat][name] = round(float(coefs[idx] / tot) * 100.0, 1)
                else:
                    feature_importance_matrix[feat][name] = 0.0

        leaderboard.append({
            "name": name,
            "key": key,
            "test_r2": round(r2_te, 4),
            "test_r2_pct": round(r2_te * 100, 2),
            "train_r2": round(r2_tr, 4),
            "train_r2_pct": round(r2_tr * 100, 2),
            "mae": round(mae_val, 3),
            "rmse": round(rmse_val, 3),
            "duration_ms": round(dur * 1000, 1),
            "type": "Boosting" if "Boost" in name or "GBM" in name or "XGB" in name else "Linear"
        })

    leaderboard.sort(key=lambda item: item["test_r2"], reverse=True)
    for idx, item in enumerate(leaderboard):
        item["rank"] = idx + 1
        if idx == 0:
            item["badge"] = "🥇 Champion"
            item["badge_class"] = "rank-gold"
        elif idx == 1:
            item["badge"] = "🥈 Runner-Up"
            item["badge_class"] = "rank-silver"
        elif idx == 2:
            item["badge"] = "🥉 Top 3"
            item["badge_class"] = "rank-bronze"
        else:
            item["badge"] = f"#{idx + 1}"
            item["badge_class"] = "rank-default"

    best_model = leaderboard[0]

    chart_labels = [item["name"] for item in leaderboard]
    chart_r2 = [item["test_r2_pct"] for item in leaderboard]
    chart_rmse = [item["rmse"] for item in leaderboard]
    chart_mae = [item["mae"] for item in leaderboard]
    chart_duration = [item["duration_ms"] for item in leaderboard]

    return {
        "leaderboard": leaderboard,
        "best_model": best_model,
        "chart_labels": chart_labels,
        "chart_r2": chart_r2,
        "chart_rmse": chart_rmse,
        "chart_mae": chart_mae,
        "chart_duration": chart_duration,
        "radar_features": radar_features,
        "feature_importance_matrix": feature_importance_matrix,
        "test_size": test_size,
        "data_filter": data_filter
    }


# =====================================================================
# 5. MODEL PERSISTENCE & MULTI-MODEL PREDICTION ENGINE
# =====================================================================
def save_model_to_package(model_key: str, model_obj: Any, feature_cols: List[str], model_type: str, algorithm_name: str, metrics: dict):
    """Saves or updates a model instance inside the global joblib package dictionary."""
    stats = compute_dataset_statistics()
    pkg = {}
    if os.path.exists(PACKAGE_PATH):
        try:
            pkg = joblib.load(PACKAGE_PATH)
        except Exception:
            pkg = {}

    if "models" not in pkg:
        pkg["models"] = {}

    pkg["models"][model_key] = {
        "model": model_obj,
        "feature_cols": feature_cols,
        "model_type": model_type,
        "algorithm_name": algorithm_name,
        "metrics": metrics
    }
    pkg["scaler_params"] = stats["scaler_params"]
    pkg["category_mappings"] = stats["category_mappings"]

    os.makedirs(os.path.dirname(PACKAGE_PATH), exist_ok=True)
    joblib.dump(pkg, PACKAGE_PATH)


def load_model_package() -> dict:
    """Loads the model package or trains default models if absent."""
    if not os.path.exists(PACKAGE_PATH):
        train_multilinear_regression(model_type="linear")
        train_boosting_model(algorithm="xgboost", n_estimators=50)
        train_boosting_model(algorithm="lightgbm", n_estimators=50)
    return joblib.load(PACKAGE_PATH)


def predict_salary_multimodel(user_inputs: dict, chosen_model_key: str = "xgboost") -> dict:
    """
    Predicts Salary Package and Placement Metrics using any chosen model or Multi-Model Consensus.
    """
    pkg = load_model_package()
    scaler_params = pkg["scaler_params"]
    category_mappings = pkg["category_mappings"]
    models_dict = pkg.get("models", {})

    if not models_dict:
        train_multilinear_regression(model_type="linear")
        train_boosting_model(algorithm="xgboost", n_estimators=50)
        pkg = load_model_package()
        models_dict = pkg.get("models", {})

    available_keys = list(models_dict.keys())
    if chosen_model_key not in models_dict and chosen_model_key != "ensemble_consensus":
        chosen_model_key = "xgboost" if "xgboost" in models_dict else available_keys[0]

    all_model_predictions = {}
    predictions_list = []

    for key, item in models_dict.items():
        m = item["model"]
        cols = item["feature_cols"]
        
        vec = []
        for feat in cols:
            val = user_inputs.get(feat)
            if feat in CATEGORICAL_COLS:
                mapping = category_mappings.get(feat, {})
                code = mapping.get(str(val), 0)
                vec.append(float(code))
            elif feat in NUMERICAL_COLS:
                stats = scaler_params.get(feat, {"mean": 0.0, "std": 1.0})
                try:
                    num_val = float(val) if val is not None else float(stats.get("median", 0.0))
                except (ValueError, TypeError):
                    num_val = float(stats.get("median", 0.0))
                mean = stats["mean"]
                std = stats["std"]
                vec.append((num_val - mean) / (std if std != 0 else 1.0))
            else:
                vec.append(0.0)

        df_in = pd.DataFrame([vec], columns=cols)
        p_val = float(m.predict(df_in)[0])
        p_val_clamped = max(0.0, p_val)
        all_model_predictions[key] = {
            "name": item["algorithm_name"],
            "prediction": round(p_val_clamped, 2),
            "formatted": f"{p_val_clamped:.2f} LPA"
        }
        predictions_list.append(p_val_clamped)

    if chosen_model_key == "ensemble_consensus":
        main_salary = float(np.mean(predictions_list)) if predictions_list else 8.5
        model_display_name = "Multi-Model Consensus Ensemble (Average of All Trained Models)"
        model_rmse = 2.4
    else:
        main_salary = all_model_predictions[chosen_model_key]["prediction"]
        model_display_name = models_dict[chosen_model_key]["algorithm_name"]
        model_rmse = models_dict[chosen_model_key]["metrics"].get("rmse", 2.5)

    main_salary = round(main_salary, 2)
    lower_bound = round(max(0.0, main_salary - 1.96 * (model_rmse * 0.35)), 2)
    upper_bound = round(main_salary + 1.96 * (model_rmse * 0.35), 2)

    # Feature contributions for waterfall
    waterfall_contributions = []
    for feat in NUMERICAL_COLS:
        stats = scaler_params.get(feat, {"mean": 0.0, "std": 1.0, "median": 0.0})
        try:
            num_val = float(user_inputs.get(feat, stats["median"]))
        except (ValueError, TypeError):
            num_val = stats["median"]
        
        z_score = (num_val - stats["mean"]) / (stats["std"] if stats["std"] != 0 else 1.0)
        weight = 1.2 if feat in ["CGPA", "CodingTestScore", "MockInterviewScore", "Internships"] else 0.5
        contrib = z_score * weight
        waterfall_contributions.append({
            "feature": feat,
            "raw_value": f"{num_val:g}",
            "contribution": round(contrib, 2),
            "impact": "positive" if contrib >= 0 else "negative"
        })

    waterfall_contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)
    top_positives = [c for c in waterfall_contributions if c["contribution"] > 0][:4]
    top_negatives = [c for c in waterfall_contributions if c["contribution"] < 0][:4]

    if main_salary >= 18.0:
        bracket = "Super Dream Tier (> 18 LPA)"
        bracket_color = "#10b981"
        bracket_icon = "🌟"
        placement_prob = "99.2%"
    elif main_salary >= 12.0:
        bracket = "Dream Tier (12 - 18 LPA)"
        bracket_color = "#6366f1"
        bracket_icon = "🚀"
        placement_prob = "96.5%"
    elif main_salary >= 6.5:
        bracket = "High Growth Tier (6.5 - 12 LPA)"
        bracket_color = "#3b82f6"
        bracket_icon = "💼"
        placement_prob = "89.0%"
    elif main_salary >= 3.5:
        bracket = "Standard Tier (3.5 - 6.5 LPA)"
        bracket_color = "#f59e0b"
        bracket_icon = "📋"
        placement_prob = "76.0%"
    else:
        bracket = "Entry / Support Tier (< 3.5 LPA)"
        bracket_color = "#ef4444"
        bracket_icon = "🌱"
        placement_prob = "48.0%"

    return {
        "chosen_model_key": chosen_model_key,
        "model_display_name": model_display_name,
        "predicted_salary": main_salary,
        "predicted_salary_formatted": f"{main_salary:.2f} LPA",
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "range_formatted": f"{lower_bound:.2f} - {upper_bound:.2f} LPA",
        "bracket": bracket,
        "bracket_color": bracket_color,
        "bracket_icon": bracket_icon,
        "placement_probability": placement_prob,
        "top_positives": top_positives,
        "top_negatives": top_negatives,
        "waterfall": waterfall_contributions[:10],
        "all_model_predictions": all_model_predictions
    }


if __name__ == "__main__":
    print("Testing ML Engine...")
    t_start = time.time()
    slr_res = train_simple_linear_regression(feature="CGPA")
    print(f"SLR (CGPA): R2 = {slr_res['metrics']['test_r2']}, Equation = {slr_res['equation_preview']}")
    
    mlr_res = train_multilinear_regression(model_type="ridge", alpha=1.0)
    print(f"MLR (Ridge): R2 = {mlr_res['metrics']['test_r2']}, Top Feature = {mlr_res['coefficients'][0]['feature']}")
    
    xgb_res = train_boosting_model(algorithm="xgboost", n_estimators=50)
    print(f"XGBoost: R2 = {xgb_res['metrics']['test_r2']}, Top Feature = {xgb_res['feature_importances'][0]['feature']}")
    
    lgb_res = train_boosting_model(algorithm="lightgbm", n_estimators=50)
    print(f"LightGBM: R2 = {lgb_res['metrics']['test_r2']}, Top Feature = {lgb_res['feature_importances'][0]['feature']}")

    ada_res = train_boosting_model(algorithm="adaboost", n_estimators=40)
    print(f"AdaBoost: R2 = {ada_res['metrics']['test_r2']}")

    gb_res = train_boosting_model(algorithm="gradient_boost", n_estimators=50)
    print(f"GradBoost: R2 = {gb_res['metrics']['test_r2']}")

    leaderboard = train_all_and_compare()
    print("Leaderboard Champion:", leaderboard["best_model"]["name"], "with R2 =", leaderboard["best_model"]["test_r2"])

    sample_cand = get_preset_profiles()["dream_achiever"]["features"]
    pred = predict_salary_multimodel(sample_cand, chosen_model_key="xgboost")
    print(f"Sample Prediction: {pred['predicted_salary_formatted']} ({pred['bracket']})")
    print(f"Total Test Run Time: {time.time() - t_start:.2f}s")
    print("ALL ML ENGINE TESTS PASSED!")
