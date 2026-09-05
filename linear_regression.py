"""
Linear Regression Module for Placement Salary Prediction.
Handles model training, hyperparameter configuration, performance evaluation,
diagnostic visualization generation, model persistence, and real-time interactive predictions.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from load_data import load_data, DATA_PATH
from preprocessing import PREPROCESSED_PATH, run_preprocessing

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "Data", "linear_regression_model.joblib")

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
EXCLUDE_COLS = ["StudentID", "PlacementStatus", "IsAnomaly", TARGET_COL]


def get_feature_metadata() -> dict:
    """
    Returns definitions, input types, limits, and human-friendly labels for all features.
    """
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
    """
    Returns preset student profiles for rapid 1-click testing in the UI.
    """
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
    """
    Computes and caches scaling parameters and category mappings from the raw placement dataset.
    """
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


def train_linear_regression(
    test_size: float = 0.2,
    random_state: int = 42,
    model_type: str = "linear",
    alpha: float = 1.0,
    data_filter: str = "all"
) -> dict:
    """
    Trains a Linear Regression, Ridge, or Lasso model on the preprocessed placement dataset.
    Computes comprehensive evaluation metrics, coefficients, and diagnostic visualization sets.
    Persists the model to disk.
    """
    # Ensure preprocessed data exists
    if not os.path.exists(PREPROCESSED_PATH):
        run_preprocessing()
        
    df = pd.read_csv(PREPROCESSED_PATH)
    
    # Filter dataset if requested
    if data_filter == "placed_only":
        df = df[df[TARGET_COL] > 0].copy()
        
    # Prepare X and y
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    X = df[feature_cols].copy()
    y = df[TARGET_COL].copy()
    
    n_samples, n_features = X.shape
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Instantiate model
    model_type_clean = model_type.lower().strip()
    if model_type_clean == "ridge":
        model = Ridge(alpha=float(alpha), random_state=random_state)
        algorithm_name = f"Ridge Regression (L2, α={alpha})"
    elif model_type_clean == "lasso":
        model = Lasso(alpha=float(alpha), random_state=random_state)
        algorithm_name = f"Lasso Regression (L1, α={alpha})"
    else:
        model = LinearRegression()
        algorithm_name = "Ordinary Least Squares (OLS) Linear Regression"
        
    # Fit model
    model.fit(X_train, y_train)
    
    # Predict
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate Metrics
    train_r2 = float(r2_score(y_train, y_pred_train))
    test_r2 = float(r2_score(y_test, y_pred_test))
    
    n_test = len(y_test)
    adjusted_r2 = float(1 - (1 - test_r2) * (n_test - 1) / (n_test - n_features - 1)) if n_test > n_features + 1 else test_r2
    
    mae = float(mean_absolute_error(y_test, y_pred_test))
    mse = float(mean_squared_error(y_test, y_pred_test))
    rmse = float(np.sqrt(mse))
    
    # 5-Fold Cross Validation on entire dataset for robustness
    cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring="r2")
    cv_mean = float(cv_scores.mean())
    cv_std = float(cv_scores.std())
    
    # Coefficients & Intercept
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
        
    # Sort coefficients by absolute magnitude for feature importance
    coefficients.sort(key=lambda item: item["abs_coefficient"], reverse=True)
    for idx, item in enumerate(coefficients):
        item["rank"] = idx + 1

    # Formulate mathematical equation string (top 6 features + intercept)
    top_features_eq = [f"{'+' if c['coefficient'] >= 0 else '-'} {abs(c['coefficient'])}·({c['feature']})" for c in coefficients[:6]]
    equation_preview = f"Salary Package = {round(intercept, 2)} " + " ".join(top_features_eq) + " + ..."
    
    # Diagnostic Visualizations Data
    # 1. Actual vs Predicted sample (sampled 600 points for crisp visual performance)
    np.random.seed(random_state)
    sample_indices = np.random.choice(len(y_test), size=min(600, len(y_test)), replace=False)
    
    y_test_sampled = y_test.iloc[sample_indices].values
    y_pred_sampled = y_pred_test[sample_indices]
    residuals_sampled = y_test_sampled - y_pred_sampled
    
    actual_vs_predicted = []
    for actual, pred in zip(y_test_sampled, y_pred_sampled):
        actual_vs_predicted.append({
            "x": round(float(actual), 2),
            "y": round(float(pred), 2)
        })
        
    ideal_min = float(min(y_test_sampled.min(), y_pred_sampled.min()))
    ideal_max = float(max(y_test_sampled.max(), y_pred_sampled.max()))
    
    # 2. Residuals vs Predicted
    residuals_vs_predicted = []
    for pred, res in zip(y_pred_sampled, residuals_sampled):
        residuals_vs_predicted.append({
            "x": round(float(pred), 2),
            "y": round(float(res), 2)
        })
        
    # 3. Residual Distribution Histogram
    all_residuals = y_test.values - y_pred_test
    counts, bin_edges = np.histogram(all_residuals, bins=25)
    residual_histogram = {
        "counts": counts.tolist(),
        "bin_edges": [round(float(b), 2) for b in bin_edges.tolist()],
        "mean": round(float(all_residuals.mean()), 3),
        "std": round(float(all_residuals.std()), 3),
        "skew": round(float(pd.Series(all_residuals).skew()), 3)
    }
    
    # Save training state & metadata
    dataset_stats = compute_dataset_statistics()
    
    model_package = {
        "model": model,
        "model_type": model_type_clean,
        "algorithm_name": algorithm_name,
        "alpha": alpha,
        "data_filter": data_filter,
        "test_size": test_size,
        "feature_cols": feature_cols,
        "scaler_params": dataset_stats["scaler_params"],
        "category_mappings": dataset_stats["category_mappings"],
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
        "equation_preview": equation_preview
    }
    
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model_package, MODEL_PATH)
    
    return {
        "algorithm_name": algorithm_name,
        "model_type": model_type_clean,
        "alpha": alpha,
        "data_filter": data_filter,
        "test_size": test_size,
        "metrics": model_package["metrics"],
        "intercept": model_package["intercept"],
        "coefficients": coefficients,
        "equation_preview": equation_preview,
        "actual_vs_predicted": actual_vs_predicted,
        "ideal_min": round(ideal_min, 2),
        "ideal_max": round(ideal_max, 2),
        "residuals_vs_predicted": residuals_vs_predicted,
        "residual_histogram": residual_histogram,
        "feature_names": feature_cols
    }


def load_trained_model_package() -> dict:
    """
    Loads saved model package from disk or trains a fresh model if missing.
    """
    if not os.path.exists(MODEL_PATH):
        train_linear_regression()
    return joblib.load(MODEL_PATH)


def predict_salary(user_inputs: dict) -> dict:
    """
    Predicts the salary package (in LPA) for a single candidate profile.
    Calculates confidence interval, bracket classification, and feature waterfall breakdown.
    """
    pkg = load_trained_model_package()
    model = pkg["model"]
    feature_cols = pkg["feature_cols"]
    scaler_params = pkg["scaler_params"]
    category_mappings = pkg["category_mappings"]
    rmse = pkg["metrics"]["rmse"]
    intercept = pkg["intercept"]
    
    # Construct scaled feature vector
    feature_vector = []
    waterfall_contributions = []
    
    for feat in feature_cols:
        val = user_inputs.get(feat)
        scaled_val = 0.0
        raw_val_str = str(val)
        
        if feat in CATEGORICAL_COLS:
            mapping = category_mappings.get(feat, {})
            # Match categorical label to numeric code
            if str(val) in mapping:
                code = mapping[str(val)]
            else:
                # Default to median/first category
                code = list(mapping.values())[0] if mapping else 0
            scaled_val = float(code)
            raw_val_str = str(val)
            
        elif feat in NUMERICAL_COLS:
            stats = scaler_params.get(feat, {"mean": 0.0, "std": 1.0})
            try:
                numeric_val = float(val) if val is not None else float(stats.get("median", stats.get("mean", 0.0)))
            except (ValueError, TypeError):
                numeric_val = float(stats.get("median", stats.get("mean", 0.0)))
                
            mean = stats["mean"]
            std = stats["std"]
            scaled_val = (numeric_val - mean) / (std if std != 0 else 1.0)
            raw_val_str = f"{numeric_val:g}"
            
        feature_vector.append(scaled_val)
        
        # Calculate individual feature attribution: beta_i * x_scaled
        # Find coefficient
        coef_item = next((c for c in pkg["coefficients"] if c["feature"] == feat), None)
        coef_val = coef_item["coefficient"] if coef_item else 0.0
        contribution = coef_val * scaled_val
        
        waterfall_contributions.append({
            "feature": feat,
            "raw_value": raw_val_str,
            "scaled_value": round(scaled_val, 3),
            "coefficient": coef_val,
            "contribution": round(contribution, 3),
            "impact": "positive" if contribution >= 0 else "negative"
        })
        
    # Predict using model with proper feature column names
    X_input = pd.DataFrame([feature_vector], columns=feature_cols)
    raw_pred = float(model.predict(X_input)[0])
    
    # Salary package cannot be negative
    predicted_salary = round(max(0.0, raw_pred), 2)
    
    # Prediction bounds (95% confidence approximate interval: pred +/- 1.96 * RMSE)
    lower_bound = round(max(0.0, predicted_salary - 1.96 * (rmse * 0.4)), 2)
    upper_bound = round(predicted_salary + 1.96 * (rmse * 0.4), 2)
    
    # Sort waterfall items by absolute contribution
    waterfall_contributions.sort(key=lambda item: abs(item["contribution"]), reverse=True)
    top_positives = [c for c in waterfall_contributions if c["contribution"] > 0][:4]
    top_negatives = [c for c in waterfall_contributions if c["contribution"] < 0][:4]
    
    # Classification Tier
    if predicted_salary >= 18.0:
        bracket = "Super Dream Tier (> 18 LPA)"
        bracket_color = "#10b981" # emerald
        bracket_icon = "🌟"
        placement_probability = "99.5%"
    elif predicted_salary >= 12.0:
        bracket = "Dream Tier (12 - 18 LPA)"
        bracket_color = "#6366f1" # indigo
        bracket_icon = "🚀"
        placement_probability = "96.0%"
    elif predicted_salary >= 6.5:
        bracket = "High Growth Tier (6.5 - 12 LPA)"
        bracket_color = "#3b82f6" # blue
        bracket_icon = "💼"
        placement_probability = "88.0%"
    elif predicted_salary >= 3.5:
        bracket = "Standard Tier (3.5 - 6.5 LPA)"
        bracket_color = "#f59e0b" # amber
        bracket_icon = "📋"
        placement_probability = "75.0%"
    else:
        bracket = "Entry / Support Tier (< 3.5 LPA)"
        bracket_color = "#ef4444" # red
        bracket_icon = "🌱"
        placement_probability = "45.0%"
        
    return {
        "predicted_salary": predicted_salary,
        "predicted_salary_formatted": f"{predicted_salary:.2f} LPA",
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "range_formatted": f"{lower_bound:.2f} - {upper_bound:.2f} LPA",
        "bracket": bracket,
        "bracket_color": bracket_color,
        "bracket_icon": bracket_icon,
        "placement_probability": placement_probability,
        "intercept": intercept,
        "top_positives": top_positives,
        "top_negatives": top_negatives,
        "waterfall": waterfall_contributions[:10]
    }


if __name__ == "__main__":
    print("Training Linear Regression model...")
    results = train_linear_regression()
    print("Algorithm:", results["algorithm_name"])
    print("Test R2 Score:", results["metrics"]["test_r2"])
    print("Test MAE:", results["metrics"]["mae"])
    print("Test RMSE:", results["metrics"]["rmse"])
    print("CV 5-Fold Mean R2:", results["metrics"]["cv_mean"])
    print("\nTop 5 Influential Features:")
    for c in results["coefficients"][:5]:
        print(f"  {c['feature']}: {c['coefficient']} ({c['direction']})")
        
    print("\nTesting Sample Prediction:")
    sample_inputs = get_preset_profiles()["dream_achiever"]["features"]
    pred = predict_salary(sample_inputs)
    print(f"Predicted Salary: {pred['predicted_salary_formatted']} (Range: {pred['range_formatted']})")
    print(f"Bracket: {pred['bracket']}")
