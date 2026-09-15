import os
from flask import Flask, render_template, request, jsonify
from load_data import get_data_summary, get_duplicate_count
from preprocessing import run_preprocessing
from eda import get_eda_summary_data
from models_ml import (
    train_simple_linear_regression,
    train_multilinear_regression,
    train_boosting_model,
    train_all_and_compare,
    predict_salary_multimodel,
    get_feature_metadata,
    get_preset_profiles
)

app = Flask(__name__, template_folder="Templates", static_folder="Static")


# -------------------------------------------------------------
# 1. HOME & DATA PIPELINE ROUTES
# -------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html", active="none")


@app.route("/data-loading")
def data_loading():
    error = None
    summary = None
    duplicate_count = 0
    try:
        summary = get_data_summary()
        duplicate_count = get_duplicate_count()
    except Exception as e:
        error = f"Data loading error: {e}"

    return render_template(
        "data_loading.html",
        active="data-loading",
        summary=summary,
        Duplicate_count=duplicate_count,
        error=error
    )


@app.route("/preprocessing")
def preprocessing_route():
    error = None
    summary = None
    try:
        summary = run_preprocessing()
    except Exception as e:
        error = f"Preprocessing error: {e}"

    return render_template(
        "preprocessing.html",
        active="preprocessing",
        summary=summary,
        error=error
    )


@app.route("/eda")
def eda_route():
    error = None
    summary = None
    try:
        summary = get_eda_summary_data()
    except Exception as e:
        error = f"EDA error: {e}"

    return render_template(
        "eda.html",
        active="eda",
        summary=summary,
        error=error
    )


# -------------------------------------------------------------
# 2. SIMPLE LINEAR REGRESSION ROUTE (Univariate)
# -------------------------------------------------------------
@app.route("/simple-linear-regression", methods=["GET", "POST"])
def simple_linear_regression_route():
    error = None
    summary = None

    feature = request.values.get("feature", "CGPA")
    test_size = float(request.values.get("test_size", 0.2))
    data_filter = request.values.get("data_filter", "all")

    try:
        summary = train_simple_linear_regression(
            feature=feature,
            test_size=test_size,
            data_filter=data_filter
        )
    except Exception as e:
        error = f"Simple Linear Regression error: {e}"

    return render_template(
        "simple_linear_regression.html",
        active="simple-linear-regression",
        summary=summary,
        error=error,
        params={
            "feature": feature,
            "test_size": test_size,
            "data_filter": data_filter
        }
    )


# -------------------------------------------------------------
# 3. MULTILINEAR REGRESSION ROUTE (OLS, Ridge, Lasso, ElasticNet)
# -------------------------------------------------------------
@app.route("/multilinear-regression", methods=["GET", "POST"])
@app.route("/linear-regression", methods=["GET", "POST"])
@app.route("/model-training", methods=["GET", "POST"])
def multilinear_regression_route():
    error = None
    summary = None

    test_size = float(request.values.get("test_size", 0.2))
    model_type = request.values.get("model_type", "linear")
    alpha = float(request.values.get("alpha", 1.0))
    l1_ratio = float(request.values.get("l1_ratio", 0.5))
    data_filter = request.values.get("data_filter", "all")

    try:
        summary = train_multilinear_regression(
            model_type=model_type,
            alpha=alpha,
            l1_ratio=l1_ratio,
            test_size=test_size,
            data_filter=data_filter
        )
    except Exception as e:
        error = f"Multilinear Regression error: {e}"

    return render_template(
        "multilinear_regression.html",
        active="multilinear-regression",
        summary=summary,
        error=error,
        params={
            "test_size": test_size,
            "model_type": model_type,
            "alpha": alpha,
            "l1_ratio": l1_ratio,
            "data_filter": data_filter
        }
    )


# -------------------------------------------------------------
# 4. BOOSTING TECHNIQUES ROUTE (AdaBoost, Gradient Boost, XGBoost, LightGBM)
# -------------------------------------------------------------
@app.route("/boosting", methods=["GET", "POST"])
def boosting_route():
    error = None
    summary = None

    algorithm = request.values.get("algorithm", "xgboost")
    n_estimators = int(request.values.get("n_estimators", 80))
    learning_rate = float(request.values.get("learning_rate", 0.1))
    max_depth = int(request.values.get("max_depth", 4))
    subsample = float(request.values.get("subsample", 0.8))
    colsample_bytree = float(request.values.get("colsample_bytree", 0.8))
    num_leaves = int(request.values.get("num_leaves", 31))
    loss = request.values.get("loss", "squared_error")
    test_size = float(request.values.get("test_size", 0.2))
    data_filter = request.values.get("data_filter", "all")

    try:
        summary = train_boosting_model(
            algorithm=algorithm,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            num_leaves=num_leaves,
            loss=loss,
            test_size=test_size,
            data_filter=data_filter
        )
    except Exception as e:
        error = f"Boosting Training error: {e}"

    return render_template(
        "boosting.html",
        active="boosting",
        summary=summary,
        error=error,
        params={
            "algorithm": algorithm,
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "num_leaves": num_leaves,
            "loss": loss,
            "test_size": test_size,
            "data_filter": data_filter
        }
    )


# -------------------------------------------------------------
# 5. CROSS-MODEL BENCHMARK & COMPARISON LEADERBOARD
# -------------------------------------------------------------
@app.route("/model-comparison", methods=["GET", "POST"])
def model_comparison_route():
    error = None
    summary = None
    test_size = float(request.values.get("test_size", 0.2))
    data_filter = request.values.get("data_filter", "all")

    try:
        summary = train_all_and_compare(
            test_size=test_size,
            data_filter=data_filter
        )
    except Exception as e:
        error = f"Model Comparison error: {e}"

    return render_template(
        "model_comparison.html",
        active="model-comparison",
        summary=summary,
        error=error,
        params={
            "test_size": test_size,
            "data_filter": data_filter
        }
    )


# -------------------------------------------------------------
# 6. INTERACTIVE MULTI-MODEL PREDICTION ROUTE
# -------------------------------------------------------------
@app.route("/prediction", methods=["GET", "POST"])
def prediction_route():
    error = None
    prediction_result = None
    feature_meta = get_feature_metadata()
    presets = get_preset_profiles()
    
    selected_model = request.values.get("model_choice", "xgboost")
    selected_features = presets["dream_achiever"]["features"].copy()

    if request.method == "POST":
        for group in feature_meta.values():
            for field in group["fields"]:
                fname = field["name"]
                if fname in request.form:
                    selected_features[fname] = request.form.get(fname)
        try:
            prediction_result = predict_salary_multimodel(selected_features, chosen_model_key=selected_model)
        except Exception as e:
            error = f"Prediction error: {e}"
    else:
        try:
            prediction_result = predict_salary_multimodel(selected_features, chosen_model_key=selected_model)
        except Exception as e:
            error = f"Prediction error: {e}"

    return render_template(
        "prediction.html",
        active="prediction",
        feature_meta=feature_meta,
        presets=presets,
        selected_features=selected_features,
        selected_model=selected_model,
        prediction=prediction_result,
        error=error
    )


# -------------------------------------------------------------
# 7. ASYNC REST API ENDPOINTS
# -------------------------------------------------------------
@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        data = request.get_json() or {}
        features = data.get("features", {})
        model_choice = data.get("model_choice", "xgboost")
        result = predict_salary_multimodel(features, chosen_model_key=model_choice)
        return jsonify({"success": True, "prediction": result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/train-slr", methods=["POST"])
def api_train_slr():
    try:
        data = request.get_json() or {}
        feature = data.get("feature", "CGPA")
        test_size = float(data.get("test_size", 0.2))
        data_filter = data.get("data_filter", "all")
        result = train_simple_linear_regression(feature=feature, test_size=test_size, data_filter=data_filter)
        return jsonify({"success": True, "data": result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/train-mlr", methods=["POST"])
def api_train_mlr():
    try:
        data = request.get_json() or {}
        test_size = float(data.get("test_size", 0.2))
        model_type = str(data.get("model_type", "linear"))
        alpha = float(data.get("alpha", 1.0))
        l1_ratio = float(data.get("l1_ratio", 0.5))
        data_filter = str(data.get("data_filter", "all"))
        result = train_multilinear_regression(
            test_size=test_size,
            model_type=model_type,
            alpha=alpha,
            l1_ratio=l1_ratio,
            data_filter=data_filter
        )
        return jsonify({"success": True, "data": result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/train-boosting", methods=["POST"])
def api_train_boosting():
    try:
        data = request.get_json() or {}
        algorithm = data.get("algorithm", "xgboost")
        n_estimators = int(data.get("n_estimators", 80))
        learning_rate = float(data.get("learning_rate", 0.1))
        max_depth = int(data.get("max_depth", 4))
        subsample = float(data.get("subsample", 0.8))
        colsample_bytree = float(data.get("colsample_bytree", 0.8))
        num_leaves = int(data.get("num_leaves", 31))
        loss = data.get("loss", "squared_error")
        test_size = float(data.get("test_size", 0.2))
        data_filter = data.get("data_filter", "all")
        result = train_boosting_model(
            algorithm=algorithm,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            num_leaves=num_leaves,
            loss=loss,
            test_size=test_size,
            data_filter=data_filter
        )
        return jsonify({"success": True, "data": result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True, port=5000)