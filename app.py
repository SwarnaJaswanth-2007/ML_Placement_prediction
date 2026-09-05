from flask import Flask, render_template, request, jsonify
from load_data import get_data_summary, get_duplicate_count
from preprocessing import run_preprocessing
from eda import get_eda_summary_data
from linear_regression import (
    train_linear_regression,
    load_trained_model_package,
    predict_salary,
    get_feature_metadata,
    get_preset_profiles
)


app = Flask(__name__, template_folder="Templates", static_folder="Static")

@app.route("/")
def index():
    return render_template(
        "index.html",
        active="none"
    )

@app.route("/data-loading")
def data_loading():
    error = None
    summary = None
    duplicate_count = 0

    try:
        summary = get_data_summary()
        duplicate_count = get_duplicate_count()
    except FileNotFoundError as e:
        error = str(e)
    except Exception as e:
        error = f"Unexpected error: {e}"

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
    except FileNotFoundError as e:
        error = str(e)
    except Exception as e:
        error = f"Unexpected error: {e}"

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
    except FileNotFoundError as e:
        error = str(e)
    except Exception as e:
        error = f"Unexpected error: {e}"

    return render_template(
        "eda.html",
        active="eda",
        summary=summary,
        error=error
    )

@app.route("/model-training", methods=["GET", "POST"])
@app.route("/linear-regression", methods=["GET", "POST"])
def model_training_route():
    error = None
    summary = None

    test_size = float(request.values.get("test_size", 0.2))
    model_type = request.values.get("model_type", "linear")
    alpha = float(request.values.get("alpha", 1.0))
    data_filter = request.values.get("data_filter", "all")

    try:
        summary = train_linear_regression(
            test_size=test_size,
            model_type=model_type,
            alpha=alpha,
            data_filter=data_filter
        )
    except Exception as e:
        error = f"Model training error: {e}"

    return render_template(
        "linear_regression.html",
        active="model-training",
        summary=summary,
        error=error,
        params={
            "test_size": test_size,
            "model_type": model_type,
            "alpha": alpha,
            "data_filter": data_filter
        }
    )

@app.route("/prediction", methods=["GET", "POST"])
def prediction_route():
    error = None
    prediction_result = None
    feature_meta = get_feature_metadata()
    presets = get_preset_profiles()
    
    # Default values from preset
    selected_features = presets["dream_achiever"]["features"].copy()

    if request.method == "POST":
        # Extract features from form submission
        for group in feature_meta.values():
            for field in group["fields"]:
                fname = field["name"]
                if fname in request.form:
                    val = request.form.get(fname)
                    selected_features[fname] = val
        try:
            prediction_result = predict_salary(selected_features)
        except Exception as e:
            error = f"Prediction error: {e}"
    else:
        # Generate initial prediction on load
        try:
            prediction_result = predict_salary(selected_features)
        except Exception as e:
            error = f"Prediction error: {e}"

    return render_template(
        "prediction.html",
        active="prediction",
        feature_meta=feature_meta,
        presets=presets,
        selected_features=selected_features,
        prediction=prediction_result,
        error=error
    )

@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        data = request.get_json() or {}
        features = data.get("features", {})
        result = predict_salary(features)
        return jsonify({"success": True, "prediction": result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

@app.route("/api/train-model", methods=["POST"])
def api_train_model():
    try:
        data = request.get_json() or {}
        test_size = float(data.get("test_size", 0.2))
        model_type = str(data.get("model_type", "linear"))
        alpha = float(data.get("alpha", 1.0))
        data_filter = str(data.get("data_filter", "all"))
        
        result = train_linear_regression(
            test_size=test_size,
            model_type=model_type,
            alpha=alpha,
            data_filter=data_filter
        )
        return jsonify({"success": True, "data": result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)
