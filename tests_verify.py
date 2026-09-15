import json
import time
import sys
import os

# Set UTF-8 encoding for stdout on Windows
sys.stdout.reconfigure(encoding='utf-8')

from app import app
from models_ml import (
    train_simple_linear_regression,
    train_multilinear_regression,
    train_boosting_model,
    train_all_and_compare,
    predict_salary_multimodel,
    get_preset_profiles
)

def test_all():
    print("=== STARTING ML SUITE VERIFICATION ===")
    client = app.test_client()

    # 1. Test Simple Linear Regression
    print("\n[1/7] Testing Simple Linear Regression...")
    slr = train_simple_linear_regression(feature="CGPA", test_size=0.2)
    assert "slope" in slr and "intercept" in slr
    assert len(slr["line_points"]) > 0
    assert slr["metrics"]["test_r2"] > 0.6
    print(f"[OK] SLR (CGPA): R2={slr['metrics']['test_r2']}, Slope={slr['slope']}, Intercept={slr['intercept']}")

    # 2. Test Multilinear Regression (OLS, Ridge, Lasso, ElasticNet)
    print("\n[2/7] Testing Multilinear Regression Variations...")
    for mtype in ["linear", "ridge", "lasso", "elasticnet"]:
        mlr = train_multilinear_regression(model_type=mtype, alpha=1.0, l1_ratio=0.5)
        assert "coefficients" in mlr
        assert "vif_table" in mlr
        assert mlr["metrics"]["test_r2"] > 0.7
        print(f"[OK] MLR ({mtype}): R2={mlr['metrics']['test_r2']}, Top Feature={mlr['coefficients'][0]['feature']}")

    # 3. Test Boosting Techniques (AdaBoost, Gradient Boosting, XGBoost, LightGBM)
    print("\n[3/7] Testing Boosting Techniques...")
    for algo in ["adaboost", "gradient_boost", "xgboost", "lightgbm"]:
        boost = train_boosting_model(algorithm=algo, n_estimators=40, learning_rate=0.1)
        assert "feature_importances" in boost
        assert "stagewise_loss" in boost
        assert boost["metrics"]["test_r2"] > 0.8
        print(f"[OK] Boosting ({algo}): R2={boost['metrics']['test_r2']}, Top Feature={boost['feature_importances'][0]['feature']} ({boost['feature_importances'][0]['importance_formatted']})")

    # 4. Test Model Leaderboard & Comparison
    print("\n[4/7] Testing Cross-Model Comparison & Leaderboard...")
    leaderboard_res = train_all_and_compare(test_size=0.2)
    assert len(leaderboard_res["leaderboard"]) >= 7
    best = leaderboard_res["best_model"]
    print(f"[OK] Leaderboard: {len(leaderboard_res['leaderboard'])} models ranked. Champion={best['name']} (R2={best['test_r2_pct']}%)")

    # 5. Test Multi-Model Prediction Engine
    print("\n[5/7] Testing Multi-Model Prediction...")
    presets = get_preset_profiles()
    cand = presets["dream_achiever"]["features"]
    pred = predict_salary_multimodel(cand, chosen_model_key="xgboost")
    assert pred["predicted_salary"] > 0
    assert len(pred["top_positives"]) > 0
    assert "all_model_predictions" in pred
    print(f"[OK] Prediction: {pred['predicted_salary_formatted']}, Tier={pred['bracket']}, Probability={pred['placement_probability']}")
    print(f"     Consensus across models: {[m['name'].split('(')[0].strip() + ': ' + m['formatted'] for m in pred['all_model_predictions'].values()]}")

    # 6. Test Flask HTML Routes
    print("\n[6/7] Testing Flask HTML Web Routes...")
    routes_to_test = [
        "/",
        "/data-loading",
        "/preprocessing",
        "/eda",
        "/simple-linear-regression",
        "/multilinear-regression",
        "/boosting?algorithm=xgboost",
        "/boosting?algorithm=lightgbm",
        "/boosting?algorithm=adaboost",
        "/boosting?algorithm=gradient_boost",
        "/model-comparison",
        "/prediction"
    ]
    for r in routes_to_test:
        resp = client.get(r)
        assert resp.status_code == 200, f"Route {r} failed with code {resp.status_code}"
        print(f"[OK] Route GET {r}: 200 OK ({len(resp.data)} bytes)")

    # 7. Test REST API Endpoints
    print("\n[7/7] Testing REST APIs...")
    # API Predict
    api_p = client.post("/api/predict", json={"features": cand, "model_choice": "lightgbm"})
    assert api_p.status_code == 200
    p_data = json.loads(api_p.data)
    assert p_data["success"] is True
    print(f"[OK] API POST /api/predict: 200 OK -> {p_data['prediction']['predicted_salary_formatted']}")

    # API Train SLR
    api_slr = client.post("/api/train-slr", json={"feature": "CodingTestScore", "test_size": 0.2})
    assert api_slr.status_code == 200
    print("[OK] API POST /api/train-slr: 200 OK")

    # API Train MLR
    api_mlr = client.post("/api/train-mlr", json={"model_type": "ridge", "alpha": 2.0})
    assert api_mlr.status_code == 200
    print("[OK] API POST /api/train-mlr: 200 OK")

    # API Train Boosting
    api_boost = client.post("/api/train-boosting", json={"algorithm": "xgboost", "n_estimators": 40})
    assert api_boost.status_code == 200
    print("[OK] API POST /api/train-boosting: 200 OK")

    print("\n=======================================================")
    print("ALL TESTS & ENDPOINTS PASSED PERFECTLY WITH 100% SUCCESS!")
    print("=======================================================")

if __name__ == "__main__":
    test_all()
