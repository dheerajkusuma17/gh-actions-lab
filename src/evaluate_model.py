# Part1: tools, folders and the scoring function

import argparse
import json
import os
import sys

import joblib
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

DATA_DIR = "data"
MODELS_DIR = "models"
METRICS_DIR = "metrics"

def compute_metrics(model, X_test, y_test):
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:,1]
    return{
        "f1_score": round(float(f1_score(y_test, predictions)), 4),
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),

    }


# Part2: the gate and the GitHub score table

def passes_gate(metrics, min_f1):
    return metrics["f1_score"] >= min_f1

def write_github_summary(version, metrics, min_f1, passed):
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return # not running on GitHub, so do nothing
    status = "PASSED - model will be saved" if passed else "FAILED - model blocked"
    lines = [
        f"## Model {version}",
        "",
        "| Metrics | Value |",
        "|---|---|,"
        f"| F1 score | {metrics['f1_score']} |",
        f"| ROC-AUC | {metrics['roc_auc']} |",
        f"| Minimum F1 required | {min_f1} |",
        f"| Quality gate | **{status}** |",
    ]
    with open(summary_path, "a") as f:
        f.write("\n".join(lines) + "\n")

# Part3: the main recipe

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required = True)
    parser.add_argument("--min-f1", type=float, default=0.95)
    args = parser.parse_args()

    model = joblib.load(os.path.join(MODELS_DIR, f"model_{args.version}.joblib"))
    X_test, y_test = joblib.load(os.path.join(DATA_DIR,"test_set.joblib"))

    metrics = compute_metrics(model, X_test, y_test)
    passed = passes_gate(metrics, args.min_f1)

    record = {"version": args.version, **metrics, "min_f1":args.min_f1, "passed": passed}
    os.makedirs(METRICS_DIR, exist_ok=True)
    metrics_path = os.path.join(METRICS_DIR, f"metrics_{args.version}.json")
    with open(metrics_path, "w") as f:
        json.dump(record, f, indent=4)

    print(json.dumps(record, indent=4))
    write_github_summary(args.version, metrics, args.min_f1, passed)

    if not passed:
        print(f"QUALITY GATE FAILED: F1 {metrics['f1_score']} is below {args.min_f1}")
        sys.exit(1)
    print("QUALITY GATE PASSED")

if __name__=="__main__":
    main()

