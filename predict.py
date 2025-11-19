import argparse
import json
import pickle
import pandas as pd
import os
import sys
import subprocess

MODEL_PATH = "malware_xgb.pkl"
OUTPUT_FOLDER = r"E:\data for life\model AI 1\xgboost-input_moi\json extract file"


def extract_to_json(pe_path):
    basename = os.path.basename(pe_path)
    out_json = os.path.join(OUTPUT_FOLDER, os.path.splitext(basename)[0] + ".json")

    cmd = [
        "python",
        "extract_pe_features.py",
        "--file", pe_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("Feature extraction failed:\n", result.stderr)
        sys.exit(1)

    if not os.path.exists(out_json):
        print("ERROR: No output created:", out_json)
        sys.exit(1)

    return out_json


def predict_file(pe_path):

    print("Extracting features...")
    json_file = extract_to_json(pe_path)

    print("Loading features:", json_file)
    with open(json_file, "r", encoding="utf-8") as f:
        feat = json.load(f)

    df = pd.DataFrame([feat])

    # Convert lists/dicts → JSON string (same format as training data)
    for col in df.columns:
        if isinstance(df[col].iloc[0], (list, dict)):
            df[col] = df[col].apply(json.dumps)

    # Load model
    print("Loading model:", MODEL_PATH)
    model = pickle.load(open(MODEL_PATH, "rb"))

    # Predict
    pred = model.predict(df)[0]
    proba = model.predict_proba(df)[0][1]

    label = "MALWARE" if pred == 1 else "BENIGN"

    print("\n========== RESULT ==========")
    print("File:", pe_path)
    print("Prediction:", label)
    print("Malware Probability:", proba)
    print("============================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    args = parser.parse_args()

    predict_file(args.file)
