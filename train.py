import os
import json
import glob
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from xgboost import XGBClassifier
import pickle

DATA_FOLDER = r"E:\data for life\model AI 1\xgboost-input_moi\data"  # folder chứa 5 file jsonl

# ----------------------------
# 1. HÀM ĐỌC JSONL
# ----------------------------
def load_jsonl(path):
    items = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except:
                try:
                    items.append(json.loads(line.replace("'", '"')))
                except:
                    continue
    return items


# ----------------------------
# 2. LOAD TẤT CẢ FILE JSONL
# ----------------------------
records = []
for file in glob.glob(os.path.join(DATA_FOLDER, "*.jsonl")):
    print("Loading:", file)
    records += load_jsonl(file)

print("Total records loaded:", len(records))

df = pd.DataFrame.from_records(records)
print("Columns:", df.columns.tolist())

# ----------------------------
# 3. XÁC ĐỊNH CỘT LABEL
# ----------------------------
possible = ["label", "y", "target", "class"]
label_col = None
for c in possible:
    if c in df.columns:
        label_col = c
        break

if label_col is None:
    raise Exception("Không tìm thấy cột label trong file!")

print("Using label column:", label_col)

# ----------------------------
# 4. XỬ LÝ LABEL: loại -1
# ----------------------------
df = df[df[label_col].isin([-1, 0, 1])]
print("Label distribution BEFORE:", Counter(df[label_col]))

df = df[df[label_col] != -1]   # loại mẫu không có nhãn

df[label_col] = df[label_col].astype(int)
y = df[label_col]
X = df.drop(columns=[label_col])

print("Label AFTER:", Counter(y))

# ----------------------------
# 5. CHUYỂN list/dict → json string
# ----------------------------
for c in X.columns:
    if X[c].dtype == object:
        sample = X[c].dropna().head(10).tolist()
        if any(isinstance(x, (list, dict)) for x in sample):
            X[c] = X[c].apply(
                lambda x: json.dumps(x) if isinstance(x,(list,dict)) else x
            )

# ----------------------------
# 6. XÓA CỘT MÃ HASH (unique)
# ----------------------------
to_drop = []
for c in X.columns:
    try:
        if X[c].dtype == object and X[c].nunique() == len(X):
            to_drop.append(c)
    except:
        pass

if to_drop:
    print("Dropping ID-like columns:", to_drop)
    X = X.drop(columns=to_drop)

# ----------------------------
# 7. CHUYỂN STRING → NUMERIC
# ----------------------------
for c in X.columns:
    if X[c].dtype == object:
        conv = pd.to_numeric(X[c], errors="ignore")
        if conv.notna().sum() > 0:
            X[c] = conv

numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

print("Numeric cols:", len(numeric_cols))
print("Categorical cols:", len(cat_cols))

# ----------------------------
# 8. PREPROCESSING PIPELINE
# ----------------------------
num_tf = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler())
])

cat_tf = Pipeline([
    ("impute", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", num_tf, numeric_cols),
    ("cat", cat_tf, cat_cols)
])

# ----------------------------
# 9. TRAIN/TEST SPLIT
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ----------------------------
# 10. XGBOOST MODEL
# ----------------------------
model = XGBClassifier(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    eval_metric="logloss",
    n_jobs=4
)

pipeline = Pipeline([
    ("pre", preprocessor),
    ("clf", model)
])

print("Training...")
pipeline.fit(X_train, y_train)

# ----------------------------
# 11. EVALUATION
# ----------------------------
pred = pipeline.predict(X_test)
proba = pipeline.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))
print("ROC AUC:", roc_auc_score(y_test, proba))
print("Confusion matrix:\n", confusion_matrix(y_test, pred))

# ----------------------------
# 12. SAVE MODEL
# ----------------------------
pickle.dump(pipeline, open("malware_xgb.pkl", "wb"))
print("Model saved → malware_xgb.pkl")
