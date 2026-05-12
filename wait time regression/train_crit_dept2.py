import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


# ============================================================
# FILE PATHS
# ============================================================

DATA_PATH = "hospital_data_analysis_with_department.csv"
MODEL_PATH = "criticality_arrival_model.pkl"


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

# Remove rows without Criticality
df = df.dropna(subset=["Criticality"])


# ============================================================
# INPUTS & TARGET
# ============================================================

X = df[
    ["Age", "Gender", "Condition", "Procedure"]
]

y = df["Criticality"]


# ============================================================
# PREPROCESSING
# ============================================================

preprocess = ColumnTransformer(
    transformers=[
        ("condition_text", TfidfVectorizer(max_features=1000), "Condition"),
        ("procedure_text", TfidfVectorizer(max_features=500), "Procedure"),
        ("gender", OneHotEncoder(handle_unknown="ignore"), ["Gender"]),
        ("age", "passthrough", ["Age"])
    ]
)


# ============================================================
# MODEL
# ============================================================

model = Pipeline([
    ("preprocess", preprocess),
    ("classifier", LogisticRegression(max_iter=3000))
])


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ============================================================
# TRAIN
# ============================================================

model.fit(X_train, y_train)


# ============================================================
# EVALUATION
# ============================================================

y_pred = model.predict(X_test)

print("\n===== CRITICALITY CLASSIFICATION REPORT =====\n")
print(classification_report(y_test, y_pred))


# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(model, MODEL_PATH)

print("✅ Criticality arrival-time model trained successfully")
print(f"✅ Model saved as: {MODEL_PATH}")
