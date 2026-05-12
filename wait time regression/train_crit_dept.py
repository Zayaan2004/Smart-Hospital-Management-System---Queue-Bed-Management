import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
import joblib

DATA_PATH = "hospital_data_analysis_with_criticality.csv"

# 1. Load data
df = pd.read_csv(DATA_PATH)

# 2. Separate inputs and outputs
X = df.drop(columns=["Criticality", "Department"])
y = df[["Criticality", "Department"]]

# 3. Define columns
categorical_cols = ["Gender", "Condition", "Procedure",
                    "Readmission", "Outcome"]
numeric_cols = [col for col in X.columns if col not in categorical_cols]

# 4. Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols),
    ]
)

# 5. Model – one RandomForest that outputs both labels
base_rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

multi_rf = MultiOutputClassifier(base_rf)

pipe = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", multi_rf),
    ]
)

# 6. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y["Criticality"]
)

# 7. Train
pipe.fit(X_train, y_train)

# 8. Simple score
score = pipe.score(X_test, y_test)
print(f"Overall multi-output accuracy: {score:.3f}")

# 9. Save trained pipeline
joblib.dump(pipe, "crit_dept_model.pkl")
print("Model saved as crit_dept_model.pkl")
