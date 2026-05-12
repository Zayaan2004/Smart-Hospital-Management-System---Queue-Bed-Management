import pandas as pd
import joblib
import time


# ============================================================
# FILE PATHS
# ============================================================

DATA_PATH = "hospital_data_analysis_with_department.csv"
MODEL_PATH = "criticality_arrival_model.pkl"


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load(MODEL_PATH)


# ============================================================
# DEPARTMENT MAPPING (RULE-BASED)
# ============================================================

def map_department(condition: str, procedure: str = "") -> str:
    c = str(condition).lower()
    p = str(procedure).lower()

    if "heart" in c or "cardiac" in c:
        return "Cardiology"
    if "stroke" in c or "brain" in c or "neuro" in c:
        return "Neurology"
    if "cancer" in c or "tumor" in c or "chemo" in p:
        return "Oncology"
    if "fractur" in c or "broken" in c or "ortho" in p:
        return "Orthopedics"
    if "diabet" in c or "thyroid" in c:
        return "Endocrinology"
    if "kidney" in c or "stone" in c or "urolog" in p:
        return "Urology"
    if "append" in c or "surgery" in p:
        return "General Surgery"
    if "asthma" in c or "respir" in c or "lung" in c:
        return "Pulmonology"
    if "allerg" in c:
        return "Emergency Medicine"
    if "pregnan" in c or "childbirth" in c or "c-section" in p:
        return "Obstetrics & Gynaecology"

    return "General Medicine"


# ============================================================
# UTILITIES
# ============================================================

def get_next_patient_id(df: pd.DataFrame) -> int:
    return int(df["Patient_ID"].max()) + 1


def select_from_list(prompt: str, options: list[str]) -> str:
    print(f"\n{prompt}")
    for i, opt in enumerate(options):
        print(f"{i + 1}. {opt}")
    while True:
        choice = input("Enter option number: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            return options[int(choice) - 1]
        print("Invalid selection. Try again.")


def safe_save_csv(df: pd.DataFrame, path: str):
    try:
        df.to_csv(path, index=False)
    except PermissionError:
        ts = int(time.time())
        fallback = f"hospital_data_analysis_with_department_backup_{ts}.csv"
        df.to_csv(fallback, index=False)
        print(f"\n⚠ CSV locked. Data saved to: {fallback}")


# ============================================================
# PATIENT INPUT (ARRIVAL-TIME)
# ============================================================

def get_patient_input() -> dict:
    print("\n==== ENTER NEW PATIENT DETAILS ====\n")

    age = int(input("Age: ").strip())
    gender = select_from_list("Gender:", ["Male", "Female", "Other"])
    condition = input("Condition (e.g., Asthma, Heart Attack): ").strip()
    procedure = input("Procedure (if known / planned): ").strip()

    return {
        "Age": age,
        "Gender": gender,
        "Condition": condition,
        "Procedure": procedure
    }


# ============================================================
# MAIN FLOW: PREDICT + APPEND
# ============================================================

def predict_and_append():
    df = pd.read_csv(DATA_PATH)

    new_input = get_patient_input()
    new_input["Patient_ID"] = get_next_patient_id(df)

    Xnew = pd.DataFrame([{
        "Age": new_input["Age"],
        "Gender": new_input["Gender"],
        "Condition": new_input["Condition"],
        "Procedure": new_input["Procedure"]
    }])

    # Predict Criticality
    criticality = model.predict(Xnew)[0]

    # Assign Department
    department = map_department(
        new_input["Condition"],
        new_input["Procedure"]
    )

    new_input["Criticality"] = criticality
    new_input["Department"] = department

    df = pd.concat([df, pd.DataFrame([new_input])], ignore_index=True)
    safe_save_csv(df, DATA_PATH)

    print("\n==== NEW PATIENT DATA SAVED ====\n")
    print(new_input)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    predict_and_append()
