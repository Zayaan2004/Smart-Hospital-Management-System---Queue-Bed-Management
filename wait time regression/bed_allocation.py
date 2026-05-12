import pandas as pd
import joblib
import time


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "hospital_data_analysis_with_department.csv"
MODEL_PATH = "criticality_arrival_model.pkl"

ICU_BEDS_AVAILABLE = 2
WARD_BEDS_AVAILABLE = 3


# ============================================================
# LOAD MODEL
# ============================================================

criticality_model = joblib.load(MODEL_PATH)


# ============================================================
# DEPARTMENT MAPPING (RULE-BASED)
# ============================================================

def map_department(condition: str, procedure: str = "") -> str:
    c = condition.lower()
    p = procedure.lower()

    if "heart" in c:
        return "Cardiology"
    if "stroke" in c or "brain" in c:
        return "Neurology"
    if "cancer" in c or "tumor" in c:
        return "Oncology"
    if "fractur" in c or "broken" in c:
        return "Orthopedics"
    if "diabet" in c:
        return "Endocrinology"
    if "kidney" in c or "stone" in c:
        return "Urology"
    if "append" in c:
        return "General Surgery"
    if "respir" in c or "asthma" in c or "lung" in c:
        return "Pulmonology"
    if "pregnan" in c or "childbirth" in c or "c-section" in p:
        return "Obstetrics & Gynaecology"

    return "General Medicine"


# ============================================================
# DETERIORATION RISK (DERIVED, NOT INPUT)
# ============================================================

def get_deterioration_risk(condition: str, procedure: str) -> str:
    text = (condition + " " + procedure).lower()

    if any(k in text for k in ["ventilator", "bypass", "open heart", "c-section", "stroke"]):
        return "high"
    if any(k in text for k in ["stent", "surgery", "ureteroscopy"]):
        return "medium"
    return "low"


# ============================================================
# BED SCORING LOGIC
# ============================================================

CRITICALITY_WEIGHTS = {
    "Critical": 100,
    "High": 70,
    "Medium": 40,
    "Low": 10
}

RISK_WEIGHTS = {
    "high": 30,
    "medium": 15,
    "low": 0
}


def compute_bed_score(age, criticality, risk, waiting_minutes):
    score = 0

    # 1. Criticality dominates
    score += CRITICALITY_WEIGHTS[criticality]

    # 2. Deterioration risk
    score += RISK_WEIGHTS[risk]

    # 3. Waiting-time fairness
    score += min(waiting_minutes / 5, 20)

    # 4. Minimal ethical age modifier
    if age < 12 or age > 75:
        score += 5

    return round(score, 2)


# ============================================================
def select_from_list(prompt: str, options: list[str]) -> str:
    print(f"\n{prompt}")
    for i, opt in enumerate(options):
        print(f"{i + 1}. {opt}")
    while True:
        choice = input("Enter option number: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            return options[int(choice) - 1]
        print("Invalid selection. Try again.")



# USER INPUT
# ============================================================

def get_patient_input():
    print("\n==== NEW PATIENT ARRIVAL ====\n")

    age = int(input("Age: ").strip())
    gender = select_from_list("Gender:", ["Male", "Female", "Other"])
    condition = input("Condition: ").strip()
    procedure = input("Procedure: ").strip()

    waiting_time = int(input("Waiting time (minutes): ").strip())

    bed_type = input("Required bed type (ICU/Ward): ").strip().upper()
    while bed_type not in ["ICU", "WARD"]:
        bed_type = input("Enter ICU or Ward: ").strip().upper()

    return age, gender, condition, procedure, waiting_time, bed_type


# ============================================================
# MAIN FLOW
# ============================================================

def run_allocation():
    global ICU_BEDS_AVAILABLE, WARD_BEDS_AVAILABLE

    age, gender, condition, procedure, waiting_time, bed_type = get_patient_input()

    # Predict criticality
    Xnew = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "Condition": condition,
        "Procedure": procedure
    }])

    criticality = criticality_model.predict(Xnew)[0]

    # Assign department
    department = map_department(condition, procedure)

    # Deterioration risk
    risk = get_deterioration_risk(condition, procedure)

    # Compute score
    score = compute_bed_score(age, criticality, risk, waiting_time)

    # Allocate bed
    allocated = False

    if bed_type == "ICU" and ICU_BEDS_AVAILABLE > 0:
        ICU_BEDS_AVAILABLE -= 1
        allocated = True
    elif bed_type == "WARD" and WARD_BEDS_AVAILABLE > 0:
        WARD_BEDS_AVAILABLE -= 1
        allocated = True

    # Output
    print("\n==== TRIAGE & BED DECISION ====\n")
    print(f"Predicted Criticality : {criticality}")
    print(f"Assigned Department  : {department}")
    print(f"Deterioration Risk   : {risk}")
    print(f"Bed Score            : {score}")
    print(f"Requested Bed Type   : {bed_type}")

    if allocated:
        print("✅ Bed ALLOCATED")
    else:
        print("❌ No bed available — patient queued")

    print("\nRemaining Beds:")
    print(f"ICU  : {ICU_BEDS_AVAILABLE}")
    print(f"Ward : {WARD_BEDS_AVAILABLE}")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    run_allocation()
