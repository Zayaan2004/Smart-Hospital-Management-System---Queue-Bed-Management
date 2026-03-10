import pandas as pd
import joblib
import time


# ============================================================
# CONFIGURATION
# ============================================================

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
# DETERIORATION RISK (DERIVED)
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

    score += CRITICALITY_WEIGHTS[criticality]
    score += RISK_WEIGHTS[risk]
    score += min(waiting_minutes / 5, 20)

    if age < 12 or age > 75:
        score += 5

    return round(score, 2)


# ============================================================
# AUTOMATIC BED ALLOCATION
# ============================================================

def allocate_bed(score, icu_beds, ward_beds):
    ICU_ONLY_THRESHOLD = 120
    FLEX_THRESHOLD = 80

    if score >= ICU_ONLY_THRESHOLD:
        required = "ICU_ONLY"
    elif score >= FLEX_THRESHOLD:
        required = "FLEX"
    else:
        required = "WARD_ONLY"

    if required == "ICU_ONLY":
        if icu_beds > 0:
            return "ICU", "High-risk patient requires ICU", icu_beds - 1, ward_beds
        else:
            return None, "ICU required but no ICU beds available", icu_beds, ward_beds

    if required == "FLEX":
        if icu_beds > 0:
            return "ICU", "ICU preferred and available", icu_beds - 1, ward_beds
        elif ward_beds > 0:
            return "Ward", "Ward acceptable due to ICU unavailability", icu_beds, ward_beds - 1
        else:
            return None, "No beds available", icu_beds, ward_beds

    if required == "WARD_ONLY":
        if ward_beds > 0:
            return "Ward", "ICU not required for this risk level", icu_beds, ward_beds - 1
        else:
            return None, "Ward required but no ward beds available", icu_beds, ward_beds

    return None, "Allocation error", icu_beds, ward_beds


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

    return age, gender, condition, procedure, waiting_time


# ============================================================
# MAIN LOOP
# ============================================================

def run_triage_and_bed_allocation():
    global ICU_BEDS_AVAILABLE, WARD_BEDS_AVAILABLE

    while True:
        if ICU_BEDS_AVAILABLE == 0 and WARD_BEDS_AVAILABLE == 0:
            print("\n🚨 ALL BEDS ARE FULL. NO FURTHER ALLOCATION POSSIBLE.")
            break

        age, gender, condition, procedure, waiting_time = get_patient_input()

        Xnew = pd.DataFrame([{
            "Age": age,
            "Gender": gender,
            "Condition": condition,
            "Procedure": procedure
        }])

        criticality = criticality_model.predict(Xnew)[0]
        department = map_department(condition, procedure)
        risk = get_deterioration_risk(condition, procedure)
        score = compute_bed_score(age, criticality, risk, waiting_time)

        allocation, reason, ICU_BEDS_AVAILABLE, WARD_BEDS_AVAILABLE = allocate_bed(
            score,
            ICU_BEDS_AVAILABLE,
            WARD_BEDS_AVAILABLE
        )

        print("\n==== TRIAGE & BED DECISION ====\n")
        print(f"Predicted Criticality : {criticality}")
        print(f"Assigned Department  : {department}")
        print(f"Deterioration Risk   : {risk}")
        print(f"Bed Score            : {score}")

        if allocation:
            print(f"Allocated Bed        : {allocation}")
            print(f"Reason               : {reason}")
        else:
            print("❌ No bed allocated")
            print(f"Reason               : {reason}")

        print("\nRemaining Beds:")
        print(f"ICU  : {ICU_BEDS_AVAILABLE}")
        print(f"Ward : {WARD_BEDS_AVAILABLE}")

        choice = input("\nIs there another patient? (yes/no): ").strip().lower()
        if choice != "yes":
            print("\nSession ended by operator.")
            break


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    run_triage_and_bed_allocation()
