import pandas as pd
import joblib

DATA_PATH = "hospital_data_analysis_with_criticality.csv"
MODEL_PATH = "crit_dept_model.pkl"

model = joblib.load(MODEL_PATH)


# ---------- Fixed Department Mapping ----------
def map_department_from_condition(condition: str, model_dept: str | None = None) -> str:
    c = condition.strip().lower()

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
    if "allerg" in c:
        return "Emergency Medicine"
    if "pregnan" in c or "childbirth" in c or "delivery" in c:
        return "Obstetrics & Gynaecology"

    known = {
        "Cardiology", "Neurology", "Oncology", "Orthopedics",
        "Endocrinology", "General Surgery", "Pulmonology",
        "Emergency Medicine", "Urology", "Obstetrics & Gynaecology",
        "General Medicine"
    }
    if model_dept in known:
        return model_dept

    return "General Medicine"


def get_next_patient_id(df: pd.DataFrame) -> int:
    return int(df["Patient_ID"].max()) + 1


# ---------- Option Based Inputs ----------
def select_from_list(prompt: str, options: list[str]) -> str:
    print(f"\n{prompt}")
    for i, opt in enumerate(options):
        print(f"{i + 1}. {opt}")
    while True:
        choice = input("Enter option number: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            return options[int(choice) - 1]  # return selected text
        print("❌ Invalid selection. Try again.")


def get_patient_input() -> dict:
    print("\n==== ENTER NEW PATIENT DETAILS ====\n")

    age = int(input("Age: ").strip())
    gender = select_from_list("Gender:", ["Male", "Female", "Other"])
    condition = input("Condition (e.g., Heart Attack): ").strip()
    procedure = input("Procedure: ").strip()
    cost = float(input("Cost (INR): ").strip())
    los = int(input("Length of Stay (days): ").strip())

    readmission = select_from_list("Readmission:", ["Yes", "No"])
    outcome = select_from_list("Outcome:", ["Recovered", "Stable", "Deteriorated"])

    satisfaction = int(select_from_list("Satisfaction:", ["1", "2", "3", "4", "5"]))

    priority = int(select_from_list("Priority Level:", [
        "0 - Low", "1 - Medium", "2 - High", "3 - Critical"
    ])[0])  # first char is number

    return {
        "Age": age,
        "Gender": gender,
        "Condition": condition,
        "Procedure": procedure,
        "Cost": cost,
        "Length_of_Stay": los,
        "Readmission": readmission,
        "Outcome": outcome,
        "Satisfaction": satisfaction,
        "priority_level": priority
    }


def predict_and_append():
    df = pd.read_csv(DATA_PATH)
    new_input = get_patient_input()
    new_input["Patient_ID"] = get_next_patient_id(df)

    Xnew = pd.DataFrame([new_input])
    crit_pred, dept_pred = model.predict(Xnew)[0]

    final_dept = map_department_from_condition(new_input["Condition"], dept_pred)

    new_input["Criticality"] = crit_pred
    new_input["Department"] = final_dept

    df = pd.concat([df, pd.DataFrame([new_input])], ignore_index=True)
    df.to_csv(DATA_PATH, index=False)

    print("\n==== NEW PATIENT DATA SAVED ====")
    print(new_input)


if __name__ == "__main__":
    predict_and_append()
5