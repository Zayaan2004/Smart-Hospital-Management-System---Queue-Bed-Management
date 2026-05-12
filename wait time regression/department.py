import pandas as pd

# Load previously processed dataset
df = pd.read_csv("hospital_data_analysis_with_criticality.csv")

# Map Condition -> Department
def map_department(condition: str) -> str:
    condition = str(condition).strip()

    if condition in ["Heart Disease", "Heart Attack"]:
        return "Cardiology"
    elif condition == "Stroke":
        return "Neurology"
    elif condition == "Hypertension":
        return "Cardiology"
    elif condition == "Diabetes":
        return "Endocrinology"
    elif condition in ["Cancer", "Prostate Cancer"]:
        return "Oncology"
    elif condition in ["Fractured Arm", "Fractured Leg"]:
        return "Orthopedics"
    elif condition == "Osteoarthritis":
        return "Orthopedics"
    elif condition == "Kidney Stones":
        return "Urology"
    elif condition == "Appendicitis":
        return "General Surgery"
    elif condition == "Respiratory Infection":
        return "Pulmonology"
    elif condition == "Allergic Reaction":
        return "Emergency Medicine"
    elif condition == "Childbirth":
        return "Obstetrics & Gynaecology"
    else:
        return "General Medicine"

# Apply department mapping
df["Department"] = df["Condition"].apply(map_department)

# Save updated dataset
output_file = "hospital_data_analysis_with_criticality.csv"
df.to_csv(output_file, index=False)

print(f"✅ Department column added!")
print(f"📄 New file saved as: {output_file}\n")

# Display preview
print(df[["Patient_ID", "Condition", "Department", "Criticality", "priority_level"]].head(12))
