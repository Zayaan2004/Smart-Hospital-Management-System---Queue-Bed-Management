import pandas as pd

# 1. Load your hospital dataset
df = pd.read_csv("hospital data analysis.csv")

# 2. Function to assign a criticality score based on condition + length of stay + outcome
def classify_criticality(row):
    condition = str(row["Condition"]).strip()
    los = row["Length_of_Stay"]  # Length of Stay
    outcome = str(row["Outcome"]).strip()

    score = 0

    # --- Base score from condition (you can tweak these) ---
    if condition in ["Heart Attack", "Stroke"]:
        score += 3   # very serious
    elif condition in ["Heart Disease", "Cancer"]:
        score += 2   # serious chronic / life-threatening
    elif condition in ["Appendicitis", "Fractured Leg", "Fractured Arm",
                       "Prostate Cancer", "Kidney Stones"]:
        score += 1   # acute but treatable
    else:
        score += 0   # relatively less acute (e.g. mild infections, osteoarthritis, etc.)

    # --- Length of stay effect (longer stay → higher severity/complexity) ---
    if los >= 15:
        score += 2
    elif los >= 7:
        score += 1

    # --- Outcome effect (if not fully recovered, bump score) ---
    if outcome != "Recovered":
        score += 1

    # --- Map total score to criticality level ---
    if score >= 5:
        return "Critical"
    elif score >= 3:
        return "High"
    elif score >= 1:
        return "Medium"
    else:
        return "Low"

# 3. Apply the function to every row to create a new column
df["Criticality"] = df.apply(classify_criticality, axis=1)

# 4. (OPTIONAL) Map Criticality to numeric priority_level for OPD model
criticality_to_priority = {
    "Low": 0,
    "Medium": 1,
    "High": 2,
    "Critical": 3
}

df["priority_level"] = df["Criticality"].map(criticality_to_priority)

# 5. Save the updated dataset
df.to_csv("hospital_data_analysis_with_criticality.csv", index=False)

print("✅ Criticality column added and file saved as 'hospital_data_analysis_with_criticality.csv'")
print(df[["Patient_ID", "Condition", "Length_of_Stay", "Outcome", "Criticality", "priority_level"]].head(10))
