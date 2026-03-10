import pandas as pd

# ============================================================
# FILE PATHS
# ============================================================

INPUT_CSV = "hospital_data_analysis_arrival_only.csv"
OUTPUT_CSV = "hospital_data_analysis_with_department.csv"


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_CSV)


# ============================================================
# DEPARTMENT MAPPING LOGIC
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
# APPLY MAPPING
# ============================================================

df["Department"] = df.apply(
    lambda row: map_department(
        row.get("Condition", ""),
        row.get("Procedure", "")
    ),
    axis=1
)


# ============================================================
# SAVE RESULT
# ============================================================

df.to_csv(OUTPUT_CSV, index=False)

print("✅ Department column successfully appended")
print(f"✅ New file saved as: {OUTPUT_CSV}")
print("\nDepartment distribution:")
print(df["Department"].value_counts())

