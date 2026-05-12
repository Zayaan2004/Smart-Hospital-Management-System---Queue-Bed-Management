import pandas as pd

# Input and output file paths
INPUT_CSV = "hospital_data_analysis_with_criticality.csv"
OUTPUT_CSV = "hospital_data_analysis_arrival_only.csv"

# Load the dataset
df = pd.read_csv(INPUT_CSV)

# Columns to remove
columns_to_remove = [
    "Cost",
    "Length_of_Stay",
    "Readmission",
    "Outcome",
    "Satisfaction"
]

# Drop columns (ignore if some don't exist to avoid crashes)
df_cleaned = df.drop(columns=columns_to_remove, errors="ignore")

# Save to new CSV
df_cleaned.to_csv(OUTPUT_CSV, index=False)

print("✅ Columns removed successfully.")
print(f"✅ New file saved as: {OUTPUT_CSV}")
print("\nRemaining columns:")
print(df_cleaned.columns.tolist())

