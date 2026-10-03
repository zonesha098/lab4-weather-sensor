import pandas as pd
from scipy.stats import skew, kurtosis, zscore

# ---------------------------------------------------
# Create a 15-reading weather sensor dataset
# ---------------------------------------------------

data = {
    "Hour": [
        "01:00", "02:00", "03:00", "04:00", "05:00",
        "06:00", "07:00", "08:00", "09:00", "10:00",
        "11:00", "12:00", "13:00", "14:00", "15:00"
    ],
    "Temperature": [
        20.0, 20.5, 21.0, None, None,
        None, 22.5, 23.0, 23.5, 24.0,
        24.5, 25.0, 25.5, 80.0, 26.0
    ]
}

df = pd.DataFrame(data)

# Save original temperature values for comparison
df["Raw_Temperature"] = df["Temperature"]

print("Original Weather Sensor Data:")
print(df[["Hour", "Temperature"]])

# ---------------------------------------------------
# (a) Descriptive statistics BEFORE cleaning
# ---------------------------------------------------

print("\n--- Descriptive Statistics Before Cleaning ---")
print(df["Temperature"].describe())

# Skewness and kurtosis before cleaning
temp_before = df["Temperature"].dropna()

print("\nSkewness Before Cleaning:", skew(temp_before))
print("Kurtosis Before Cleaning:", kurtosis(temp_before))

# ---------------------------------------------------
# (b) Fill 3 consecutive missing readings
# using linear interpolation
# ---------------------------------------------------

df["Temperature"] = df["Temperature"].interpolate(
    method="linear"
)

print("\n--- After Linear Interpolation ---")
print(df[["Hour", "Temperature"]])

# ---------------------------------------------------
# (c) Detect faulty outlier using Z-score
# threshold = 2
# ---------------------------------------------------

# Calculate Z-scores after interpolation
df["Zscore_Check"] = zscore(df["Temperature"])

threshold = 2

df["Outlier"] = df["Zscore_Check"].abs() > threshold

print("\n--- Z-score Outlier Detection ---")
print("Z-score threshold:", threshold)

print("\nFlagged Outlier Reading(s):")
print(
    df.loc[
        df["Outlier"],
        ["Hour", "Temperature", "Zscore_Check"]
    ]
)

# Find median temperature
temperature_median = df["Temperature"].median()

print("\nTemperature Median:", temperature_median)

# Replace ONLY flagged outlier with median
df.loc[df["Outlier"], "Temperature"] = temperature_median

# ---------------------------------------------------
# (d) Standardize final cleaned Temperature column
# ---------------------------------------------------

df["Standardized_Temperature"] = zscore(df["Temperature"])

# ---------------------------------------------------
# Mean and standard deviation before vs after cleaning
# ---------------------------------------------------

mean_before = df["Raw_Temperature"].mean()
std_before = df["Raw_Temperature"].std()

mean_after = df["Temperature"].mean()
std_after = df["Temperature"].std()

print("\n--- Mean and Standard Deviation Comparison ---")

print("Before Cleaning:")
print("Mean:", mean_before)
print("Standard Deviation:", std_before)

print("\nAfter Cleaning:")
print("Mean:", mean_after)
print("Standard Deviation:", std_after)

# ---------------------------------------------------
# Final comparison table
# ---------------------------------------------------

print("\n--- Final Before/After Comparison Table ---")

final_table = df[
    [
        "Hour",
        "Raw_Temperature",
        "Temperature",
        "Standardized_Temperature"
    ]
]

print(final_table.to_string(index=False))

