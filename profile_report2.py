import pandas as pd
from ydata_profiling import ProfileReport

# Load dataset
df = pd.read_csv("OnsiteDCDDump.csv", encoding="latin1")
print("Dataset loaded successfully!")
print("Shape:", df.shape)
print(df.head())

# Generate profiling report
profile = ProfileReport(
    df,
    title="Onsite DCD Dataset - EDA Report",
    explorative=True
)

# Save HTML report
profile.to_file("OnsiteDCD_Profile_Report.html")

print("HTML report generated successfully!")