import pandas as pd
from ydata_profiling import ProfileReport

# Read Excel file
df = pd.read_excel("ITSM_Data.xlsx")

print(df.shape)
print(df.head())

profile = ProfileReport(
    df,
    title="ITSM Dataset Profiling Report",
    explorative=True
)

profile.to_file("ITSM_Profile_Report.html")

print("HTML report generated successfully!")