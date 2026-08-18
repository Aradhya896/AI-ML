import pandas as pd
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split

df = pd.read_csv("OnsiteDCDDump.csv", encoding="latin1")

print("Dataset loaded successfully!")
print("Shape:", df.shape)
print(df.head())

train_data, test_data = train_test_split(
    df,
    test_size=0.20,
    random_state=42
)

print("Training data shape:", train_data.shape)
print("Testing data shape:", test_data.shape)

train_data.to_csv("train_data.csv", index=False)
test_data.to_csv("test_data.csv", index=False)