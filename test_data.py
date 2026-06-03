import kagglehub
import pandas as pd
import os

print("Downloading dataset...")
path = kagglehub.dataset_download("mirzayasirabdullah07/customer-support-tickets-dataset-200k-records")
print("Path to dataset files:", path)

files = os.listdir(path)
print("Files:", files)

csv_file = [f for f in files if f.endswith('.csv')][0]
df = pd.read_csv(os.path.join(path, csv_file))
print("Columns:", df.columns.tolist())
print(df.head(2))
