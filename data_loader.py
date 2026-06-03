import pandas as pd
import kagglehub
import os

def load_and_detect_data():
    """
    Downloads the dataset from Kaggle using kagglehub, loads it,
    and automatically detects text and label/category columns appropriately.
    """
    print("Downloading dataset using kagglehub...")
    path = kagglehub.dataset_download("mirzayasirabdullah07/customer-support-tickets-dataset-200k-records")
    
    csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
    if not csv_files:
        raise FileNotFoundError("No CSV file found in the downloaded dataset.")
        
    csv_file = os.path.join(path, csv_files[0])
    print(f"Loading data from {csv_file}")
    df = pd.read_csv(csv_file)
    
    print("\n--- Dataset Schema Insights ---")
    print("All Original Columns in Dataset:")
    print(df.columns.tolist())
    
    # Automatically detect relevant columns, strongly prioritizing customer-authored fields over agent replies
    text_candidates = ['ticket_description', 'customer_message', 'customer_remark', 'description', 
                       'ticket_subject', 'customer_complaint', 'text', 'ticket', 'issue']
    label_candidates = ['ticket_type', 'category', 'issue_type', 'label', 'topic', 'department', 'ticket_category']
    
    df_cols = [c.lower() for c in df.columns]
    
    text_col = None
    for cand in text_candidates:
        if cand in df_cols:
            text_col = df.columns[df_cols.index(cand)]
            break
            
    label_col = None
    for cand in label_candidates:
        if cand in df_cols:
            label_col = df.columns[df_cols.index(cand)]
            break
            
    if not text_col:
        # Fallback to the object column with longest average string length, explicitly avoiding "resolution notes"
        max_len = 0
        for col in df.columns:
            if df[col].dtype == 'object' and 'resolution' not in col.lower() and 'reply' not in col.lower():
                avg_len = df[col].dropna().sample(min(100, len(df[col]))).astype(str).str.len().mean()
                if avg_len > max_len:
                    max_len = avg_len
                    text_col = col
                    
    if not label_col:
        # Fallback to the first object column with few unique values that isn't the text column
        for col in df.columns:
            if df[col].dtype == 'object' and col != text_col:
                if df[col].nunique() < 30:
                    label_col = col
                    break
                    
    print(f"-> Selected Text Column (Customer Complaint): '{text_col}'")
    print(f"-> Selected Label Column (Category): '{label_col}'\n")
    
    # Keep only target columns and drop NA to clean data
    df = df[[text_col, label_col]].dropna()
    df.rename(columns={text_col: 'text', label_col: 'category'}, inplace=True)
    return df
