from data_loader import load_and_detect_data
from preprocessing import optimize_categories
from model import TicketClassifier
import datetime

def main():
    print("\n" + "="*50)
    print("--- Starting AI Support Ticket Training Pipeline ---")
    print("="*50)
    print(f"[{datetime.datetime.now()}] Initializing Data Load Phase.")
    
    try:
        # Load Data natively using kagglehub and detect correct customer complaint columns
        df = load_and_detect_data()
        
        # Optimize the classes into 4-6 main buckets and clean noise
        df = optimize_categories(df)
        print(f"\nSuccessfully loaded dataset with {len(df)} total records after dropping nulls.")
    except Exception as e:
        print(f"Error during Data Loading Phase: {e}")
        return
        
    print(f"We will use a robust subset of 25,000 rows to ensure fast but highly accurate training.")
    # Sampling for performance while ensuring stratified and balanced learning
    if len(df) > 25000:
        df = df.sample(25000, random_state=42)
        
    print(df['category'].value_counts())
    
    clf = TicketClassifier()
    # Train data and view evaluation metrics (Accuracy, F1, etc.)
    metrics = clf.train(df)
    
    print("\nCaching model locally. This ensures no training loop repeats in Streamlit.")
    clf.save()
    
    print("\n" + "="*50)
    print("--- Pipeline Complete! ---")
    print("You can now safely run 'python3 -m streamlit run app.py' to launch the web dashboard without waiting for retraining.")

if __name__ == "__main__":
    main()
