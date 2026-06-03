from data_loader import load_and_detect_data
from preprocessing import optimize_categories
from model import TicketClassifier
import datetime

def main():
    print("--- Starting AI Support Ticket Classification Pipeline ---")
    print(f"[{datetime.datetime.now()}] Initializing Data Load Phase.")
    
    try:
        df = load_and_detect_data()
        df = optimize_categories(df)
        print(f"Successfully loaded dataset with {len(df)} total records after dropping nulls.")
    except Exception as e:
        print(f"Error during Data Loading Phase: {e}")
        return
        
    # We can use the full dataset here (or sample for extremely fast testing)
    # df = df.sample(min(20000, len(df)), random_state=42)
    print(f"Training on {len(df)} rows.")
    
    clf = TicketClassifier(model_type='logistic') # For baseline, or use 'naive_bayes'
    metrics = clf.train(df)
    print("Optimization complete. Saving artifacts to disk...")
    clf.save()
    
    print("--- Pipeline Complete ---")
    print("You can now run 'streamlit run app.py' to launch the web dashboard.")

if __name__ == "__main__":
    main()
