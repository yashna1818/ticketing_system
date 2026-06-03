import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from preprocessing import clean_text

class TicketClassifier:
    def __init__(self):
        # Improved accuracy using bigrams and more features
        self.vectorizer = TfidfVectorizer(max_features=8000, ngram_range=(1, 2))
        
        # Adding 'balanced' class weights to heavily penalize misclassified minority classes
        self.model = LogisticRegression(max_iter=1000, class_weight='balanced')
            
    def train(self, df):
        print("Cleaning training text... (Performing lemmatization and stopword removal)")
        df['cleaned_text'] = df['text'].apply(clean_text)
        
        X = df['cleaned_text']
        y = df['category']
        
        # Splitting the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        print(f"Vectorizing text with TF-IDF... Training on {len(X_train)} rows.")
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        
        print("Training classification model (Logistic Regression with Balanced Weights)...")
        self.model.fit(X_train_vec, y_train)
        
        preds = self.model.predict(X_test_vec)
        
        print("\n--- Model Evaluation Metrics ---")
        metrics = {
            'Accuracy': accuracy_score(y_test, preds),
            'Precision': precision_score(y_test, preds, average='macro', zero_division=0),
            'Recall': recall_score(y_test, preds, average='macro', zero_division=0),
            'F1-score': f1_score(y_test, preds, average='macro', zero_division=0)
        }
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
            
        return metrics
        
    def save(self, model_path='model.pkl', vec_path='vectorizer.pkl'):
        # Ensure we write safely
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        with open(vec_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        print("Model and vectorizer saved successfully to disk.")
        
    def load(self, model_path='model.pkl', vec_path='vectorizer.pkl'):
        if os.path.exists(model_path) and os.path.exists(vec_path):
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            with open(vec_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
            return True
        return False
        
    def predict(self, text):
        cleaned = clean_text(text)
        vec = self.vectorizer.transform([cleaned])
        pred = self.model.predict(vec)[0]
        return pred
