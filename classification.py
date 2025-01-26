import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
import re
import time
import scipy.sparse

class ImprovedPhishingDetector:
    def __init__(self):
        # Create separate vectorizers for each field
        self.body_vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=3000,
            ngram_range=(1, 2)
        )
        self.subject_vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=1000,
            ngram_range=(1, 2)
        )
        self.from_vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=1000,
            ngram_range=(1, 2)
        )
        self.pipeline = Pipeline([
            ('classifier', RandomForestClassifier(
                n_estimators=100,
                class_weight='balanced',
                random_state=42,
                min_samples_leaf=5,
                max_depth=10,
                min_samples_split=5
            ))
        ])

    def extract_email_features(self, df):
        # Fill NaN values
        df = df.fillna('')
        
        # Create separate feature matrices
        body_features = self.body_vectorizer.fit_transform(df['Body'])
        subject_features = self.subject_vectorizer.fit_transform(df['Subject'])
        from_features = self.from_vectorizer.fit_transform(df['From'])
        
        # Concatenate horizontally
        X = scipy.sparse.hstack([body_features, subject_features, from_features])
        
        return X
    
    def train_and_evaluate(self, X, y):
        # X is already vectorized, use directly
        # Stratified K-Fold
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Cross-validation scores
        cv_scores = cross_val_score(self.pipeline, X, y, cv=skf, scoring='f1')
        
        print("\nScores F1-Cross Validation:")
        print(cv_scores)
        print(f"Score moyen F1: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Final training
        self.pipeline.fit(X, y)
        
        # Final predictions
        y_pred = self.pipeline.predict(X)
        print("\nRapport de Classification Final:\n", 
            classification_report(y, y_pred))
        
        return self.pipeline

    def predict(self, X_new):
        # X_new is already vectorized
        return self.pipeline.predict(X_new)

def main():
    print('Chargement des données...')
    data = pd.read_csv('enron_data_fraud_labeled.csv', low_memory=False)
    
    print('Prétraitement des données...')
    data = data[data['Body'].notna()]
    features = ['Body', 'Subject', 'From']
    data = data[features + ['POI-Present']]
    
    # Split 80/20 avec stratification
    train_data, test_data = train_test_split(
        data,
        test_size=0.2,
        random_state=42,
        stratify=data['POI-Present']
    )
    
    # Préparation des features
    detector = ImprovedPhishingDetector()
    X_train = detector.extract_email_features(train_data)
    y_train = train_data['POI-Present']
    
    X_test = detector.extract_email_features(test_data)
    y_test = test_data['POI-Present']
    
    print('Entraînement du modèle...')
    start_time = time.time()
    detector.train_and_evaluate(X_train, y_train)
    end_time = time.time()
    
    print(f"\nTemps total d'entraînement : {end_time - start_time:.2f} secondes")
    
    # Évaluation sur l'ensemble de test
    print("\nÉvaluation sur l'ensemble de test:")
    predictions = detector.predict(X_test)
    print(classification_report(y_test, predictions))

if __name__ == "__main__":
    main()
