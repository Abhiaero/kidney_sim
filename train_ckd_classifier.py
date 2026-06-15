import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def train_classifier():
    print("Generating synthetic hemodynamic patient data...")
    # Generate 1000 patients
    n_samples = 1000
    np.random.seed(42)
    
    # Features: WSS_avg, WSS_max, Pressure_Drop, GFR
    # Normal (Stage 1-2): high GFR, normal WSS, normal pressure drop
    # CKD (Stage 3-5): low GFR, high/abnormal WSS, high pressure drop
    
    # Stages: 0 (Normal/Stage 1), 1 (Stage 2-3), 2 (Stage 4-5)
    X = np.zeros((n_samples, 4))
    y = np.zeros(n_samples, dtype=int)
    
    for i in range(n_samples):
        stage = np.random.choice([0, 1, 2])
        y[i] = stage
        if stage == 0:
            wss_avg = np.random.normal(15, 2)
            wss_max = np.random.normal(30, 5)
            p_drop = np.random.normal(10, 1)
            gfr = np.random.normal(120, 10)
        elif stage == 1:
            wss_avg = np.random.normal(10, 3)
            wss_max = np.random.normal(45, 8)
            p_drop = np.random.normal(15, 2)
            gfr = np.random.normal(75, 15)
        else:
            wss_avg = np.random.normal(5, 2)
            wss_max = np.random.normal(60, 10)
            p_drop = np.random.normal(25, 4)
            gfr = np.random.normal(20, 10)
            
        X[i] = [wss_avg, wss_max, p_drop, gfr]
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training Random Forest Classifier on Hemodynamic Features...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Stage 1/Normal', 'Stage 2/3', 'Stage 4/5']))
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Stage 1', 'Stage 2/3', 'Stage 4/5'], yticklabels=['Stage 1', 'Stage 2/3', 'Stage 4/5'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('CKD Stage Classification based on CFD Hemodynamics')
    os.makedirs('results', exist_ok=True)
    plt.tight_layout()
    plt.savefig('results/ckd_classifier_cm.png', dpi=300)
    print("Confusion matrix saved to results/ckd_classifier_cm.png")

if __name__ == "__main__":
    train_classifier()
