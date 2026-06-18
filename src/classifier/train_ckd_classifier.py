import numpy as np
import os
import pandas as pd
import xgboost as xgb
import shap
from scipy.stats import qmc
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def generate_virtual_cohort(n_samples=10000):
    print(f"Generating synthetic {n_samples}-patient virtual cohort using Latin Hypercube Sampling...")
    
    # We have 4 features: WSS_avg, WSS_max, Pressure_Drop, GFR
    sampler = qmc.LatinHypercube(d=4, seed=42)
    sample = sampler.random(n=n_samples)
    
    # Scale LHS samples [0, 1] to physiological bounds
    # Bounds: WSS_avg [2, 30], WSS_max [10, 80], P_drop [5, 50], GFR [10, 150]
    l_bounds = [2.0, 10.0, 5.0, 10.0]
    u_bounds = [30.0, 80.0, 50.0, 150.0]
    
    X = qmc.scale(sample, l_bounds, u_bounds)
    df = pd.DataFrame(X, columns=['WSS_Avg', 'WSS_Max', 'Pressure_Drop', 'GFR'])
    
    # Assign logic for CKD Stages based on clinical heuristics
    # High WSS & Low GFR -> High CKD stage
    y = np.zeros(n_samples, dtype=int)
    
    for i in range(n_samples):
        row = df.iloc[i]
        score = (row['WSS_Avg']/15.0) + (row['WSS_Max']/30.0) + (row['Pressure_Drop']/15.0) - (row['GFR']/120.0)
        
        # Heuristic scoring to determine stages
        if score < 1.0:
            y[i] = 0 # Stage 1/Normal
        elif score < 3.0:
            y[i] = 1 # Stage 2/3
        else:
            y[i] = 2 # Stage 4/5
            
    return df, y

def train_classifier():
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    results_dir = os.path.join(PROJECT_ROOT, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    X, y = generate_virtual_cohort(10000)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training XGBoost Classifier on Hemodynamic Features...")
    clf = xgb.XGBClassifier(n_estimators=150, learning_rate=0.1, max_depth=5, random_state=42)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Stage 1/Normal', 'Stage 2/3', 'Stage 4/5']))
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Stage 1', 'Stage 2/3', 'Stage 4/5'], yticklabels=['Stage 1', 'Stage 2/3', 'Stage 4/5'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('CKD Stage Classification (XGBoost)')
    plt.tight_layout()
    cm_path = os.path.join(results_dir, 'ckd_classifier_cm.png')
    plt.savefig(cm_path, dpi=300)
    print(f"Confusion matrix saved to {cm_path}")
    
    # 2. SHAP Explainability
    print("Generating SHAP feature importance...")
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_test)
    
    # Save a shap summary plot
    plt.figure(figsize=(8,6))
    # SHAP summary plot for multi-class returns a list of arrays. We plot for class 2 (Severe CKD) as an example.
    # Note: xgboost multi-class shap_values is a list of arrays or a 3D array depending on xgboost version.
    # To be robust across versions, we'll extract the 2nd class if it's 3D/list, or just plot.
    if isinstance(shap_values, list):
        shap.summary_plot(shap_values[2], X_test, show=False)
    elif len(shap_values.shape) == 3:
        shap.summary_plot(shap_values[:, :, 2], X_test, show=False)
    else:
         shap.summary_plot(shap_values, X_test, show=False)
         
    shap_path = os.path.join(results_dir, 'shap_summary.png')
    plt.savefig(shap_path, bbox_inches='tight', dpi=300)
    print(f"SHAP summary plot saved to {shap_path}")
    
    # Save model for dashboard
    model_path = os.path.join(PROJECT_ROOT, 'assets', 'xgboost_ckd_model.json')
    clf.save_model(model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_classifier()
