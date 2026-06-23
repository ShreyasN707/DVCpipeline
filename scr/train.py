import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold,cross_val_score
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

def train_model():

    print("Loading clean data...")
    df = pd.read_csv("data/processed/balanced_stars.csv")
    X = df.drop(columns=['target'])
    y = df['target']
    
    
    Pipe=ImbPipeline(steps=[
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)) 
    ])

    cv_stratergy=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)

    print("Training pipeline...")
    scores=cross_val_score(Pipe,X,y,cv=cv_stratergy,scoring='balanced_accuracy',n_jobs=-1)

    print("\n--- Cross-Validation Results ---")
    print(f"F1-Scores for each fold: {np.round(scores, 4)}")
    print(f"Mean F1-ba: {scores.mean():.4f}")
    print(f"Standard Deviation: {scores.std():.4f}")

    if scores.std() < 0.02:
        print("[+] Status: Highly Stable. The model performs consistently across all splits.")
    else:
        print("[-] Status: Unstable. The model's performance fluctuates depending on the split.")
    
    
    metrics_dict = {"f1_macro": scores.mean()}
    
    Pipe.fit(X, y)

    with open("metrics.json", "w") as outfile:
        json.dump(metrics_dict, outfile)
    print(f"Metrics saved: {metrics_dict}")
    
    Path("models").mkdir(parents=True, exist_ok=True)
    joblib.dump(Pipe, "models/stellar_pipeline.joblib")
    print("Pipeline saved successfully!")

if __name__ == "__main__":
    train_model()