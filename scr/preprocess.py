import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.feature_selection import f_classif

def process_data():
    print("Loading raw data...")
    
    data = pd.read_csv("data/raw/star_classification.csv")
    
    # NOTEBOOK LOGIC HERE
    print(data.isnull().sum())


    plt.figure(figsize=(8,5))
    ax = sns.countplot(data=data, x='class', order=data['class'].value_counts().index)

    plt.title('Distribution of Stellar Classes', fontsize=14, fontweight='bold', color='white')
    plt.xlabel('Class (Galaxy, Star, Quasar)', fontsize=12)
    plt.ylabel('Observation Count', fontsize=12)

    plt.show()

    x=data.drop(columns="class")
    y=data["class"]

    f_score,p_value=f_classif(x,y)
    p_value_df=pd.DataFrame({
    'Feature': x.columns,
    'F-Score': f_score,
    'P-Value': p_value
    })
    p_value_df['P-Value'] = p_value_df['P-Value'].apply(lambda x: f"{x:.6f}" if x > 0.000001 else "0.000000 (Highly Significant)")
    p_value_df = p_value_df.sort_values(by='F-Score', ascending=False).reset_index(drop=True)

    print("--- Feature Significance Report ---")
    print(p_value_df)

    #End of notebook logic 
    X = data[['redshift', 'i', 'r', 'z', 'delta']]
    y = data['class']
    
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    joblib.dump(le, "models/label_encoder.joblib")
    
    processed_df = X.copy()
    processed_df['target'] = y_encoded
    
    # Save the clean dataset
    processed_df.to_csv("data/processed/balanced_stars.csv", index=False)
    print("Preprocessed data saved!")

if __name__ == "__main__":
    process_data()