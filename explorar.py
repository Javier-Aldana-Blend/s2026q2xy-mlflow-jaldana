from sklearn.datasets import load_breast_cancer
import pandas as pd

cancer = load_breast_cancer()
df = pd.DataFrame(cancer['data'], columns=cancer['feature_names'])
df['target'] = cancer['target']

print(f"Shape: {df.shape}")
print(f"Clases: {cancer['target_names']}")  # ['malignant' 'benign']
print(df.describe())
print(df['target'].value_counts())