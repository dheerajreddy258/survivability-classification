# src/prep_data.py
import pandas as pd
import numpy as np

RAW_PATH = 'data/train.csv'
OUT_PATH = 'data/titanic_clean.csv'

def prep_data():
    df = pd.read_csv(RAW_PATH)
    
    # Handle missing values
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())

    # Feature engineering
    df['IsChild'] = (df['Age'] < 16).astype(int)
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    
    # Encode categoricals
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

    # Drop columns not used for modeling
    drop_cols = ['Cabin', 'Ticket', 'Name', 'PassengerId']
    df.drop(columns=drop_cols, inplace=True, errors='ignore')

    df.to_csv(OUT_PATH, index=False)
    print(f"Cleaned data saved to {OUT_PATH}")

if __name__ == "__main__":
    prep_data()
