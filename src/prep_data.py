import pandas as pd
from sklearn.preprocessing import LabelEncoder

def prepare_data(input_path, output_path):
    # Load data
    df = pd.read_csv(input_path)
    
    # Handle missing values
    df['Age'].fillna(df['Age'].median(), inplace=True)  # Fill missing Age with median
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)  # Fill missing Embarked
    df.drop('Cabin', axis=1, inplace=True)  # Drop Cabin (too many missing)
    
    # Engineer features
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1  # Total family size
    df['IsChild'] = (df['Age'] < 18).astype(int)  # 1 if child, 0 otherwise
    
    # Encode categoricals
    le = LabelEncoder()
    df['Sex'] = le.fit_transform(df['Sex'])  # 0 for female, 1 for male
    df['Embarked'] = le.fit_transform(df['Embarked'])  # Encode ports
    
    # Drop unnecessary columns
    df.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)
    
    # Save cleaned data
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")

if __name__ == "__main__":
    prepare_data('data/train.csv', 'data/titanic_clean.csv')