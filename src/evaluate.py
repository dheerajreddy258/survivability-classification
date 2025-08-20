import joblib
import pandas as pd
from sklearn.metrics import accuracy_score

def evaluate_model(model_path):
    model = joblib.load(model_path)
    df = pd.read_csv('data/titanic_clean.csv')
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    y_pred = model.predict(X)
    print(f"Accuracy: {accuracy_score(y, y_pred)}")

if __name__ == "__main__":
    evaluate_model('models/random_forest.joblib')  # Change to your best model