# src/evaluate.py
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import mlflow.pyfunc

DATA_PATH = 'data/titanic_clean.csv'
MODEL_NAME = "YourRegisteredModelName"  # Change this to your MLflow registered model name or local model path

def load_data():
    df = pd.read_csv(DATA_PATH)
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    # Same test split as train.py to ensure consistency
    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_test, y_test

def load_model():
    # Load model from MLflow Model Registry by name and stage 'Production'
    model_uri = f"models:/{MODEL_NAME}/Production"
    model = mlflow.pyfunc.load_model(model_uri)
    return model

def evaluate():
    X_test, y_test = load_data()
    model = load_model()

    preds = model.predict(X_test)
    if hasattr(preds, "astype"):
        preds = preds.astype(int)  # Ensure labels are int (if probability thresholding applied outside)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    print(f"Model Evaluation on Test Set:")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print("\nClassification Report:\n")
    print(classification_report(y_test, preds))

if __name__ == "__main__":
    evaluate()
