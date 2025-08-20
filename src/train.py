import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn

DATA_PATH = 'data/titanic_clean.csv'

def load_data():
    df = pd.read_csv(DATA_PATH)
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_and_log_model(model, model_name, X_train, X_test, y_train, y_test):
    with mlflow.start_run(run_name=model_name):
        mlflow.sklearn.autolog()  # Automatically log parameters, metrics, and model

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        # Explicit logging (optional)
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        print(f"{model_name} - Accuracy: {acc:.4f}, F1: {f1:.4f}")

def main():
    X_train, X_test, y_train, y_test = load_data()

    models = {
        "LogisticRegression": LogisticRegression(max_iter=200),
        "RandomForest": RandomForestClassifier(),
        "GradientBoosting": GradientBoostingClassifier()
    }

    for name, model in models.items():
        train_and_log_model(model, name, X_train, X_test, y_train, y_test)

if _name_ == '_main_':
    main()
