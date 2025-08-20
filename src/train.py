import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
import joblib
import os

def train_model(model_type):
    # Load cleaned data
    df = pd.read_csv('data/titanic_clean.csv')
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Select model
    if model_type == 'logistic':
        model = LogisticRegression()
    elif model_type == 'random_forest':
        model = RandomForestClassifier(n_estimators=100)
    elif model_type == 'gradient_boosting':
        model = GradientBoostingClassifier(n_estimators=100)
    else:
        raise ValueError("Invalid model type")
    
    # Train
    model.fit(X_train, y_train)
    
    # Predict and evaluate
    y_pred = model.predict(X_test)
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    }
    
    # Log to MLflow
    with mlflow.start_run(run_name=model_type):
        mlflow.log_params(model.get_params())
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")
        model_uri = mlflow.get_artifact_uri("model")
        mlflow.register_model(model_uri, "TitanicModel")
    
    # Save model locally
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, f'models/{model_type}.joblib')
    print(f"Model {model_type} trained and logged. Metrics: {metrics}")

if __name__ == "__main__":
    mlflow.set_tracking_uri("http://localhost:5000")  # Start MLflow server first
    for model_type in ['logistic', 'random_forest', 'gradient_boosting']:
        train_model(model_type)