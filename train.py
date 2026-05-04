import pandas as pd
import mlflow
import mlflow.sklearn
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# Chargement des données
df = pd.read_csv("hr_data.csv")

# Features et target
X = df.drop("churn", axis=1)
y = df["churn"]

# Séparation train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Sauvegarde du schéma de référence pour le monitoring
X_train.to_csv("reference_data.csv", index=False)

# MLflow tracking
mlflow.set_experiment("churn_prediction")

with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Log paramètres et métriques
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)
    mlflow.sklearn.log_model(model, "churn_model")
    
    # Log du schéma de référence
    mlflow.log_artifact("reference_data.csv")
    
    print(f"Accuracy : {acc:.4f}")
    print(f"F1 Score : {f1:.4f}")

# Sauvegarde du modèle
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Modèle sauvegardé : model.pkl")