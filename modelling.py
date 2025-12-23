import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# LOAD DATA
def load_data():
    print("[INFO] Loading data...")
    # Pastikan file csv berada di folder yang sama
    train = pd.read_csv('Membangun_model/data_train.csv')
    test = pd.read_csv('Membangun_model/data_test.csv')
    
    X_train = train.drop(columns=['DEATH_EVENT'])
    y_train = train['DEATH_EVENT']
    X_test = test.drop(columns=['DEATH_EVENT'])
    y_test = test['DEATH_EVENT']
    
    return X_train, y_train, X_test, y_test

def plot_confusion_matrix(y_true, y_pred, filename="confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Hidup', 'Meninggal'], yticklabels=['Hidup', 'Meninggal'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_feature_importance(model, feature_names, filename="feature_importance.png"):
    importances = model.feature_importances_
    indices = importances.argsort()[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances")
    plt.bar(range(len(importances)), importances[indices], align="center")
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


if __name__ == "__main__":
    # Load Data
    X_train, y_train, X_test, y_test = load_data()

    # Set Experiment Name
    mlflow.set_experiment("Eksperimen_Jesika_Heart_Failure_Basic")

    with mlflow.start_run():
        print("[INFO] Memulai Training Model Basic...")
        params = {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': 42
        }
        
        # Training Model
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        print(f"[INFO] Model trained with params: {params}")

        # valuasi Model
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"[RESULT] Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f}")

        # --- D. Logging ke MLflow ---
        mlflow.log_params(params)
        mlflow.log_metrics({
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1
        })
        mlflow.sklearn.log_model(model, "model_random_forest_basic")
        
        plot_confusion_matrix(y_test, y_pred, "confusion_matrix.png")
        plot_feature_importance(model, X_train.columns, "feature_importance.png")
        
        mlflow.log_artifact("confusion_matrix.png")
        mlflow.log_artifact("feature_importance.png")
        
        print("[INFO] Logging MLflow selesai.")