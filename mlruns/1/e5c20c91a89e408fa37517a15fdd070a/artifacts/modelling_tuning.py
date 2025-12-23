import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# LOAD DATA
def load_data():
    print("[INFO] Loading data...")
    # Pastikan file csv berada di folder yang sama
    train = pd.read_csv('data_train.csv')
    test = pd.read_csv('data_test.csv')
    
    X_train = train.drop(columns=['DEATH_EVENT'])
    y_train = train['DEATH_EVENT']
    X_test = test.drop(columns=['DEATH_EVENT'])
    y_test = test['DEATH_EVENT']
    
    return X_train, y_train, X_test, y_test

# FUNGSI PEMBANTU: MEMBUAT ARTEFAK GAMBAR
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

# UTAMA: TRAINING DENGAN TUNING & MLFLOW
if __name__ == "__main__":
    # Load Data
    X_train, y_train, X_test, y_test = load_data()

    # Set Experiment Name
    mlflow.set_experiment("Eksperimen_Jesika_Heart_Failure")

    with mlflow.start_run():
        print("[INFO] Memulai Hyperparameter Tuning...")

        # --- A. Definisikan Model & Grid Search ---
        rf = RandomForestClassifier(random_state=42)
        
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        }
        
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='accuracy', verbose=1)
        grid_search.fit(X_train, y_train)
        
        # Ambil model terbaik
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        print(f"[INFO] Best Params found: {best_params}")

        # --- B. Evaluasi Model ---
        y_pred = best_model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"[RESULT] Accuracy: {acc:.4f}")

        # --- C. LOGGING KE MLFLOW (BAGIAN PENTING) ---
        
        # 1. Log Parameter & Metrics
        mlflow.log_params(best_params)
        mlflow.log_metrics({"accuracy": acc, "precision": prec, "recall": rec, "f1_score": f1})
        
        # 2. Log Model (INI YANG MEMBUAT .PKL MUNCUL)
        # Pastikan baris ini ada!
        print("[INFO] Menyimpan Model...")
        mlflow.sklearn.log_model(best_model, "model_random_forest")
        
        # 3. Log Artefak Gambar
        print("[INFO] Menyimpan Gambar...")
        plot_confusion_matrix(y_test, y_pred, "confusion_matrix.png")
        plot_feature_importance(best_model, X_train.columns, "feature_importance.png")
        
        mlflow.log_artifact("confusion_matrix.png")
        mlflow.log_artifact("feature_importance.png")

        # 4. Log Script Python ini sendiri
        mlflow.log_artifact(__file__) 
        
        print("[INFO] Logging MLflow selesai. Cek dengan perintah 'mlflow ui'")