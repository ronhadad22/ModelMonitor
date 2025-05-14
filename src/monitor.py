import joblib
from sklearn.metrics import accuracy_score
import json

def monitor_model(model_path, test_data_path):
    model = joblib.load(model_path)
    with open(test_data_path, 'r') as f:
        data = json.load(f)
    
    X, y = data['X'], data['y']
    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)
    print(f"Model Accuracy: {accuracy}")

if __name__ == "__main__":
    monitor_model("model/model.joblib", "data/test_data.json")