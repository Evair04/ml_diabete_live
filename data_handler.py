import pandas as pd
import json
import pickle

def load_data():
    # Load data from csv file
    df = pd.read_csv('./data/diabetes.csv')

    return df

def get_all_predictions():
    data = None
    with open('prediction.json', 'r') as f:
        data = json.load(f)

    return data

def save_prediction(pessoa):
    data = get_all_predictions()

    # data = json.loads(data)
    data.append(pessoa)

    with open('prediction.json', 'w') as f:
        json.dump(data, f)

    return True

def survival_predict(paciente):

    values = pd.DataFrame([paciente])

    with open('./models/knn_model.pkl', 'rb') as file:
        model = pickle.load(file)

    results = model.predict(values)

    result = None

    if len(results) == 1:
        result = int(results[0])

    return result