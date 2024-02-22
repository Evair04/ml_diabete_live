import pandas as pd
import json
import pickle

def load_data():
    # carrrega o dataset de diabete
    df = pd.read_csv('./data/diabetes.csv')

    # retorna o dataset
    return df

def get_all_predictions():
    data = None
    # abre o arquivo json com as predições
    with open('prediction.json', 'r') as f:
        data = json.load(f)

    # retorna o conteúdo do arquivo em formato json
    return data

# salva a predição realizada
def save_prediction(pessoa):
    # busca todas as predições realizadas
    data = get_all_predictions()

    # adiciona a nova predição
    data.append(pessoa)

    # salva o arquivo com a nova predição
    with open('prediction.json', 'w') as f:
        json.dump(data, f)

    # retorna True para indicar que a predição foi salva com sucesso
    return True

# função para realizar a predição
def survival_predict(paciente):
    # cria um dataframe com os dados do paciente
    values = pd.DataFrame([paciente])

    # carrega o modelo treinado
    with open('./models/knn_model.pkl', 'rb') as file:
        model = pickle.load(file)

    # realiza a predição
    results = model.predict(values)

    # inicializa a variável que irá armazenar o resultado da predição
    result = None

    # verifica se a predição foi realizada com sucesso
    if len(results) == 1:
        result = int(results[0])

    # retorna o resultado da predição
    return result