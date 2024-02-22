from fastapi import Body, FastAPI
from typing import Any
import data_handler
import json

# inicializa a API
api = FastAPI()

# comando para rodar a API
# uvicorn main:api --reload

# rota para o endpoint /hello_world/
@api.get("/hello_world/")
def hello_world():
    return {"message": "Hello World"}

# rota para o endpoint retornar os dados do dataset de diabete
@api.get("/get_diabete_data/")
def get_titanic():
    dados = data_handler.load_data()

    dados_json = dados.to_json(orient='records')

    return dados_json

# rota para retornar todas as predições realizadas
@api.post("/get_all_predictions/")
def get_predictions():
    all_predictions = data_handler.get_all_predictions()

    return all_predictions

# rota para salvar a predição realizada
@api.post("/save_prediction/")
def save_prediction(pessoa_json: Any = Body(None)):
    pessoa = json.loads(pessoa_json)

    result = data_handler.save_prediction(pessoa)

    return result

# rota para realizar a predição
@api.post("/predict/")
def predict(pessoa_json: Any = Body(None)):

    pessoa = json.loads(pessoa_json)

    result = data_handler.survival_predict(pessoa)

    return result