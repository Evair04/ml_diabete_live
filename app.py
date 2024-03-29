import streamlit as st
import data_handler as dh
import util
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import requests
import json
import numpy as np

# verifica se a senha está correta
if not util.check_password():
    st.stop()  # Do not continue if check_password is not True.

# define a ULR da API
API_URL = 'http://localhost:8000'

# busca os dados do dataset de diabete
response = requests.get(f'{API_URL}/get_diabete_data/')

# inicializa a variável que irá armazenar os dados
dados = None

# verifica se a resposta da API foi bem sucedida
if response.status_code == 200:
    # converte a resposta da API para um DataFrame
    dados_json = json.loads(response.json())
    dados = pd.DataFrame(dados_json)
else:
    # exibe uma mensagem de erro caso a resposta da API não tenha sido bem sucedida
    print("Error: ", response.status_code)

# toggle de controle para exibir os gráficos
data_analyses_on = st.toggle('Mostrar gráficos')

# carrega o modelo de predição
model = pickle.load(open('./models/modelo_knn.pkl', 'rb'))

# Caso o usuário queira ver os gráficos, exibe o dataframe e o histograma das idades
if(data_analyses_on):
    st.dataframe(dados)
    st.header('Histograma das idade')
    fig = plt.figure()
    plt.hist(dados['Age'], bins=30)
    plt.xlabel('Idade')
    plt.ylabel('Quantidade')
    st.pyplot(fig)

# exibe o título da página
st.header('Preditor de Diabetes')

# exibe o formulário para o usuário preencher os dados
# cria 2 colunas para o formulário, um para quantidade de vezes que já engravidou e outro para a glicose
col1, col2 = st.columns([1,2])
with col1:
    vezesEngravidou = st.number_input('Número de vezes que já engravidou', min_value=0, max_value=20, value=2, step=1)

with col2:
    glicose = st.number_input('Concentração de glicose pós 2h no teste oral de tolerância à glicose', min_value=0, max_value=200, value=85, step=1)

# cria 2 colunas para o formulário, um para a pressão arterial diastólica e outro para a espessura da dobra cutânea do tríceps
col1, col2 = st.columns(2)
with col1:
    pressao = st.number_input('Pressão arterial diastólica (mm Hg)', min_value=0, max_value=122, value=100, step=1)

with col2:
    espessuraPele = st.number_input('Espessura da dobra cutânea do tríceps (mm)', min_value=0.00, max_value=99.00, value=30.50, step=0.01)

# cria 2 colunas para o formulário, um para a insulina sérica de 2 horas e outro para o índice de massa corporal
col1, col2= st.columns(2)

with col1:
    insulina = st.number_input('Insulina sérica de 2 horas (mu U/ml)', min_value=0, max_value=900, value=100, step=1)

with col2:
    imc = st.number_input('Índice de massa corporal (peso em kg/(altura em m)^2)', min_value=0, max_value=100, value=50, step=1)

# cria 3 colunas para o formulário, um para a função de pedigree do diabetes, outro para a idade e outro para o botão de submissão
col1, col2, col3 = st.columns(3)

with col1:
    funcaoPedigree = st.number_input('Função de pedigree do diabetes', min_value=0.00, max_value=3.00, value=1.50, step=0.01)

with col2:
    idade = st.number_input('Idade', min_value=0, max_value=120, value=30, step=1)

with col3:
    submit = st.button('Verificar')

# verifica se o usuário pressionou o botão de submissão ou se já existe um diabete no cache
if(submit or 'diabete' in st.session_state):
    # define a variável que irá armazenar os dados do paciente informados no formulário
    paciente = {
        "vezesEngravidou": vezesEngravidou,
        "glicose": glicose,
        "pressao": pressao,
        "espessuraPele": espessuraPele,
        "insulina": insulina,
        "imc": imc,
        "funcaoPedigree": funcaoPedigree,
        "idade": idade
    }

    #  transforma o dict do paciente em um array para ser enviado para a API, devido ao padrão de envio de dados
    array = [paciente['vezesEngravidou'], paciente['glicose'], paciente['pressao'], paciente['espessuraPele'], paciente['insulina'], paciente['imc'], paciente['funcaoPedigree'], paciente['idade']]

    # transforma o array em um JSON
    paciente_json = json.dumps(array)

    # realiza a predição do modelo, utilizando a API
    response = requests.post(f'{API_URL}/predict/', json=paciente_json)

    # inicializa a variável que irá armazenar o resultado da predição
    result = None

    # verifica se a resposta da API foi bem sucedida
    if response.status_code == 200:
        result = response.json()
    else:
        print("Error: ", response.status_code)

    # verifica se o resultado da predição não é nulo
    if result is not None:
        # verifica se o resultado da predição é igual a 1 indicando que o paciente tem diabete
        diabete = result
        if diabete == 1:
            st.subheader('Paciente Com Diabetes')
            if 'diabete' in st.session_state:
                # nem queria que aparecesse isso, pq very sad ter diabetes
                st.snow()
        # caso contrário, o paciente não tem diabete
        else:
            st.subheader('Paciente Sem Diabetes ')
            if 'diabete' in st.session_state:
                st.balloons()

        # adiciona o resultado da predição no cache
        st.session_state['diabete'] = diabete

    # verifica se o usuário já realizou a predição
    if paciente and 'diabete' in st.session_state:
        # cria um botão para o usuário informar se a predição está correta ou não
        st.write('A predição está correta?')

        # cria 3 colunas para os botões de sim, não e para exibir a mensagem de agradecimento
        col1, col2, col3 = st.columns([1,1,5])

        with col1:
            correct_prediction = st.button('Sim')

        with col2:
            wrong_prediction = st.button('Não')

        # verifica se o usuário pressionou um dos botões e exibe a mensagem de agradecimento
        if correct_prediction or wrong_prediction:
            message = "Muito obrigado pelo feedback"
            if wrong_prediction:
                message += ", iremos usar esses dados para melhorar as predições"
            message += "."

            # adiciona no dict do passageiro se a predição está correta ou não
            if correct_prediction:
                paciente['CorrectPrediction'] = True
            elif wrong_prediction:
                paciente['CorrectPrediction'] = False

            # adiciona no dict do passageiro se ele sobreviveu ou não
            paciente['Diabete'] = st.session_state['diabete']

            # escreve a mensagem na tela
            st.write(message)
            print(message)

            # salva a predição no JSON para cálculo das métricas de avaliação do sistema
            paciente_json = json.dumps(paciente)

            response = requests.post(f'{API_URL}/save_prediction/', json=paciente_json)

            if response.status_code == 200:
                print("paciente salvo")
            else:
                print("Error: ", response.status_code)

            #dh.save_prediction(paciente)

        st.write('')
        # adiciona um botão para permitir o usuário realizar uma nova análise
        col1, col2, col3 = st.columns(3)
        with col2:
            new_test = st.button('Iniciar Nova Análise')

            # se o usuário pressionar no botão e já existe um passageiro, remove ele do cache
            if new_test and 'diabete' in st.session_state:
                del st.session_state['diabete']
                st.rerun()
        accuracy_predictions_on = st.toggle('Exibir acurácia')

        # verifica se o usuário quer ver a acurácia
        if accuracy_predictions_on:

            # inicializa a variável que irá armazenar as predições
            predictions = None

            # busca todas as predições realizadas
            response = requests.post(f'{API_URL}/get_all_predictions/', json=dados_json)

            # verifica se a resposta da API foi bem sucedida
            if response.status_code == 200:
                predictions = response.json()
            else:
                print("Error: ", response.status_code)

            # inicializa a variável que irá armazenar o número total de predições
            num_total_predictions = len(predictions)

            # inicializa a variável que irá armazenar o histórico da acurácia
            accuracy_hist = [0]

            # inicializa a variável que irá armazenar o número de predições corretas
            correct_predictions = 0

            # calcula a acurácia
            for index, paciente in enumerate(predictions):
                total = index + 1
                if paciente['CorrectPrediction'] == True:
                    correct_predictions += 1

                temp_accuracy = correct_predictions / total if total else 0

                accuracy_hist.append(round(temp_accuracy, 2))

            # calcula a acurácia final
            accuracy = correct_predictions / num_total_predictions if num_total_predictions else 0

            st.metric(label='Acurácia', value=round(accuracy, 2))
            # TODO: usar o attr delta do st.metric para exibir a diferença na variação da acurácia

            # exibe o histórico da acurácia
            st.subheader("Histórico de acurácia")
            st.line_chart(accuracy_hist)

