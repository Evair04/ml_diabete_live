import streamlit as st
import data_handler as dh
import util
import matplotlib.pyplot as plt
import pandas as pd
import pickle

if not util.check_password():
    st.stop()  # Do not continue if check_password is not True.

dados = dh.load_data()

data_analyses_on = st.toggle('Mostrar gráficos')

model = pickle.load(open('./models/modelo_knn.pkl', 'rb'))


if(data_analyses_on):
    st.dataframe(dados)
    st.header('Histograma das idade')
    fig = plt.figure()
    plt.hist(dados['Age'], bins=30)
    plt.xlabel('Idade')
    plt.ylabel('Quantidade')
    st.pyplot(fig)

st.header('Preditor de Diabetes')

col1, col2 = st.columns([1,2])
with col1:
    vezesEngravidou = st.number_input('Número de vezes que já engravidou', min_value=0, max_value=20, value=2, step=1)

with col2:
    glicose = st.number_input('Concentração de glicose pós 2h no teste oral de tolerância à glicose', min_value=0, max_value=200, value=85, step=1)

col1, col2 = st.columns(2)
with col1:
    pressao = st.number_input('Pressão arterial diastólica (mm Hg)', min_value=0, max_value=122, value=100, step=1)

with col2:
    espessuraPele = st.number_input('Espessura da dobra cutânea do tríceps (mm)', min_value=0.00, max_value=99.00, value=30.50, step=0.01)

col1, col2= st.columns(2)

with col1:
    insulina = st.number_input('Insulina sérica de 2 horas (mu U/ml)', min_value=0, max_value=900, value=100, step=1)

with col2:
    imc = st.number_input('Índice de massa corporal (peso em kg/(altura em m)^2)', min_value=0, max_value=100, value=50, step=1)

col1, col2, col3 = st.columns(3)

with col1:
    funcaoPedigree = st.number_input('Função de pedigree do diabetes', min_value=0.00, max_value=3.00, value=1.50, step=0.01)

with col2:
    idade = st.number_input('Idade', min_value=0, max_value=120, value=30, step=1)

with col3:
    submit = st.button('Verificar')

if(submit or 'survived' in st.session_state):
    p_class_map = {
        '1st': 1,
        '2nd': 2,
        '3rd': 3
    }

    sex_map = {
        'Male': 0,
        'Female': 1
    }
    embarked_map = {
        'Cherbourg': 0,
        'Queenstown': 1,
        'Southampton': 2
    }
    passageiro = {
        'Pclass': p_class_map[p_class],
        'Sex': sex_map[sex],
        'Age': age,
        'SibSp': sib_sp,
        'Parch': par_ch,
        'Fare': fare,
        'Embarked': embarked_map[embarked]
    }

    values = pd.DataFrame([passageiro])
    # st.dataframe(values)

    results = model.predict(values)

    if len(results) == 1:
        survived = int(results[0])
        if survived == 1:
            st.subheader('Passageiro Sobreviveu')
            if 'survived' in st.session_state:
                st.balloons()
        else:
            st.subheader('Passageiro Não sobreviveu ')
            if 'survived' in st.session_state:
                st.snow()

    st.session_state['survived'] = survived

    if passageiro and 'survived' in st.session_state:
        st.write('A predição está correta?')

        col1, col2, col3 = st.columns([1,1,5])

        with col1:
            correct_prediction = st.button('Sim')

        with col2:
            wrong_prediction = st.button('Não')

        if correct_prediction or wrong_prediction:
            message = "Muito obrigado pelo feedback"
            if wrong_prediction:
                message += ", iremos usar esses dados para melhorar as predições"
            message += "."

            # adiciona no dict do passageiro se a predição está correta ou não
            if correct_prediction:
                passageiro['CorrectPrediction'] = True
            elif wrong_prediction:
                passageiro['CorrectPrediction'] = False

            # adiciona no dict do passageiro se ele sobreviveu ou não
            passageiro['Survived'] = st.session_state['survived']

            # escreve a mensagem na tela
            st.write(message)
            print(message)

            # salva a predição no JSON para cálculo das métricas de avaliação do sistema
            dh.save_prediction(passageiro)

        st.write('')
        # adiciona um botão para permitir o usuário realizar uma nova análise
        col1, col2, col3 = st.columns(3)
        with col2:
            new_test = st.button('Iniciar Nova Análise')

            # se o usuário pressionar no botão e já existe um passageiro, remove ele do cache
            if new_test and 'survived' in st.session_state:
                del st.session_state['survived']
                st.rerun()
        accuracy_predictions_on = st.toggle('Exibir acurácia')

        if accuracy_predictions_on:

            predictions = dh.get_all_predictions()

            num_total_predictions = len(predictions)

            accuracy_hist = [0]

            correct_predictions = 0

            for index, passageiro in enumerate(predictions):
                total = index + 1
                if passageiro['CorrectPrediction'] == True:
                    correct_predictions += 1

                temp_accuracy = correct_predictions / total if total else 0

                accuracy_hist.append(round(temp_accuracy, 2))

            accuracy = correct_predictions / num_total_predictions if num_total_predictions else 0

            st.metric(label='Acurácia', value=round(accuracy, 2))
            # TODO: usar o attr delta do st.metric para exibir a diferença na variação da acurácia

            # exibe o histórico da acurácia
            st.subheader("Histórico de acurácia")
            st.line_chart(accuracy_hist)

